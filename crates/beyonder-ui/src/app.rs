//! Application state machine — wires all subsystems together.

use anyhow::Result;
use beyonder_core::{
    AgentKind, Block, BlockContent, BlockKind, BlockStatus, CapabilitySet, Session,
    TerminalCell, TerminalOutput, TerminalRow,
};
use beyonder_config::{BeyonderConfig, ProviderConfig};
use beyonder_gpu::Renderer;
use beyonder_acp::client::AgentEvent;
use beyonder_runtime::{
    capability_broker::{ApprovalDecision, BrokerEvent, CapabilityBroker},
    supervisor::{AgentSupervisor, SupervisorEvent},
};
use beyonder_store::{BlockStore, SessionStore, Store};
use beyonder_terminal::{BlockBuilder, PtySession, TermGrid};
use beyonder_terminal::block_builder::BuildEvent;
use beyonder_terminal::pty::PtyEvent;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};
use winit::event::{ElementState, KeyEvent, MouseScrollDelta, WindowEvent};
use winit::keyboard::{Key, ModifiersState, NamedKey};
use winit::window::Window;

use crate::commands;
use crate::input_editor::InputEditor;
use crate::mode_detector::{detect_mode, InputMode};

// ── Context pill helpers ──────────────────────────────────────────────────────

fn current_conda_env() -> String {
    std::env::var("CONDA_DEFAULT_ENV").unwrap_or_else(|_| "—".to_string())
}

fn current_node_version() -> String {
    let output = std::process::Command::new("node")
        .arg("--version")
        .output();
    match output {
        Ok(o) if o.status.success() => {
            String::from_utf8_lossy(&o.stdout).trim().to_string()
        }
        _ => "—".to_string(),
    }
}

fn fetch_conda_envs() -> Vec<String> {
    let output = std::process::Command::new("conda")
        .args(["env", "list"])
        .output();
    match output {
        Ok(o) if o.status.success() => {
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
                .filter_map(|l| l.split_whitespace().next())
                .map(|s| s.to_string())
                .collect()
        }
        _ => vec![],
    }
}

fn fetch_node_versions() -> Vec<String> {
    let nvm_dir = std::env::var("NVM_DIR")
        .unwrap_or_else(|_| {
            std::env::var("HOME")
                .map(|h| format!("{h}/.nvm"))
                .unwrap_or_default()
        });
    let versions_path = std::path::Path::new(&nvm_dir).join("versions").join("node");
    if let Ok(entries) = std::fs::read_dir(&versions_path) {
        let mut versions: Vec<String> = entries
            .filter_map(|e| e.ok())
            .filter_map(|e| e.file_name().into_string().ok())
            .filter(|n| !n.starts_with('.'))
            .collect();
        versions.sort();
        versions
    } else {
        vec![]
    }
}

fn child_dirs(cwd: &std::path::Path) -> Vec<String> {
    if let Ok(entries) = std::fs::read_dir(cwd) {
        let mut dirs: Vec<String> = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().map(|t| t.is_dir()).unwrap_or(false))
            .filter_map(|e| e.file_name().into_string().ok())
            .filter(|n| !n.starts_with('.'))
            .collect();
        dirs.sort();
        dirs.truncate(30);
        dirs
    } else {
        vec![]
    }
}

fn block_to_text(block: &Block) -> String {
    match &block.content {
        BlockContent::ShellCommand { input, output, .. } => {
            let mut s = format!("$ {input}\n");
            for row in &output.rows {
                let line: String = row.cells.iter().map(|c| c.character).collect();
                s.push_str(&line);
                s.push('\n');
            }
            s
        }
        BlockContent::AgentMessage { content_blocks, .. } => {
            content_blocks.iter().map(|cb| match cb {
                beyonder_core::ContentBlock::Text { text } => text.clone(),
                beyonder_core::ContentBlock::Code { code, language } => {
                    format!("```{}\n{}\n```", language.as_deref().unwrap_or(""), code)
                }
                beyonder_core::ContentBlock::Thinking { thinking } => thinking.clone(),
            }).collect::<Vec<_>>().join("\n")
        }
        BlockContent::Text { text } => text.clone(),
        _ => String::new(),
    }
}

/// Serialize blocks[watermark..] into a terminal transcript for agent context.
/// ToolCall blocks are skipped — the agent already has them in its Ollama message history.
/// Long command outputs are truncated to keep the context token-efficient.
fn format_blocks_as_context(blocks: &[Block], watermark: usize) -> String {
    let new_blocks = &blocks[watermark.min(blocks.len())..];
    if new_blocks.is_empty() {
        return String::new();
    }

    let mut parts: Vec<String> = Vec::new();
    for block in new_blocks {
        match &block.content {
            BlockContent::ShellCommand { input, output, exit_code, .. } => {
                let mut entry = format!("$ {}\n", input.trim());
                let mut lines: Vec<String> = output.rows.iter().map(|row| {
                    let line: String = row.cells.iter().map(|c| c.character).collect();
                    line.trim_end().to_string()
                }).filter(|l| !l.is_empty()).collect();
                // Truncate very long outputs so context stays token-efficient.
                const MAX_OUTPUT_LINES: usize = 60;
                if lines.len() > MAX_OUTPUT_LINES {
                    let kept = MAX_OUTPUT_LINES / 2;
                    let omitted = lines.len() - kept * 2;
                    lines = lines[..kept].iter()
                        .chain(std::iter::once(&format!("... ({omitted} lines omitted) ...")))
                        .chain(lines[lines.len() - kept..].iter())
                        .cloned()
                        .collect();
                }
                entry.push_str(&lines.join("\n"));
                if let Some(code) = exit_code {
                    if *code != 0 {
                        entry.push_str(&format!("\n[exit {}]", code));
                    }
                }
                parts.push(entry);
            }
            BlockContent::AgentMessage { role, content_blocks } => {
                let label = match role {
                    beyonder_core::MessageRole::User => "User",
                    beyonder_core::MessageRole::Assistant => "Assistant",
                    _ => continue,
                };
                let text: String = content_blocks.iter().filter_map(|cb| match cb {
                    beyonder_core::ContentBlock::Text { text } => Some(text.as_str()),
                    _ => None,
                }).collect::<Vec<_>>().join("\n");
                if !text.trim().is_empty() {
                    parts.push(format!("[{label}]: {text}"));
                }
            }
            // ToolCall: skip — agent already has these in its own message history.
            // ApprovalRequest, FileEdit, PlanNode, Text: not relevant as agent context.
            _ => {}
        }
    }

    if parts.is_empty() {
        return String::new();
    }

    format!(
        "<terminal_context>\n{}\n</terminal_context>",
        parts.join("\n\n")
    )
}


/// User-selected input routing mode shown in the bottom-left pill.
#[derive(Debug, Clone, PartialEq)]
pub enum AppMode {
    /// Smart detection: `/` → command, `@name` → agent (power-user), else → shell.
    Auto,
    /// Everything goes to the shell PTY.
    Cmd,
    /// Everything goes to the active LLM agent.
    Agent,
}

impl AppMode {
    fn label(&self) -> &'static str {
        match self {
            AppMode::Auto => "auto",
            AppMode::Cmd => "cmd",
            AppMode::Agent => "agent",
        }
    }
}

pub struct App {
    pub renderer: Renderer,
    pub input: InputEditor,
    pub config: BeyonderConfig,
    pub store: Store,
    pub session: Session,
    pub supervisor: AgentSupervisor,
    pub capability_broker: CapabilityBroker,
    pub pty: Option<PtySession>,
    pub block_builder: BlockBuilder,
    pub term_grid: TermGrid,
    pub blocks: Vec<Block>,
    pub broker_rx: mpsc::Receiver<BrokerEvent>,
    pub supervisor_rx: mpsc::UnboundedReceiver<SupervisorEvent>,
    /// Index of the currently selected block (for copy).
    pub selected_block: Option<usize>,
    /// For ShellCommand blocks: true = output panel selected, false = cmd bar selected.
    pub selected_sub_output: bool,
    /// Current mouse cursor position in physical pixels.
    pub cursor_pos: (f32, f32),
    /// Currently active keyboard modifiers.
    pub modifiers: ModifiersState,
    /// True while a shell command (or TUI) is running — input is forwarded to PTY.
    pub command_running: bool,
    /// Previous TUI active state — used to detect transitions and resize PTY.
    prev_tui_active: bool,

    // Context pill state
    pub conda_envs: Vec<String>,
    pub node_versions: Vec<String>,
    pub current_conda: String,
    pub current_node: String,
    /// Which pill (0=conda, 1=node, 2=dir) currently has its dropdown open.
    pub open_pill_dropdown: Option<usize>,
    /// Items in the currently open dropdown.
    pub dropdown_items: Vec<String>,

    /// Active AI model name (used by /model command).
    pub active_model: String,
    /// Active AI provider (used by /provider command).
    pub active_provider: String,
    /// Current input routing mode (shown as bottom-left pill).
    pub app_mode: AppMode,
    /// Set to true by /quit or /exit — checked by the event loop.
    pub should_quit: bool,
    /// In Auto mode, if a shell command exits with code 127, route it to the agent instead.
    pending_agent_fallback: Option<String>,
    /// Index into self.blocks up to which context has already been sent to the agent.
    /// Blocks at index >= this are "new" and will be included in the next agent prompt.
    agent_context_watermark: usize,
    /// Currently executing tool per agent: agent_id → tool_name.
    agent_running_tool: std::collections::HashMap<beyonder_core::AgentId, String>,
    /// Throttle for blocking env-probe subprocesses (node --version, conda env list).
    /// Only re-run them if this much time has elapsed since the last probe.
    last_env_probe: std::time::Instant,
}

impl App {
    pub async fn new(window: Arc<Window>, config: BeyonderConfig) -> Result<Self> {
        let renderer = Renderer::new(Arc::clone(&window)).await?;

        // Open data store.
        std::fs::create_dir_all(&config.data_dir)?;
        let store = Store::open(&config.db_path())?;

        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/"));
        let session = Session::new(cwd.clone());
        let session_id = session.id.clone();

        let session_store = SessionStore::new(&store);
        session_store.insert(&session)?;

        let (broker_tx, broker_rx) = mpsc::channel(64);
        let (supervisor_tx, supervisor_rx) = mpsc::unbounded_channel();

        let capability_broker = CapabilityBroker::new(broker_tx);
        let supervisor = AgentSupervisor::new(supervisor_tx);

        // Calculate PTY dimensions from the renderer — single source of truth.
        let (pty_cols, pty_rows) = renderer.terminal_grid_size();

        let mut block_builder = BlockBuilder::new(session_id.clone(), cwd.clone());
        block_builder.set_grid_size(pty_cols as usize, pty_rows as usize);

        let term_grid = TermGrid::new(pty_cols as usize, pty_rows as usize);

        // Spawn the shell PTY.
        let shell_env = std::env::var("SHELL").unwrap_or_else(|_| "/bin/zsh".to_string());
        let shell = config
            .shell
            .program
            .as_deref()
            .unwrap_or(&shell_env);

        let pty = match PtySession::spawn_sized(session_id.clone(), shell, &cwd, &[], pty_cols, pty_rows) {
            Ok(p) => {
                info!("Shell PTY spawned");
                Some(p)
            }
            Err(e) => {
                warn!("Failed to spawn PTY: {e}");
                None
            }
        };

        let active_model = config.model.clone();
        let active_provider = config.provider.name().to_string();

        Ok(Self {
            renderer,
            input: InputEditor::new(),
            config,
            store,
            session,
            supervisor,
            capability_broker,
            pty,
            block_builder,
            term_grid,
            blocks: vec![],
            broker_rx,
            supervisor_rx,
            selected_block: None,
            selected_sub_output: false,
            cursor_pos: (0.0, 0.0),
            modifiers: ModifiersState::empty(),
            command_running: false,
            prev_tui_active: false,
            current_conda: current_conda_env(),
            current_node: current_node_version(),
            conda_envs: fetch_conda_envs(),
            node_versions: fetch_node_versions(),
            open_pill_dropdown: None,
            dropdown_items: vec![],
            active_model,
            active_provider,
            app_mode: AppMode::Auto,
            should_quit: false,
            pending_agent_fallback: None,
            agent_context_watermark: 0,
            agent_running_tool: std::collections::HashMap::new(),
            // Use a past instant so the first post-command probe runs after startup.
            last_env_probe: std::time::Instant::now() - std::time::Duration::from_secs(60),
        })
    }

    /// Handle a winit window event.
    pub async fn handle_window_event(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::Resized(size) => {
                self.renderer.resize(size.width, size.height);
                // Use full-window dimensions when a TUI app is active (bar is hidden),
                // above-bar dimensions otherwise.
                let (cols, rows) = if self.term_grid.tui_active() {
                    self.renderer.tui_grid_size()
                } else {
                    self.renderer.terminal_grid_size()
                };
                self.term_grid.resize(cols as usize, rows as usize);
                self.block_builder.set_grid_size(cols as usize, rows as usize);
                if let Some(pty) = &self.pty {
                    let _ = pty.resize(rows, cols);
                }
                false
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                self.renderer.set_scale_factor(*scale_factor);
                false
            }
            WindowEvent::ModifiersChanged(mods) => {
                self.modifiers = mods.state();
                false
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_pos = (position.x as f32, position.y as f32);
                // Update command palette hover.
                self.renderer.cmd_palette_hovered =
                    self.renderer.cmd_palette_hit(self.cursor_pos.0, self.cursor_pos.1);
                // Update dropdown hover highlight.
                if self.open_pill_dropdown.is_some() {
                    let hovered = self.renderer.dropdown_hover_at(self.cursor_pos.0, self.cursor_pos.1);
                    if let Some((pill_idx, ref items, ref mut h)) = self.renderer.open_dropdown {
                        *h = hovered;
                    }
                }
                false
            }
            WindowEvent::MouseInput { state: ElementState::Pressed, button: winit::event::MouseButton::Left, .. } => {
                self.handle_click(self.cursor_pos);
                false
            }
            WindowEvent::KeyboardInput { event, .. } => {
                self.handle_key(event).await;
                self.should_quit
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => -y * 20.0,
                    MouseScrollDelta::PixelDelta(pos) => -pos.y as f32,
                };
                self.renderer.scroll(scroll);
                false
            }
            WindowEvent::CloseRequested => true,
            _ => false,
        }
    }

    fn key_to_pty_bytes(&self, event: &KeyEvent) -> Option<Vec<u8>> {
        use winit::keyboard::NamedKey;
        let app_cursor = self.term_grid.app_cursor_mode();
        let shift = self.modifiers.shift_key();
        let ctrl = self.modifiers.control_key();
        let alt = self.modifiers.alt_key();

        // Compute xterm modifier parameter: 1 + shift + alt*2 + ctrl*4.
        // Only emit the modifier suffix when at least one modifier is active.
        let xterm_mod = 1u8 + (shift as u8) + (alt as u8 * 2) + (ctrl as u8 * 4);

        match &event.logical_key {
            Key::Named(NamedKey::ArrowUp) => {
                if xterm_mod > 1 {
                    Some(format!("\x1b[1;{xterm_mod}A").into_bytes())
                } else if app_cursor {
                    Some(b"\x1bOA".to_vec())
                } else {
                    Some(b"\x1b[A".to_vec())
                }
            }
            Key::Named(NamedKey::ArrowDown) => {
                if xterm_mod > 1 {
                    Some(format!("\x1b[1;{xterm_mod}B").into_bytes())
                } else if app_cursor {
                    Some(b"\x1bOB".to_vec())
                } else {
                    Some(b"\x1b[B".to_vec())
                }
            }
            Key::Named(NamedKey::ArrowRight) => {
                if xterm_mod > 1 {
                    Some(format!("\x1b[1;{xterm_mod}C").into_bytes())
                } else if app_cursor {
                    Some(b"\x1bOC".to_vec())
                } else {
                    Some(b"\x1b[C".to_vec())
                }
            }
            Key::Named(NamedKey::ArrowLeft) => {
                if xterm_mod > 1 {
                    Some(format!("\x1b[1;{xterm_mod}D").into_bytes())
                } else if app_cursor {
                    Some(b"\x1bOD".to_vec())
                } else {
                    Some(b"\x1b[D".to_vec())
                }
            }
            Key::Named(NamedKey::Enter) => Some(b"\r".to_vec()),
            Key::Named(NamedKey::Backspace) => Some(b"\x7f".to_vec()),
            Key::Named(NamedKey::Escape) => Some(b"\x1b".to_vec()),
            Key::Named(NamedKey::Tab) => {
                if shift {
                    Some(b"\x1b[Z".to_vec()) // Shift+Tab = reverse-tab
                } else {
                    Some(b"\t".to_vec())
                }
            }
            Key::Named(NamedKey::Space) => Some(b" ".to_vec()),
            Key::Named(NamedKey::Delete) => Some(b"\x1b[3~".to_vec()),
            Key::Named(NamedKey::Home) => {
                if xterm_mod > 1 {
                    Some(format!("\x1b[1;{xterm_mod}H").into_bytes())
                } else {
                    Some(b"\x1b[H".to_vec())
                }
            }
            Key::Named(NamedKey::End) => {
                if xterm_mod > 1 {
                    Some(format!("\x1b[1;{xterm_mod}F").into_bytes())
                } else {
                    Some(b"\x1b[F".to_vec())
                }
            }
            Key::Named(NamedKey::PageUp) => Some(b"\x1b[5~".to_vec()),
            Key::Named(NamedKey::PageDown) => Some(b"\x1b[6~".to_vec()),
            // F1–F4: SS3 sequences; F5–F12: CSI sequences.
            Key::Named(NamedKey::F1)  => Some(b"\x1bOP".to_vec()),
            Key::Named(NamedKey::F2)  => Some(b"\x1bOQ".to_vec()),
            Key::Named(NamedKey::F3)  => Some(b"\x1bOR".to_vec()),
            Key::Named(NamedKey::F4)  => Some(b"\x1bOS".to_vec()),
            Key::Named(NamedKey::F5)  => Some(b"\x1b[15~".to_vec()),
            Key::Named(NamedKey::F6)  => Some(b"\x1b[17~".to_vec()),
            Key::Named(NamedKey::F7)  => Some(b"\x1b[18~".to_vec()),
            Key::Named(NamedKey::F8)  => Some(b"\x1b[19~".to_vec()),
            Key::Named(NamedKey::F9)  => Some(b"\x1b[20~".to_vec()),
            Key::Named(NamedKey::F10) => Some(b"\x1b[21~".to_vec()),
            Key::Named(NamedKey::F11) => Some(b"\x1b[23~".to_vec()),
            Key::Named(NamedKey::F12) => Some(b"\x1b[24~".to_vec()),
            Key::Character(s) => {
                if ctrl {
                    s.chars().next().and_then(|ch| {
                        let lo = ch.to_ascii_lowercase();
                        let byte: u8 = if lo.is_ascii_alphabetic() {
                            // Ctrl+a..z → 0x01..0x1a via AND with 0x1f.
                            lo as u8 & 0x1f
                        } else {
                            match ch {
                                ' ' => 0x00,            // Ctrl+Space = NUL
                                '[' => 0x1b,            // Ctrl+[ = ESC
                                '\\' => 0x1c,           // Ctrl+\ = FS
                                ']' => 0x1d,            // Ctrl+] = GS
                                '^' => 0x1e,            // Ctrl+^ = RS
                                '_' => 0x1f,            // Ctrl+_ = US
                                _ => return None,
                            }
                        };
                        Some(vec![byte])
                    })
                } else if alt {
                    // Alt/Meta: ESC-prefix the UTF-8 bytes.
                    let mut bytes = vec![0x1b_u8];
                    bytes.extend_from_slice(s.as_bytes());
                    Some(bytes)
                } else {
                    Some(s.as_bytes().to_vec())
                }
            }
            _ => None,
        }
    }

    fn handle_click(&mut self, pos: (f32, f32)) {
        // Mode switcher click — cycle auto → cmd → agent → auto.
        if self.renderer.mode_pill_hit(pos.0, pos.1) {
            self.app_mode = match self.app_mode {
                AppMode::Auto => AppMode::Cmd,
                AppMode::Cmd => AppMode::Agent,
                AppMode::Agent => AppMode::Auto,
            };
            return;
        }
        // Command palette click: fill the selected command into the input.
        if let Some(row) = self.renderer.cmd_palette_hit(pos.0, pos.1) {
            if let Some(ref cmds) = self.renderer.command_palette.clone() {
                if let Some((usage, _)) = cmds.get(row) {
                    // Fill input with the usage text, strip description args — keep the first token.
                    let filled = usage.split_whitespace().next().unwrap_or(usage).to_string();
                    self.input.set_text(filled);
                }
            }
            return;
        }
        // Dropdown item hit takes priority.
        if let Some(item_idx) = self.renderer.dropdown_hit(pos.0, pos.1) {
            self.select_dropdown_item(item_idx);
            return;
        }
        // Pill hit: toggle dropdown.
        if let Some(pill_idx) = self.renderer.pill_hit(pos.0, pos.1) {
            if self.open_pill_dropdown == Some(pill_idx) {
                // Toggle off.
                self.open_pill_dropdown = None;
                self.dropdown_items = vec![];
                self.renderer.open_dropdown = None;
            } else {
                self.open_pill_dropdown_for(pill_idx);
            }
            return;
        }
        // Close any open dropdown if clicking elsewhere.
        if self.open_pill_dropdown.is_some() {
            self.open_pill_dropdown = None;
            self.dropdown_items = vec![];
            self.renderer.open_dropdown = None;
            return;
        }
        if let Some((idx, _)) = self.renderer.block_hit_at(pos.1) {
            // Clicking a ToolCall output block toggles it open/closed.
            if let Some(block) = self.blocks.get(idx) {
                if let beyonder_core::BlockContent::ToolCall { output, .. } = &block.content {
                    if output.is_some() {
                        self.renderer.toggle_collapsed(&block.id);
                        return;
                    }
                }
            }
            let is_output = self.renderer.block_hit_at(pos.1).map(|(_, o)| o).unwrap_or(false);
            self.selected_block = Some(idx);
            self.selected_sub_output = is_output;
            self.renderer.selected_block = Some(idx);
            self.renderer.selected_sub_output = is_output;
        } else {
            self.selected_block = None;
            self.selected_sub_output = false;
            self.renderer.selected_block = None;
            self.renderer.selected_sub_output = false;
        }
    }

    fn open_pill_dropdown_for(&mut self, pill_idx: usize) {
        let items = match pill_idx {
            0 => self.conda_envs.clone(),
            1 => self.node_versions.clone(),
            2 => child_dirs(&self.block_builder.cwd),
            _ => vec![],
        };
        self.open_pill_dropdown = Some(pill_idx);
        self.dropdown_items = items.clone();
        self.renderer.open_dropdown = Some((pill_idx, items, None));
    }

    fn select_dropdown_item(&mut self, idx: usize) {
        let item = self.dropdown_items.get(idx).cloned();
        if let Some(item) = item {
            let cmd = match self.open_pill_dropdown {
                Some(0) => format!("conda activate {}\n", item),
                Some(1) => format!("nvm use {}\n", item),
                Some(2) => format!("cd {}\n", item),
                _ => return,
            };
            self.write_to_pty(&cmd);
        }
        self.open_pill_dropdown = None;
        self.dropdown_items = vec![];
        self.renderer.open_dropdown = None;
    }

    fn write_to_pty(&mut self, text: &str) {
        if let Some(pty) = &mut self.pty {
            let _ = pty.write(text.as_bytes());
        }
    }

    async fn handle_key(&mut self, event: &KeyEvent) {
        if event.state != ElementState::Pressed {
            return;
        }
        // TUI/interactive mode: forward all keys directly to PTY as raw bytes.
        // Covers both alt-screen TUIs (vim, htop) and raw-mode apps (claude, React Ink).
        if self.term_grid.tui_active() || self.block_builder.is_running_command() {
            if let Some(bytes) = self.key_to_pty_bytes(event) {
                if let Some(pty) = &mut self.pty {
                    let _ = pty.write(&bytes);
                }
            }
            return;
        }
        // Cmd+C — copy selected block text to clipboard.
        #[cfg(target_os = "macos")]
        if self.modifiers.super_key() {
            if let Key::Character(s) = &event.logical_key {
                if s.as_str() == "c" {
                    self.copy_selection();
                    return;
                }
            }
        }
        match &event.logical_key {
            Key::Named(NamedKey::Enter) => {
                if !self.input.is_empty() {
                    let text = self.input.submit();
                    self.input.push_history(text.clone());
                    self.route_input(text).await;
                }
            }
            Key::Named(NamedKey::Tab) => {
                // Shift+Tab cycles mode: Auto → Agent → Cmd → Auto (backwards).
                // Plain Tab cycles forward: Auto → Cmd → Agent → Auto.
                if self.modifiers.shift_key() {
                    self.app_mode = match self.app_mode {
                        AppMode::Auto  => AppMode::Agent,
                        AppMode::Agent => AppMode::Cmd,
                        AppMode::Cmd   => AppMode::Auto,
                    };
                } else {
                    self.app_mode = match self.app_mode {
                        AppMode::Auto  => AppMode::Cmd,
                        AppMode::Cmd   => AppMode::Agent,
                        AppMode::Agent => AppMode::Auto,
                    };
                }
            }
            Key::Named(NamedKey::Backspace) => {
                self.input.delete_backward();
            }
            Key::Named(NamedKey::Delete) => {
                self.input.delete_forward();
            }
            Key::Named(NamedKey::ArrowUp) => { self.input.history_prev(); }
            Key::Named(NamedKey::ArrowDown) => { self.input.history_next(); }
            Key::Named(NamedKey::ArrowLeft) => self.input.move_left(),
            Key::Named(NamedKey::ArrowRight) => self.input.move_right(),
            Key::Named(NamedKey::Home) => self.input.move_home(),
            Key::Named(NamedKey::End) => self.input.move_end(),
            Key::Named(NamedKey::Space) => {
                self.input.insert(' ');
            }
            Key::Character(s) => {
                for ch in s.chars() {
                    self.input.insert(ch);
                }
            }
            _ => {}
        }
    }

    fn copy_selection(&self) {
        let Some(idx) = self.selected_block else { return };
        let Some(block) = self.blocks.get(idx) else { return };
        let text = match &block.content {
            BlockContent::ShellCommand { input, output, .. } => {
                if self.selected_sub_output {
                    // Copy only the output rows.
                    output.rows.iter()
                        .map(|row| row.cells.iter().map(|c| c.character).collect::<String>())
                        .collect::<Vec<_>>()
                        .join("\n")
                } else {
                    // Copy only the command.
                    input.clone()
                }
            }
            _ => block_to_text(block),
        };
        if text.is_empty() { return; }
        if let Ok(mut clipboard) = arboard::Clipboard::new() {
            let _ = clipboard.set_text(text);
        }
    }

    async fn route_input(&mut self, text: String) {
        // Slash commands always work regardless of mode.
        if let InputMode::Command { .. } = detect_mode(&text) {
            return self.handle_command(&text).await;
        }
        match self.app_mode {
            AppMode::Cmd => self.send_to_shell(text).await,
            AppMode::Agent => self.prompt_active_llm(text).await,
            AppMode::Auto => match detect_mode(&text) {
                InputMode::Agent { name } => {
                    let prompt = text
                        .trim_start()
                        .strip_prefix(&format!("@{name}"))
                        .unwrap_or(&text)
                        .trim()
                        .to_string();
                    self.prompt_agent(&name, prompt).await;
                }
                InputMode::Shell => self.send_to_shell(text).await,
                InputMode::Command { .. } => self.handle_command(&text).await,
            },
        }
    }

    /// Route text directly to the active LLM (used in Agent mode).
    async fn prompt_active_llm(&mut self, text: String) {
        let name = self.active_provider.clone();
        self.prompt_agent(&name, text).await;
    }

    async fn send_to_shell(&mut self, text: String) {
        let mut cmd = text;
        cmd.push('\n');
        if let Some(pty) = &mut self.pty {
            if let Err(e) = pty.write(cmd.as_bytes()) {
                error!("PTY write error: {e}");
            }
        }
    }

    async fn prompt_agent(&mut self, name: &str, prompt: String) {
        // Serialize all blocks the agent hasn't seen yet into a terminal transcript
        // and prepend as context. The watermark tracks how far we've sent.
        // We snapshot the watermark before push_human_prompt_block so we don't
        // include the human-prompt block we're about to add in this turn's context.
        let context = format_blocks_as_context(&self.blocks, self.agent_context_watermark);
        // Advance the watermark to current block count (human prompt block not yet added).
        self.agent_context_watermark = self.blocks.len();

        let full_prompt = if context.is_empty() {
            prompt.clone()
        } else {
            format!("{context}\n\n{prompt}")
        };

        // Echo the prompt as a Human block so the user can see what was sent.
        self.push_human_prompt_block(prompt.clone());

        // Find agent by name or spawn one.
        let agents = self.supervisor.list_agents();
        let existing_id = agents
            .iter()
            .find(|a| a.name == name)
            .map(|a| a.id.clone());

        if let Some(agent_id) = existing_id {
            self.push_pending_agent_block(agent_id.clone());
            if let Err(e) = self.supervisor.prompt_agent(&agent_id, &full_prompt) {
                error!("Failed to prompt agent {name}: {e}");
            }
        } else {
            let caps = CapabilitySet::default_coding_agent(
                self.session.working_directory.clone(),
            );
            let kind = match &self.config.provider {
                ProviderConfig::Ollama { base_url, api_key_env } => {
                    // Env var takes precedence over config for cloud detection.
                    let (base_url, api_key_env) = if std::env::var("OLLAMA_API_KEY").is_ok() {
                        ("https://ollama.com".to_string(), Some("OLLAMA_API_KEY".to_string()))
                    } else {
                        (base_url.clone(), api_key_env.clone())
                    };
                    AgentKind::Ollama { base_url, model: self.active_model.clone(), api_key_env }
                }
                ProviderConfig::LlamaCpp { base_url, api_key_env } => {
                    AgentKind::LlamaCpp {
                        base_url: base_url.clone(),
                        model: self.active_model.clone(),
                        api_key_env: api_key_env.clone(),
                    }
                }
                ProviderConfig::Mlx { base_url, api_key_env } => {
                    AgentKind::Mlx {
                        base_url: base_url.clone(),
                        model: self.active_model.clone(),
                        api_key_env: api_key_env.clone(),
                    }
                }
            };
            match self
                .supervisor
                .spawn_agent(name, kind, caps)
                .await
            {
                Ok(agent_id) => {
                    self.capability_broker.register_agent(
                        agent_id.clone(),
                        CapabilitySet::default_coding_agent(
                            self.session.working_directory.clone(),
                        ),
                    );
                    self.push_pending_agent_block(agent_id.clone());
                    if let Err(e) = self.supervisor.prompt_agent(&agent_id, &full_prompt) {
                        error!("Failed to prompt new agent: {e}");
                    }
                }
                Err(e) => {
                    error!("Failed to spawn agent {name}: {e}");
                }
            }
        }
    }

    async fn handle_command(&mut self, text: &str) {
        let parts: Vec<&str> = text.split_whitespace().collect();
        match parts.as_slice() {
            // ── Generic ──────────────────────────────────────────────────────
            ["/clear"] => {
                self.blocks.clear();
                self.renderer.blocks.clear();
                self.renderer.running_block_idx = None;
                self.agent_context_watermark = 0;
                self.supervisor.reset_all_conversations();
                info!("Block stream cleared");
            }
            ["/help"] => {
                let lines: Vec<String> = commands::COMMANDS
                    .iter()
                    .map(|c| format!("{:<30} {}", c.usage, c.description))
                    .collect();
                self.push_text_block(lines.join("\n"));
            }
            ["/quit"] | ["/exit"] => {
                self.should_quit = true;
            }
            ["/scroll", "top"] => {
                self.renderer.viewport.scroll_to_top();
            }
            ["/scroll", "bottom"] => {
                self.renderer.viewport.scroll_to_bottom();
            }
            ["/font", size_str] => {
                if let Ok(size) = size_str.parse::<f32>() {
                    if size >= 8.0 && size <= 48.0 {
                        self.renderer.font_size = size;
                        info!("Font size set to {size}");
                    } else {
                        warn!("Font size must be between 8 and 48");
                    }
                } else {
                    warn!("Invalid font size: {size_str}");
                }
            }

            // ── Agent ─────────────────────────────────────────────────────────
            ["/agent", "list"] => {
                let agents = self.supervisor.list_agents();
                if agents.is_empty() {
                    self.push_text_block("No agents running.".to_string());
                } else {
                    let lines: Vec<String> = agents.iter()
                        .map(|a| format!("{:?}  {}  {:?}", a.id, a.name, a.state))
                        .collect();
                    self.push_text_block(lines.join("\n"));
                }
            }
            ["/agent", "kill", id] => {
                let agent_id = beyonder_core::AgentId(id.to_string());
                if let Err(e) = self.supervisor.kill_agent(&agent_id).await {
                    error!("Kill failed: {e}");
                }
            }

            // ── Beyonder-specific ─────────────────────────────────────────────
            ["/model", name] => {
                self.active_model = name.to_string();
                self.config.model = name.to_string();
                if let Err(e) = self.config.save() {
                    warn!("Failed to save config: {e}");
                }
                // Kill any live Ollama agent so the next prompt re-spawns with the new model.
                let ids: Vec<_> = self.supervisor.list_agents()
                    .iter()
                    .filter(|a| matches!(a.kind, AgentKind::Ollama { .. }))
                    .map(|a| a.id.clone())
                    .collect();
                for id in ids {
                    let _ = self.supervisor.kill_agent(&id).await;
                }
                self.push_text_block(format!("Model set to: {}", name));
                info!("Active model: {name}");
            }
            ["/model"] => {
                self.push_text_block(format!("Current model: {}", self.active_model));
            }
            ["/provider", name] => {
                // Preserve existing base_url if the provider kind hasn't changed;
                // otherwise fall back to the default URL for the new provider.
                let new_provider = if self.config.provider.name() == *name {
                    self.config.provider.clone()
                } else {
                    ProviderConfig::from_name(name)
                };
                self.active_provider = name.to_string();
                self.config.provider = new_provider;
                if let Err(e) = self.config.save() {
                    warn!("Failed to save config: {e}");
                }
                self.push_text_block(format!("Provider set to: {}", name));
                info!("Active provider: {name}");
            }
            ["/provider"] => {
                self.push_text_block(format!("Current provider: {}", self.active_provider));
            }
            ["/mode", mode_name] => {
                self.app_mode = match *mode_name {
                    "cmd" => AppMode::Cmd,
                    "agent" => AppMode::Agent,
                    _ => AppMode::Auto,
                };
                self.push_text_block(format!("Input mode: {}", self.app_mode.label()));
            }
            ["/mode"] => {
                self.push_text_block(format!("Current mode: {}", self.app_mode.label()));
            }
            ["/session", "new"] => {
                let cwd = self.block_builder.cwd.clone();
                let new_session = Session::new(cwd);
                self.session = new_session;
                self.blocks.clear();
                self.renderer.blocks.clear();
                info!("New session started: {:?}", self.session.id);
            }
            ["/session", "list"] => {
                self.push_text_block(format!("Current session: {:?}", self.session.id));
            }
            ["/theme", _name] => {
                self.push_text_block("Theme switching is not yet implemented.".to_string());
            }

            _ => {
                warn!("Unknown command: {text}");
                self.push_text_block(format!("Unknown command: {text}\nType /help for a list of commands."));
            }
        }
    }

    /// Push a Human/User prompt echo block so the user can see what they sent.
    fn push_human_prompt_block(&mut self, text: String) {
        use beyonder_core::{BlockContent, BlockId, BlockKind, BlockStatus, ContentBlock, MessageRole, ProvenanceChain};
        let now = chrono::Utc::now();
        let block = Block {
            id: BlockId::new(),
            kind: BlockKind::Human,
            parent_id: None,
            agent_id: None,
            session_id: self.session.id.clone(),
            status: BlockStatus::Completed,
            content: BlockContent::AgentMessage {
                role: MessageRole::User,
                content_blocks: vec![ContentBlock::Text { text }],
            },
            created_at: now,
            updated_at: now,
            provenance: ProvenanceChain::default(),
        };
        self.blocks.push(block.clone());
        self.renderer.blocks.push(block);
        self.renderer.viewport.scroll_to_bottom();
    }

    /// Push an empty Running agent block immediately so the spinner appears before streaming starts.
    fn push_pending_agent_block(&mut self, agent_id: beyonder_core::AgentId) {
        use beyonder_core::{BlockContent, BlockId, BlockKind, BlockStatus, MessageRole, ProvenanceChain};
        let now = chrono::Utc::now();
        let mut block = Block {
            id: BlockId::new(),
            kind: BlockKind::Agent,
            parent_id: None,
            agent_id: Some(agent_id),
            session_id: self.session.id.clone(),
            status: BlockStatus::Running,
            content: BlockContent::AgentMessage {
                role: MessageRole::Assistant,
                content_blocks: vec![],
            },
            created_at: now,
            updated_at: now,
            provenance: ProvenanceChain::default(),
        };
        // No DB write — Running blocks are ephemeral; persisted on completion.
        self.blocks.push(block.clone());
        self.renderer.blocks = self.blocks.clone();
        self.renderer.scroll_to_bottom();
    }

    /// Push a plain text block into the block stream (used for command output).
    fn push_text_block(&mut self, text: String) {
        use beyonder_core::{BlockContent, BlockId, BlockKind, BlockStatus, ProvenanceChain};
        let now = chrono::Utc::now();
        let block = Block {
            id: BlockId::new(),
            kind: BlockKind::System,
            parent_id: None,
            agent_id: None,
            session_id: self.session.id.clone(),
            status: BlockStatus::Completed,
            content: BlockContent::Text { text },
            created_at: now,
            updated_at: now,
            provenance: ProvenanceChain::default(),
        };
        self.blocks.push(block.clone());
        self.renderer.blocks.push(block);
        self.renderer.viewport.scroll_to_bottom();
    }

    /// Poll async channels and update state. Call on each event loop tick.
    pub async fn tick(&mut self) {
        // Drain PTY events.
        let mut pty_output: Vec<Vec<u8>> = vec![];
        let mut pty_exited: Option<Option<u32>> = None;
        if let Some(pty) = &mut self.pty {
            while let Ok(event) = pty.event_rx.try_recv() {
                match event {
                    PtyEvent::Output(bytes) => pty_output.push(bytes),
                    PtyEvent::Exited(code) => {
                        info!("Shell exited: {:?}", code);
                        pty_exited = Some(code);
                    }
                }
            }
        }

        let had_pty_output = !pty_output.is_empty();
        for bytes in pty_output {
            self.term_grid.feed(&bytes);
            for event in self.block_builder.feed(&bytes) {
                self.handle_build_event(event);
            }
        }

        // If the PTY process died, force-complete any running block.
        if let Some(exit_code) = pty_exited {
            if let Some(event) = self.block_builder.force_complete(exit_code) {
                self.handle_build_event(event);
            }
            // Clear live terminal cells so no stale grid is shown.
            self.renderer.tui_cells = vec![];
        }

        // Only sync live terminal cells when new data arrived — avoids stale redraws.
        if had_pty_output {
            let is_live = self.block_builder.is_running_command() || self.term_grid.tui_active();
            if is_live {
                self.renderer.tui_cells = self.term_grid.cell_grid();
                self.renderer.tui_cursor = self.term_grid.cursor_pos();
            }
        }

        // Detect TUI active/inactive transitions and resize PTY so the app
        // gets the correct grid dimensions (full window vs. above-bar).
        let tui_now = self.term_grid.tui_active();
        if tui_now != self.prev_tui_active {
            self.prev_tui_active = tui_now;
            let (cols, rows) = if tui_now {
                self.renderer.tui_grid_size()
            } else {
                self.renderer.terminal_grid_size()
            };
            self.term_grid.resize(cols as usize, rows as usize);
            if let Some(pty) = &self.pty {
                let _ = pty.resize(rows, cols);
            }
        }

        // Drain supervisor events. Batch renderer sync to once after the loop.
        let mut agent_events_received = false;
        while let Ok(event) = self.supervisor_rx.try_recv() {
            match event {
                SupervisorEvent::AgentEvent { agent_id, event } => {
                    agent_events_received = true;
                    match event {
                        AgentEvent::TextDelta(text) => {
                            // New text means the tool (if any) has completed.
                            self.agent_running_tool.remove(&agent_id);
                            self.append_agent_text(&agent_id, &text);
                        }
                        AgentEvent::ToolCallRequested { id, name, input } => {
                            info!(tool = %name, "Agent tool call requested");
                            self.agent_running_tool.insert(agent_id.clone(), name.clone());
                            // Finalize any in-flight agent text block (empty → removed).
                            self.finalize_agent_block(&agent_id);
                            // Show the tool call immediately so the user sees what's running.
                            self.push_tool_call_block(&agent_id, id, name, input);
                        }
                        AgentEvent::ToolResult { id, name: _, output, is_error } => {
                            // Don't clear agent_running_tool here — the LLM hasn't
                            // resumed yet. Keep the spinner/tool label visible until
                            // the next TextDelta or ToolCallRequested arrives.
                            self.complete_tool_call_block(&id, output, is_error);
                        }
                        AgentEvent::TurnComplete { ref stop_reason } => {
                            info!(%agent_id, %stop_reason, "ui: TurnComplete received");
                            self.agent_running_tool.remove(&agent_id);
                            debug!(%agent_id, "ui: calling finalize_agent_block");
                            self.finalize_agent_block(&agent_id);
                            debug!(%agent_id, "ui: finalize_agent_block done");
                        }
                        AgentEvent::Error(e) => {
                            error!(%agent_id, "ui: agent error: {e}");
                        }
                    }
                }
                SupervisorEvent::AgentSpawned(info) => {
                    info!(name = %info.name, "Agent spawned");
                }
                _ => {}
            }
        }
        if agent_events_received {
            self.renderer.blocks = self.blocks.clone();
            self.renderer.agent_running_tool = self.agent_running_tool.clone();
        }

        // Drain broker events.
        while let Ok(event) = self.broker_rx.try_recv() {
            if let BrokerEvent::ApprovalRequired(block) = event {
                self.add_block(block);
            }
        }

        // Auto-mode 127 fallback: fire after all block events are drained so the
        // block removal above has already synced to the renderer before we add new blocks.
        if let Some(cmd) = self.pending_agent_fallback.take() {
            if !cmd.is_empty() {
                let name = self.active_provider.clone();
                self.prompt_agent(&name, cmd).await;
            }
        }

    }

    fn handle_build_event(&mut self, event: BuildEvent) {
        match event {
            BuildEvent::Block(block) => {
                // Don't add a Running block for "clear" — let it run silently in the PTY.
                // The LiveUpdate below will wipe the block list when it completes.
                if let BlockContent::ShellCommand { ref input, .. } = block.content {
                    if input.trim() == "clear" {
                        return;
                    }
                }
                // Reset TermGrid so the live view only shows output from THIS command.
                // Without this, the TermGrid accumulates all terminal history since
                // startup — running `claude` would show every previous command too.
                // Skip reset when a TUI app is active — OSC markers from sub-shells
                // (e.g. `:!ls` inside nvim) must not wipe the alt-screen state.
                if !self.term_grid.tui_active() {
                    self.term_grid.reset();
                }
                self.renderer.tui_cells = vec![];
                self.add_block(block);
            }
            BuildEvent::LiveUpdate { block_id, content } => {
                // "clear" completes: wipe all blocks and reset viewport.
                if let BlockContent::ShellCommand { ref input, .. } = content {
                    if input.trim() == "clear" {
                        self.blocks.clear();
                        self.renderer.blocks.clear();
                        self.renderer.tui_cells = vec![];
                        self.renderer.selected_block = None;
                        self.selected_block = None;
                        self.renderer.viewport.scroll_offset = 0.0;
                        self.agent_context_watermark = 0;
                        self.supervisor.reset_all_conversations();
                        return;
                    }
                }

                // Use the block_builder's parsed content directly — it has already
                // run parse_ansi_output on the raw PTY bytes to produce clean text.
                // The old TermGrid-snapshot approach polluted regular commands with
                // stale terminal state left over from TUI apps (vim, claude, etc.).

                // Auto mode: exit 127 ("command not found") → silently reroute to agent.
                // Remove the block so no error appears, then queue the input as an agent prompt.
                if self.app_mode == AppMode::Auto {
                    if let BlockContent::ShellCommand { ref input, exit_code: Some(127), .. } = content {
                        let cmd = input.trim().to_string();
                        // Remove only the failed shell block — shell commands have no
                        // separate preceding human-prompt block, so removing more would
                        // silently delete the previous command's block.
                        self.blocks.retain(|b| b.id != block_id);
                        self.renderer.blocks = self.blocks.clone();
                        self.renderer.tui_cells = vec![];
                        self.pending_agent_fallback = Some(cmd);
                        return;
                    }
                }

                if let Some(b) = self.blocks.iter_mut().find(|b| b.id == block_id) {
                    b.content = content;
                    b.status = BlockStatus::Completed;
                    let _ = BlockStore::new(&self.store).update(b);
                }
                self.renderer.blocks = self.blocks.clone();
                // Clear live cells — the command is done.
                self.renderer.tui_cells = vec![];
                // Refresh pill labels — user may have run conda activate / nvm use.
                // node --version and conda env list are blocking subprocesses.
                // Only re-run them every 30 s to avoid freezing the event loop.
                if self.last_env_probe.elapsed().as_secs() >= 30 {
                    self.current_conda = current_conda_env();
                    self.current_node = current_node_version();
                    self.last_env_probe = std::time::Instant::now();
                }
            }
        }
    }

    fn add_block(&mut self, block: Block) {
        // Persist to DB.
        let _ = BlockStore::new(&self.store).insert(&block);
        self.blocks.push(block);
        self.renderer.blocks = self.blocks.clone();
        self.renderer.scroll_to_bottom();
    }

    fn append_agent_text(&mut self, agent_id: &beyonder_core::AgentId, text: &str) {
        // Find the most recent Running agent block for this agent_id.
        let idx = self.blocks.iter().rposition(|b| {
            b.agent_id.as_ref() == Some(agent_id)
                && matches!(b.status, BlockStatus::Running)
        });

        if let Some(idx) = idx {
            let block = &mut self.blocks[idx];
            if let BlockContent::AgentMessage { content_blocks, .. } = &mut block.content {
                if let Some(beyonder_core::ContentBlock::Text { text: t }) = content_blocks.last_mut() {
                    t.push_str(text);
                } else {
                    content_blocks.push(beyonder_core::ContentBlock::Text { text: text.to_string() });
                }
                // No per-delta DB write — persist only on finalize to avoid blocking tick().
            }
        } else {
            let mut block = Block::new(
                BlockKind::Agent,
                self.session.id.clone(),
                BlockContent::AgentMessage {
                    role: beyonder_core::MessageRole::Assistant,
                    content_blocks: vec![beyonder_core::ContentBlock::Text {
                        text: text.to_string(),
                    }],
                },
            );
            block.agent_id = Some(agent_id.clone());
            block.status = BlockStatus::Running;
            self.add_block(block);
        }
    }

    fn finalize_agent_block(&mut self, agent_id: &beyonder_core::AgentId) {
        if let Some(idx) = self.blocks.iter().rposition(|b| {
            b.agent_id.as_ref() == Some(agent_id)
                && matches!(b.status, BlockStatus::Running)
        }) {
            let is_empty = match &self.blocks[idx].content {
                beyonder_core::BlockContent::AgentMessage { content_blocks, .. } =>
                    content_blocks.is_empty(),
                _ => false,
            };
            if is_empty {
                // Empty pending block — drop it rather than leaving a blank completed block.
                self.blocks.remove(idx);
            } else {
                self.blocks[idx].status = BlockStatus::Completed;
                // DB write deferred — avoid blocking the main thread with fsync.
            }
        }
        self.renderer.blocks = self.blocks.clone();
    }

    fn push_tool_call_block(
        &mut self,
        agent_id: &beyonder_core::AgentId,
        tool_use_id: String,
        tool_name: String,
        input: serde_json::Value,
    ) {
        use beyonder_core::{BlockContent, BlockId, BlockKind, BlockStatus, ProvenanceChain};
        let now = chrono::Utc::now();
        let block = Block {
            id: BlockId::new(),
            kind: BlockKind::Tool,
            parent_id: None,
            agent_id: Some(agent_id.clone()),
            session_id: self.session.id.clone(),
            status: BlockStatus::Running,
            content: BlockContent::ToolCall {
                tool_name,
                tool_use_id,
                input,
                output: None,
                streaming_text: None,
                error: None,
                collapsed_default: true,
            },
            created_at: now,
            updated_at: now,
            provenance: ProvenanceChain::default(),
        };
        self.blocks.push(block);
        // renderer.blocks is synced by the agent_events_received flag in tick().
    }

    fn complete_tool_call_block(&mut self, tool_use_id: &str, output: String, is_error: bool) {
        if let Some(block) = self.blocks.iter_mut().rev().find(|b| {
            if let beyonder_core::BlockContent::ToolCall { tool_use_id: tid, .. } = &b.content {
                tid == tool_use_id
            } else {
                false
            }
        }) {
            if let beyonder_core::BlockContent::ToolCall { output: out, error, .. } = &mut block.content {
                if is_error {
                    *error = Some(output);
                } else {
                    *out = Some(output);
                }
            }
            block.status = beyonder_core::BlockStatus::Completed;
        }
    }

    pub fn render(&mut self) -> Result<()> {
        self.command_running = self.block_builder.is_running_command() || self.term_grid.tui_active();

        // Interactive CLIs that take over the terminal but don't use alt-screen
        // (e.g. `claude`) should hide the input bar just like nvim/htop do.
        let interactive_cli = self.block_builder.running_command_name()
            .map(|name| matches!(name, "claude" | "claude-code"))
            .unwrap_or(false);

        // Full-screen TUI takeover: alt-screen apps OR known interactive CLIs.
        self.renderer.tui_active = self.term_grid.tui_active() || interactive_cli;
        self.renderer.tui_cursor_shape = self.term_grid.cursor_shape_code();

        // Tell renderer which block (if any) is the live running command.
        self.renderer.running_block_idx = if self.block_builder.is_running_command() {
            self.blocks.iter().rposition(|b| b.status == beyonder_core::BlockStatus::Running)
        } else {
            None
        };

        // Input bar: always show normally — no running state or color change.
        self.renderer.input_text = self.input.text.clone();
        self.renderer.input_cursor = self.input.cursor;
        self.renderer.input_mode_prefix = match detect_mode(&self.input.text) {
            InputMode::Shell => String::new(),
            InputMode::Agent { name } => format!("@{} ", name),
            InputMode::Command { .. } => String::new(),
        };
        self.renderer.input_running = self.block_builder.is_running_command();

        // Command palette — filter commands by what's been typed after the leading /.
        self.renderer.command_palette = if let InputMode::Command { ref cmd } = detect_mode(&self.input.text) {
            let matches = commands::filter(cmd);
            if matches.is_empty() {
                None
            } else {
                Some(matches.iter().map(|c| (c.usage.to_string(), c.description.to_string())).collect())
            }
        } else {
            None
        };

        // Sync mode switcher label.
        self.renderer.mode_label = self.app_mode.label().to_string();

        // Sync active model name.
        self.renderer.agent_model = self.active_model.clone();

        // Sync context pill labels.
        self.renderer.context_pills = vec![
            format!("conda: {}", self.current_conda),
            format!("node: {}", self.current_node),
            self.block_builder.cwd
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("~")
                .to_string(),
        ];

        self.renderer.render()
    }
}
