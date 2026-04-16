//! Application state machine — wires all subsystems together.

use anyhow::Result;
use beyonder_acp::client::AgentEvent;
use beyonder_config::{BeyonderConfig, ProviderConfig};
use beyonder_core::{
    AgentKind, Block, BlockContent, BlockKind, BlockStatus, CapabilitySet, Session,
};
use beyonder_gpu::Renderer;
use beyonder_runtime::{
    capability_broker::{BrokerEvent, CapabilityBroker},
    supervisor::{AgentSupervisor, SupervisorEvent},
};
use beyonder_store::{BlockStore, SessionStore, Store};
use beyonder_terminal::block_builder::BuildEvent;
use beyonder_terminal::pty::PtyEvent;
use beyonder_terminal::{BlockBuilder, PtySession, TermGrid};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};
use winit::event::{ElementState, Ime, KeyEvent, MouseScrollDelta, WindowEvent};
use winit::event_loop::EventLoopProxy;
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
    let output = std::process::Command::new("node").arg("--version").output();
    match output {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).trim().to_string(),
        _ => "—".to_string(),
    }
}

fn fetch_conda_envs() -> Vec<String> {
    let output = std::process::Command::new("conda")
        .args(["env", "list"])
        .output();
    match output {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout)
            .lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
            .filter_map(|l| l.split_whitespace().next())
            .map(|s| s.to_string())
            .collect(),
        _ => vec![],
    }
}

fn fetch_node_versions() -> Vec<String> {
    let nvm_dir = std::env::var("NVM_DIR").unwrap_or_else(|_| {
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

fn longest_common_prefix(items: &[&str]) -> String {
    if items.is_empty() {
        return String::new();
    }
    let mut prefix = items[0].to_string();
    for s in &items[1..] {
        let mut new_len = 0;
        for ((i, a), b) in prefix.char_indices().zip(s.chars()) {
            if a == b {
                new_len = i + a.len_utf8();
            } else {
                break;
            }
        }
        prefix.truncate(new_len);
        if prefix.is_empty() {
            break;
        }
    }
    prefix
}

/// File-path completion. Returns candidates with `/` appended for directories.
/// Each candidate is the full token text (including dir prefix) so the caller can
/// drop it straight in. `~` is expanded against `$HOME`.
fn path_completions(token: &str, cwd: &std::path::Path) -> Vec<String> {
    let home = std::env::var("HOME").unwrap_or_default();
    // Split token into directory portion and final-component prefix.
    let (dir_part, prefix) = match token.rfind('/') {
        Some(i) => (&token[..=i], &token[i + 1..]),
        None => ("", token),
    };
    // Resolve dir_part to a real filesystem path for reading.
    let resolved_dir: std::path::PathBuf = if dir_part.is_empty() {
        cwd.to_path_buf()
    } else if let Some(rest) = dir_part.strip_prefix("~/") {
        std::path::PathBuf::from(&home).join(rest)
    } else if dir_part == "~/" || dir_part == "~" {
        std::path::PathBuf::from(&home)
    } else if dir_part.starts_with('/') {
        std::path::PathBuf::from(dir_part)
    } else {
        cwd.join(dir_part)
    };

    let entries = match std::fs::read_dir(&resolved_dir) {
        Ok(e) => e,
        Err(_) => return vec![],
    };
    let show_hidden = prefix.starts_with('.');
    let mut out: Vec<String> = entries
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let name = e.file_name().into_string().ok()?;
            if !show_hidden && name.starts_with('.') {
                return None;
            }
            if !name.starts_with(prefix) {
                return None;
            }
            let is_dir = e.file_type().map(|t| t.is_dir()).unwrap_or(false);
            let suffix = if is_dir { "/" } else { "" };
            Some(format!("{dir_part}{name}{suffix}"))
        })
        .collect();
    out.sort();
    out
}

/// Command completion: scan `$PATH` for executables starting with `prefix`.
fn command_completions(prefix: &str) -> Vec<String> {
    if prefix.is_empty() {
        return vec![];
    }
    let path_var = std::env::var("PATH").unwrap_or_default();
    let mut seen: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for dir in path_var.split(':').filter(|s| !s.is_empty()) {
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for e in entries.flatten() {
            if let Ok(name) = e.file_name().into_string() {
                if name.starts_with(prefix) {
                    // Only include executables (best-effort: any file with x bit on unix).
                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::PermissionsExt;
                        let exec = e
                            .metadata()
                            .map(|m| m.permissions().mode() & 0o111 != 0)
                            .unwrap_or(false);
                        if !exec {
                            continue;
                        }
                    }
                    seen.insert(name);
                }
            }
        }
    }
    seen.into_iter().collect()
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
                let line: String = row.cells.iter().map(|c| c.grapheme.as_str()).collect();
                s.push_str(&line);
                s.push('\n');
            }
            s
        }
        BlockContent::AgentMessage { content_blocks, .. } => content_blocks
            .iter()
            .map(|cb| match cb {
                beyonder_core::ContentBlock::Text { text } => text.clone(),
                beyonder_core::ContentBlock::Code { code, language } => {
                    format!("```{}\n{}\n```", language.as_deref().unwrap_or(""), code)
                }
                beyonder_core::ContentBlock::Thinking { thinking } => thinking.clone(),
            })
            .collect::<Vec<_>>()
            .join("\n"),
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
            BlockContent::ShellCommand {
                input,
                output,
                exit_code,
                ..
            } => {
                let mut entry = format!("$ {}\n", input.trim());
                let mut lines: Vec<String> = output
                    .rows
                    .iter()
                    .map(|row| {
                        let line: String = row.cells.iter().map(|c| c.grapheme.as_str()).collect();
                        line.trim_end().to_string()
                    })
                    .filter(|l| !l.is_empty())
                    .collect();
                // Truncate very long outputs so context stays token-efficient.
                const MAX_OUTPUT_LINES: usize = 60;
                if lines.len() > MAX_OUTPUT_LINES {
                    let kept = MAX_OUTPUT_LINES / 2;
                    let omitted = lines.len() - kept * 2;
                    lines = lines[..kept]
                        .iter()
                        .chain(std::iter::once(&format!(
                            "... ({omitted} lines omitted) ..."
                        )))
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
            BlockContent::AgentMessage {
                role,
                content_blocks,
            } => {
                let label = match role {
                    beyonder_core::MessageRole::User => "User",
                    beyonder_core::MessageRole::Assistant => "Assistant",
                    _ => continue,
                };
                let text: String = content_blocks
                    .iter()
                    .filter_map(|cb| match cb {
                        beyonder_core::ContentBlock::Text { text } => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
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

/// Persistent state for Tab-cycle completion.
#[derive(Debug, Clone)]
struct CompletionCycle {
    /// Byte offset where the completed token begins in `input.text`.
    token_start: usize,
    /// All candidates that share the user's original prefix.
    candidates: Vec<String>,
    /// Currently displayed candidate index (None ⇒ we only inserted the LCP, haven't started cycling yet).
    index: Option<usize>,
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
            AppMode::Cmd => "shell",
            AppMode::Agent => "agent",
        }
    }
}

/// Per-tab state — everything that needs to be isolated between tabs.
/// When a tab is not active, its fields are stashed here; the active tab's
/// fields live directly on `App` to keep existing code paths unchanged.
pub struct TabState {
    pub session: Session,
    pub pty: Option<PtySession>,
    pub block_builder: BlockBuilder,
    pub term_grid: TermGrid,
    pub blocks: Vec<Block>,
    pub input: InputEditor,
    pub agent_context_watermark: usize,
    pub agent_running_tool: std::collections::HashMap<beyonder_core::AgentId, String>,
    pub pending_agent_fallback: Option<String>,
    pub selected_block: Option<usize>,
    pub selected_sub_output: bool,
    pub prev_tui_active: bool,
    pub command_running: bool,
    pub title: String,
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
    /// Sub-line accumulator for smooth mouse-wheel scrolling in TUI mode.
    scroll_accum: f32,
    /// Currently pressed mouse button for SGR drag reporting.
    mouse_button_down: Option<winit::event::MouseButton>,
    /// Last cell (1-based col, row) forwarded to the PTY, for motion de-duping.
    last_mouse_cell: Option<(u32, u32)>,
    /// Timestamp + position of the last left-button press — used to detect double-clicks.
    last_lmb_press: Option<(std::time::Instant, (f32, f32))>,

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
    /// Tab-completion cycle state. Set when the user has multiple matching candidates;
    /// repeated Tab cycles through them. Any non-Tab keystroke clears this.
    completion_cycle: Option<CompletionCycle>,
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

    // ── Tabs ──────────────────────────────────────────────────────────────────
    /// Stashed per-tab state. `None` at `active_tab` (its fields live on App).
    pub tabs: Vec<Option<TabState>>,
    /// Index of the currently active tab in `tabs`.
    pub active_tab: usize,
    /// Displayable titles for each tab (synced to renderer before each frame).
    pub tab_titles: Vec<String>,

    /// Window handle — retained so IME cursor-area updates can be pushed on
    /// every caret-move. Rendering already holds its own Arc internally.
    window: Arc<Window>,
    /// Active IME preedit (composition) text — not yet committed to the editor.
    pub ime_preedit: String,
    /// Optional IME cursor range within `ime_preedit` (byte offsets).
    pub ime_preedit_cursor: Option<(usize, usize)>,

    /// Receives a unit signal whenever the config file on disk changes.
    /// The sender half lives inside a `notify::RecommendedWatcher` retained on
    /// `_config_watcher` so it isn't dropped.
    config_reload_rx: Option<std::sync::mpsc::Receiver<()>>,
    _config_watcher: Option<notify::RecommendedWatcher>,

    // ── Scrollback search ─────────────────────────────────────────────────────
    /// Active search pattern buffer. `Some` iff in search mode.
    pub search_pattern: Option<String>,
    /// Matched block indices (into `self.blocks`) for the current pattern.
    pub search_matches: Vec<usize>,
    /// Index (within `search_matches`) of the currently focused match.
    pub search_current: Option<usize>,
    /// Saved input text to restore on search-mode exit.
    pub search_saved_input: String,

    // ── /phone bridge ─────────────────────────────────────────────────────────
    /// Live WebSocket bridge to the companion iOS app. `None` until /phone on.
    pub remote: Option<beyonder_remote::RemoteHub>,
    /// How many blocks have been marked final on the phone. Running blocks
    /// beyond this get rebroadcast each tick so streaming text stays current.
    remote_cursor: usize,
    /// Phone-preferred PTY dims. Re-applied each tick so the Mac window's
    /// resize events don't clobber what the phone asked for.
    remote_pty_dims: Option<(u16, u16)>,
    remote_last_pty_frame: std::time::Instant,
    /// Previous PTY frame cells sent to phone — used to compute diffs.
    remote_prev_cells: Vec<Vec<beyonder_remote::PtyCell>>,
    /// Proxy to wake the winit event loop from async event sources (PTY, agent, broker).
    event_loop_proxy: EventLoopProxy<()>,
    remote_connect_gen: u64,
}

impl App {
    pub async fn new(
        window: Arc<Window>,
        config: BeyonderConfig,
        event_loop_proxy: EventLoopProxy<()>,
    ) -> Result<Self> {
        let mut renderer = Renderer::new(Arc::clone(&window)).await?;
        renderer.set_theme(config.resolved_theme());
        // Enable IME so winit delivers WindowEvent::Ime (preedit/commit) events.
        window.set_ime_allowed(true);

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
        let mut supervisor = AgentSupervisor::new(supervisor_tx);
        let wake_proxy_sup = event_loop_proxy.clone();
        supervisor.set_wake(std::sync::Arc::new(move || {
            let _ = wake_proxy_sup.send_event(());
        }));

        // Calculate PTY dimensions from the renderer — single source of truth.
        let (pty_cols, pty_rows) = renderer.terminal_grid_size();

        let mut block_builder = BlockBuilder::new(session_id.clone(), cwd.clone());
        block_builder.set_grid_size(pty_cols as usize, pty_rows as usize);

        let term_grid = TermGrid::new(pty_cols as usize, pty_rows as usize);

        // Spawn the shell PTY.
        let shell_env = std::env::var("SHELL").unwrap_or_else(|_| "/bin/zsh".to_string());
        let shell = config.shell.program.as_deref().unwrap_or(&shell_env);

        let wake_proxy = event_loop_proxy.clone();
        let wake_fn: beyonder_terminal::pty::WakeFn = Box::new(move || {
            let _ = wake_proxy.send_event(());
        });
        let pty = match PtySession::spawn_sized_with_wake(
            session_id.clone(),
            shell,
            &cwd,
            &[],
            pty_cols,
            pty_rows,
            wake_fn,
        ) {
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

        // ── Config file watcher ─────────────────────────────────────────────
        // Watch the config file's parent dir (not the file itself) so that
        // atomic-rename editors like vim still trigger events. Events are
        // normalised to a unit signal — the tick() handler re-reads the file.
        let (config_reload_rx, config_watcher) = {
            use notify::{Event, EventKind, RecursiveMode, Watcher};
            let cfg_path = beyonder_config::config_path();
            let cfg_dir = cfg_path.parent().map(|p| p.to_path_buf());
            let cfg_name = cfg_path.file_name().map(|n| n.to_os_string());
            let (tx, rx) = std::sync::mpsc::channel::<()>();
            let watcher_tx = tx.clone();
            let cfg_name_cb = cfg_name.clone();
            let result = notify::recommended_watcher(move |res: notify::Result<Event>| {
                if let Ok(ev) = res {
                    if !matches!(
                        ev.kind,
                        EventKind::Modify(_) | EventKind::Create(_) | EventKind::Remove(_)
                    ) {
                        return;
                    }
                    let touches_cfg = match &cfg_name_cb {
                        Some(name) => ev.paths.iter().any(|p| p.file_name() == Some(name)),
                        None => true,
                    };
                    if touches_cfg {
                        let _ = watcher_tx.send(());
                    }
                }
            });
            match (result, cfg_dir) {
                (Ok(mut w), Some(dir)) => {
                    // Ensure the parent dir exists so the watcher has something to attach to.
                    let _ = std::fs::create_dir_all(&dir);
                    match w.watch(&dir, RecursiveMode::NonRecursive) {
                        Ok(()) => (Some(rx), Some(w)),
                        Err(e) => {
                            warn!("Config watcher failed to start: {e}");
                            (None, None)
                        }
                    }
                }
                (Err(e), _) => {
                    warn!("Could not create config watcher: {e}");
                    (None, None)
                }
                _ => (None, None),
            }
        };

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
            scroll_accum: 0.0,
            mouse_button_down: None,
            last_mouse_cell: None,
            last_lmb_press: None,
            current_conda: current_conda_env(),
            current_node: current_node_version(),
            conda_envs: fetch_conda_envs(),
            node_versions: fetch_node_versions(),
            open_pill_dropdown: None,
            dropdown_items: vec![],
            active_model,
            active_provider,
            app_mode: AppMode::Auto,
            completion_cycle: None,
            should_quit: false,
            pending_agent_fallback: None,
            agent_context_watermark: 0,
            agent_running_tool: std::collections::HashMap::new(),
            // Use a past instant so the first post-command probe runs after startup.
            last_env_probe: std::time::Instant::now() - std::time::Duration::from_secs(60),
            tabs: vec![None],
            active_tab: 0,
            tab_titles: vec!["1".to_string()],
            window,
            ime_preedit: String::new(),
            ime_preedit_cursor: None,
            config_reload_rx,
            _config_watcher: config_watcher,
            search_pattern: None,
            search_matches: vec![],
            search_current: None,
            search_saved_input: String::new(),
            remote: None,
            remote_cursor: 0,
            remote_pty_dims: None,
            remote_last_pty_frame: std::time::Instant::now(),
            remote_prev_cells: vec![],
            remote_connect_gen: 0,
            event_loop_proxy,
        })
    }

    /// Swap App's active per-tab fields with those in `target`.
    /// Returns the displaced state as a new `TabState`.
    fn exchange_active(&mut self, target: TabState) -> TabState {
        TabState {
            session: std::mem::replace(&mut self.session, target.session),
            pty: std::mem::replace(&mut self.pty, target.pty),
            block_builder: std::mem::replace(&mut self.block_builder, target.block_builder),
            term_grid: std::mem::replace(&mut self.term_grid, target.term_grid),
            blocks: std::mem::replace(&mut self.blocks, target.blocks),
            input: std::mem::replace(&mut self.input, target.input),
            agent_context_watermark: std::mem::replace(
                &mut self.agent_context_watermark,
                target.agent_context_watermark,
            ),
            agent_running_tool: std::mem::replace(
                &mut self.agent_running_tool,
                target.agent_running_tool,
            ),
            pending_agent_fallback: std::mem::replace(
                &mut self.pending_agent_fallback,
                target.pending_agent_fallback,
            ),
            selected_block: std::mem::replace(&mut self.selected_block, target.selected_block),
            selected_sub_output: std::mem::replace(
                &mut self.selected_sub_output,
                target.selected_sub_output,
            ),
            prev_tui_active: std::mem::replace(&mut self.prev_tui_active, target.prev_tui_active),
            command_running: std::mem::replace(&mut self.command_running, target.command_running),
            title: target.title,
        }
    }

    /// After swapping tabs, re-sync the renderer-visible fields so the next frame
    /// reflects the newly-active tab's state, and reshape the PTY / grid so the
    /// revived tab matches the current window size.
    fn resync_renderer_after_tab_switch(&mut self) {
        let (cols, rows) = if self.term_grid.tui_active() {
            self.renderer.tui_grid_size()
        } else {
            self.renderer.terminal_grid_size()
        };
        self.term_grid.resize(cols as usize, rows as usize);
        self.block_builder
            .set_grid_size(cols as usize, rows as usize);
        if let Some(pty) = &self.pty {
            let _ = pty.resize(rows, cols);
        }
        self.renderer.blocks = self.blocks.clone();
        self.renderer.selected_block = self.selected_block;
        self.renderer.selected_sub_output = self.selected_sub_output;
        self.renderer.agent_running_tool = self.agent_running_tool.clone();
        self.renderer.running_block_idx = None;
        self.renderer.tui_cells = vec![];
        self.renderer.viewport.scroll_to_bottom();
        self.renderer.snap_input_scroll_to_cursor();
    }

    /// Construct a fresh `TabState` — spawns a new PTY sized to the renderer.
    fn fresh_tab_state(&self, title: String) -> TabState {
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/"));
        let session = Session::new(cwd.clone());
        let session_id = session.id.clone();
        let _ = SessionStore::new(&self.store).insert(&session);

        let (pty_cols, pty_rows) = self.renderer.terminal_grid_size();

        let mut block_builder = BlockBuilder::new(session_id.clone(), cwd.clone());
        block_builder.set_grid_size(pty_cols as usize, pty_rows as usize);
        let term_grid = TermGrid::new(pty_cols as usize, pty_rows as usize);

        let shell_env = std::env::var("SHELL").unwrap_or_else(|_| "/bin/zsh".to_string());
        let shell = self.config.shell.program.as_deref().unwrap_or(&shell_env);
        let wake_proxy = self.event_loop_proxy.clone();
        let wake_fn: beyonder_terminal::pty::WakeFn = Box::new(move || {
            let _ = wake_proxy.send_event(());
        });
        let pty = match PtySession::spawn_sized_with_wake(
            session_id.clone(),
            shell,
            &cwd,
            &[],
            pty_cols,
            pty_rows,
            wake_fn,
        ) {
            Ok(p) => {
                info!("Shell PTY spawned for new tab");
                Some(p)
            }
            Err(e) => {
                warn!("Failed to spawn PTY for new tab: {e}");
                None
            }
        };

        TabState {
            session,
            pty,
            block_builder,
            term_grid,
            blocks: vec![],
            input: InputEditor::new(),
            agent_context_watermark: 0,
            agent_running_tool: std::collections::HashMap::new(),
            pending_agent_fallback: None,
            selected_block: None,
            selected_sub_output: false,
            prev_tui_active: false,
            command_running: false,
            title,
        }
    }

    /// Open a new tab and switch to it.
    pub fn new_tab(&mut self) {
        let next_idx = self.tabs.len();
        let title = format!("{}", next_idx + 1);
        let fresh = self.fresh_tab_state(title.clone());
        // Stash the currently-active tab, swap fresh into App, and append a None slot
        // at the new active index.
        let displaced = self.exchange_active(fresh);
        self.tabs[self.active_tab] = Some(displaced);
        self.tabs.push(None);
        self.tab_titles.push(title);
        self.active_tab = next_idx;
        self.remote_cursor = 0;
        self.resync_renderer_after_tab_switch();
        self.broadcast_tab_list();
    }

    /// Close the active tab. Falls back to adjacent tab; if it was the last tab,
    /// opens a fresh one so the app is never tab-less.
    pub fn close_tab(&mut self) {
        if self.tabs.len() <= 1 {
            // Don't allow closing the last tab — silently ignore.
            return;
        }
        // Pick the tab to switch to (prefer left neighbour, else right).
        let target_idx = if self.active_tab > 0 {
            self.active_tab - 1
        } else {
            1
        };
        let target = self.tabs[target_idx].take().unwrap();
        // Swap target into App; the old active state is discarded (PTY child dropped).
        let _old = self.exchange_active(target);
        // Remove the slot we just left.
        let removed = self.active_tab;
        self.tabs.remove(removed);
        self.tab_titles.remove(removed);
        // Adjust active index: if target was to the right of the removed slot, it shifted left.
        self.active_tab = if target_idx > removed {
            target_idx - 1
        } else {
            target_idx
        };
        self.remote_cursor = 0;
        self.resync_renderer_after_tab_switch();
        self.broadcast_tab_list();
    }

    /// Switch to the tab at `idx` if valid and not already active.
    pub fn switch_tab(&mut self, idx: usize) {
        if idx == self.active_tab || idx >= self.tabs.len() {
            return;
        }
        let target = match self.tabs[idx].take() {
            Some(t) => t,
            None => return, // shouldn't happen
        };
        let displaced = self.exchange_active(target);
        self.tabs[self.active_tab] = Some(displaced);
        self.active_tab = idx;
        self.remote_cursor = 0;
        self.resync_renderer_after_tab_switch();
        self.broadcast_tab_list();
    }

    pub fn prev_tab(&mut self) {
        if self.tabs.len() < 2 {
            return;
        }
        let idx = if self.active_tab == 0 {
            self.tabs.len() - 1
        } else {
            self.active_tab - 1
        };
        self.switch_tab(idx);
    }

    pub fn next_tab(&mut self) {
        if self.tabs.len() < 2 {
            return;
        }
        let idx = (self.active_tab + 1) % self.tabs.len();
        self.switch_tab(idx);
    }

    fn build_tab_list(&self) -> beyonder_remote::TabList {
        let tabs: Vec<beyonder_remote::TabInfo> = self
            .tab_titles
            .iter()
            .enumerate()
            .map(|(i, title)| {
                let sid = if i == self.active_tab {
                    self.session.id.0.clone()
                } else {
                    self.tabs[i]
                        .as_ref()
                        .map(|t| t.session.id.0.clone())
                        .unwrap_or_default()
                };
                beyonder_remote::TabInfo {
                    index: i,
                    title: title.clone(),
                    session_id: sid,
                }
            })
            .collect();
        beyonder_remote::TabList {
            tabs,
            active: self.active_tab,
        }
    }

    fn broadcast_tab_list(&self) {
        if let Some(hub) = &self.remote {
            let _ = hub.send(beyonder_remote::ServerMsg::TabList(self.build_tab_list()));
        }
    }

    /// Handle a winit window event.
    pub async fn handle_window_event(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::Resized(size) => {
                self.renderer.resize(size.width, size.height);
                // Use full-window dimensions when a TUI app is active (bar is hidden),
                // above-bar dimensions otherwise.
                let interactive_cli = self
                    .block_builder
                    .running_command_name()
                    .map(|name| matches!(name, "claude" | "claude-code"))
                    .unwrap_or(false);
                let (cols, rows) = if self.term_grid.tui_active() || interactive_cli {
                    self.renderer.tui_grid_size()
                } else {
                    self.renderer.terminal_grid_size()
                };
                self.term_grid.resize(cols as usize, rows as usize);
                self.block_builder
                    .set_grid_size(cols as usize, rows as usize);
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
                self.renderer.cmd_palette_hovered = self
                    .renderer
                    .cmd_palette_hit(self.cursor_pos.0, self.cursor_pos.1);
                // Update dropdown hover highlight.
                if self.open_pill_dropdown.is_some() {
                    let hovered = self
                        .renderer
                        .dropdown_hover_at(self.cursor_pos.0, self.cursor_pos.1);
                    if let Some((_pill_idx, ref _items, ref mut h)) = self.renderer.open_dropdown {
                        *h = hovered;
                    }
                }
                // Extend an in-progress text selection while the left button is held.
                if self.mouse_button_down == Some(winit::event::MouseButton::Left)
                    && !self.renderer.tui_active
                {
                    self.renderer
                        .update_text_selection(self.cursor_pos.0, self.cursor_pos.1);
                }
                // SGR mouse motion / drag forwarding while a TUI is active.
                if self.renderer.tui_active {
                    let mr = self.term_grid.mouse_report_mode();
                    if mr.sgr && (mr.motion || (mr.drag && self.mouse_button_down.is_some())) {
                        if let Some((cx, cy)) = self
                            .renderer
                            .cell_at_phys(self.cursor_pos.0, self.cursor_pos.1)
                        {
                            if self.last_mouse_cell != Some((cx, cy)) {
                                self.last_mouse_cell = Some((cx, cy));
                                let btn = self.mouse_button_down;
                                let cb =
                                    sgr_button_code(btn, false, false) + 32 + self.modifier_bits();
                                let seq = format!("\x1b[<{cb};{cx};{cy}M");
                                self.write_to_pty(&seq);
                            }
                        }
                    }
                }
                false
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if *state == ElementState::Pressed {
                    self.mouse_button_down = Some(*button);
                }
                // SGR mouse press/release forwarding.
                if self.renderer.tui_active {
                    let mr = self.term_grid.mouse_report_mode();
                    if mr.sgr && mr.any() {
                        if let Some((cx, cy)) = self
                            .renderer
                            .cell_at_phys(self.cursor_pos.0, self.cursor_pos.1)
                        {
                            let cb =
                                sgr_button_code(Some(*button), false, false) + self.modifier_bits();
                            let trailer = if *state == ElementState::Pressed {
                                'M'
                            } else {
                                'm'
                            };
                            let seq = format!("\x1b[<{cb};{cx};{cy}{trailer}");
                            self.write_to_pty(&seq);
                            if *state == ElementState::Released {
                                self.mouse_button_down = None;
                            }
                            return false;
                        }
                    }
                }
                if *state == ElementState::Released {
                    self.mouse_button_down = None;
                    if *button == winit::event::MouseButton::Left {
                        // Finish drag-selection; drop zero-width selections so the
                        // next click starts clean (and block-click still works).
                        self.renderer.end_text_selection();
                        if !self.renderer.has_text_selection() {
                            self.renderer.clear_text_selection();
                        }
                    }
                }
                if *state == ElementState::Pressed && *button == winit::event::MouseButton::Left {
                    // Double-click detection: second LMB press within 400ms and 5px of the
                    // first begins a text-selection drag instead of a block click.
                    let now = std::time::Instant::now();
                    let is_double = self.last_lmb_press.map_or(false, |(t, p)| {
                        now.duration_since(t).as_millis() <= 400
                            && (self.cursor_pos.0 - p.0).abs() <= 5.0
                            && (self.cursor_pos.1 - p.1).abs() <= 5.0
                    });
                    if is_double {
                        if self
                            .renderer
                            .begin_text_selection(self.cursor_pos.0, self.cursor_pos.1)
                        {
                            // Suppress block-level selection while a text range is active.
                            self.selected_block = None;
                            self.selected_sub_output = false;
                            self.renderer.selected_block = None;
                            self.renderer.selected_sub_output = false;
                        }
                        // Reset so triple-click doesn't chain into another "double".
                        self.last_lmb_press = None;
                    } else {
                        self.last_lmb_press = Some((now, self.cursor_pos));
                        self.handle_click(self.cursor_pos);
                    }
                }
                false
            }
            WindowEvent::KeyboardInput { event, .. } => {
                self.handle_key(event).await;
                self.should_quit
            }
            WindowEvent::MouseWheel { delta, .. } => {
                // Accumulate sub-line deltas (macOS trackpads send many small
                // PixelDelta events per gesture) so small scrolls aren't lost
                // to rounding. Cell height ≈ font_size * scale.
                let (_, cell_h) = self.renderer.terminal_cell_size();
                let (line_mul, px_div) = (1.0f32, cell_h.max(1.0));
                let lines_up: f32 = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y * line_mul,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / px_div,
                };
                self.scroll_accum += lines_up;
                // If the TUI has requested mouse reporting, forward wheel as SGR
                // events at the cursor position instead of scrolling the grid.
                if self.renderer.tui_active {
                    let mr = self.term_grid.mouse_report_mode();
                    if mr.sgr && mr.any() {
                        let step = self.scroll_accum.trunc() as i32;
                        if step != 0 {
                            self.scroll_accum -= step as f32;
                            if let Some((cx, cy)) = self
                                .renderer
                                .cell_at_phys(self.cursor_pos.0, self.cursor_pos.1)
                            {
                                let cb_base = if step > 0 { 64 } else { 65 };
                                let cb = cb_base + self.modifier_bits();
                                let count = step.abs();
                                for _ in 0..count {
                                    let seq = format!("\x1b[<{cb};{cx};{cy}M");
                                    self.write_to_pty(&seq);
                                }
                            }
                        }
                        return false;
                    }
                }
                // In a full-screen TUI (claude, nvim, etc.) blocks don't scroll.
                // Move the terminal's display_offset so the user can scroll back
                // through the primary-screen history. No-op in alt-screen apps
                // since they keep no scrollback — their grid history_size is 0.
                if self.renderer.tui_active {
                    let step = self.scroll_accum.trunc() as i32;
                    if step != 0 {
                        self.scroll_accum -= step as f32;
                        self.term_grid.scroll_display(step);
                        // Re-read the grid immediately so the next frame reflects
                        // the new display_offset — tui_cells is otherwise only
                        // refreshed on PTY output.
                        self.renderer.tui_cells = self.term_grid.cell_grid();
                        self.renderer.tui_cursor = self.term_grid.cursor_pos();
                    }
                    return false;
                }
                self.scroll_accum = 0.0;
                let scroll = -lines_up * 20.0;
                // If cursor is over the input bar, scroll the input text; otherwise scroll blocks.
                let win_h = self.renderer.surface_size().1;
                let bar_top = win_h - self.renderer.bar_height_phys();
                if self.cursor_pos.1 >= bar_top {
                    self.renderer.scroll_input(scroll);
                } else {
                    self.renderer.scroll(scroll);
                }
                false
            }
            WindowEvent::Focused(gained) => {
                if self.renderer.tui_active && self.term_grid.focus_reporting_enabled() {
                    let seq = if *gained { "\x1b[I" } else { "\x1b[O" };
                    self.write_to_pty(seq);
                }
                false
            }
            WindowEvent::Ime(ime) => {
                match ime {
                    Ime::Enabled | Ime::Disabled => {
                        self.ime_preedit.clear();
                        self.ime_preedit_cursor = None;
                    }
                    Ime::Preedit(text, cursor) => {
                        self.ime_preedit = text.clone();
                        self.ime_preedit_cursor = *cursor;
                    }
                    Ime::Commit(text) => {
                        self.ime_preedit.clear();
                        self.ime_preedit_cursor = None;
                        self.input.insert_text(text);
                        self.renderer.snap_input_scroll_to_cursor();
                    }
                }
                false
            }
            WindowEvent::CloseRequested => true,
            _ => false,
        }
    }

    /// Modifier mask bits used in SGR mouse Cb (+4 shift, +8 alt, +16 ctrl).
    fn modifier_bits(&self) -> u32 {
        let mut b = 0u32;
        if self.modifiers.shift_key() {
            b += 4;
        }
        if self.modifiers.alt_key() {
            b += 8;
        }
        if self.modifiers.control_key() {
            b += 16;
        }
        b
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
            Key::Named(NamedKey::F1) => Some(b"\x1bOP".to_vec()),
            Key::Named(NamedKey::F2) => Some(b"\x1bOQ".to_vec()),
            Key::Named(NamedKey::F3) => Some(b"\x1bOR".to_vec()),
            Key::Named(NamedKey::F4) => Some(b"\x1bOS".to_vec()),
            Key::Named(NamedKey::F5) => Some(b"\x1b[15~".to_vec()),
            Key::Named(NamedKey::F6) => Some(b"\x1b[17~".to_vec()),
            Key::Named(NamedKey::F7) => Some(b"\x1b[18~".to_vec()),
            Key::Named(NamedKey::F8) => Some(b"\x1b[19~".to_vec()),
            Key::Named(NamedKey::F9) => Some(b"\x1b[20~".to_vec()),
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
                                ' ' => 0x00,  // Ctrl+Space = NUL
                                '[' => 0x1b,  // Ctrl+[ = ESC
                                '\\' => 0x1c, // Ctrl+\ = FS
                                ']' => 0x1d,  // Ctrl+] = GS
                                '^' => 0x1e,  // Ctrl+^ = RS
                                '_' => 0x1f,  // Ctrl+_ = US
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
        // OSC 8 hyperlink click — open in default browser.
        for (rect, url) in &self.renderer.link_rects {
            let [rx, ry, rw, rh] = *rect;
            if pos.0 >= rx && pos.0 < rx + rw && pos.1 >= ry && pos.1 < ry + rh {
                let _ = open::that(url);
                return;
            }
        }
        // Tab strip click — switch to clicked tab.
        if let Some(tab_idx) = self.renderer.tab_hit(pos.0, pos.1) {
            self.switch_tab(tab_idx);
            return;
        }
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
            let is_output = self
                .renderer
                .block_hit_at(pos.1)
                .map(|(_, o)| o)
                .unwrap_or(false);
            self.selected_block = Some(idx);
            self.selected_sub_output = is_output;
            self.renderer.selected_block = Some(idx);
            self.renderer.selected_sub_output = is_output;
            self.renderer.clear_text_selection();
        } else {
            self.selected_block = None;
            self.selected_sub_output = false;
            self.renderer.selected_block = None;
            self.renderer.selected_sub_output = false;
            self.renderer.clear_text_selection();
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

    /// Scan PTY output for OSC 52 clipboard sequences and handle them.
    ///
    /// OSC 52 is how TUI apps (nvim, tmux, Claude Code) read and write the
    /// system clipboard without requiring xclip/pbcopy. Two operations:
    ///   • Write: `\x1b]52;{Pc};{base64_data}\x07` — set clipboard to decoded text
    ///   • Read:  `\x1b]52;{Pc};?\x07`             — respond with clipboard contents
    ///
    /// Returns a response to write back to the PTY if a read query was found.
    fn handle_osc52(&self, bytes: &[u8]) -> Option<String> {
        use base64::Engine;
        let s = std::str::from_utf8(bytes).ok()?;
        let mut response: Option<String> = None;
        let mut haystack = s;
        while let Some(start) = haystack.find("\x1b]52;") {
            let rest = &haystack[start + 5..]; // skip past "\x1b]52;"
                                               // Find the OSC terminator: BEL (\x07) or ST (\x1b\)
            let (payload, advance) = if let Some(pos) = rest.find('\x07') {
                (&rest[..pos], pos + 1)
            } else if let Some(pos) = rest.find("\x1b\\") {
                (&rest[..pos], pos + 2)
            } else {
                break; // incomplete sequence — skip
            };
            // payload = "Pc;Pd"
            if let Some(semi) = payload.find(';') {
                let pc = &payload[..semi];
                let pd = &payload[semi + 1..];
                if pd == "?" {
                    // Read query: send clipboard contents back to the PTY.
                    if let Ok(mut cb) = arboard::Clipboard::new() {
                        if let Ok(text) = cb.get_text() {
                            let encoded = base64::engine::general_purpose::STANDARD.encode(&text);
                            response = Some(format!("\x1b]52;{pc};{encoded}\x07"));
                        }
                    }
                } else if !pd.is_empty() {
                    // Write: decode base64 and put into clipboard.
                    if let Ok(decoded) = base64::engine::general_purpose::STANDARD.decode(pd) {
                        if let Ok(text) = String::from_utf8(decoded) {
                            if let Ok(mut cb) = arboard::Clipboard::new() {
                                let _ = cb.set_text(text);
                            }
                        }
                    }
                }
            }
            haystack = &rest[advance..];
        }
        response
    }

    async fn handle_key(&mut self, event: &KeyEvent) {
        if event.state != ElementState::Pressed {
            return;
        }
        // Any non-Tab keypress invalidates the active completion cycle.
        if !matches!(event.logical_key, Key::Named(NamedKey::Tab)) {
            self.completion_cycle = None;
        }

        // Cmd+F (macOS) / Ctrl+F (other) — toggle scrollback search mode.
        let super_or_ctrl_find = if cfg!(target_os = "macos") {
            self.modifiers.super_key()
        } else {
            self.modifiers.control_key()
        };
        if super_or_ctrl_find {
            if let Key::Character(s) = &event.logical_key {
                if s.as_str() == "f" {
                    if self.search_pattern.is_some() {
                        self.exit_search_mode();
                    } else {
                        self.enter_search_mode("");
                    }
                    return;
                }
            }
        }

        // While in search mode, route keystrokes to pattern editing + navigation.
        if self.search_pattern.is_some() {
            match &event.logical_key {
                Key::Named(NamedKey::Escape) => {
                    self.exit_search_mode();
                    return;
                }
                Key::Named(NamedKey::Enter) => {
                    if self.modifiers.shift_key() {
                        self.prev_match();
                    } else {
                        self.next_match();
                    }
                    return;
                }
                Key::Named(NamedKey::F3) => {
                    if self.modifiers.shift_key() {
                        self.prev_match();
                    } else {
                        self.next_match();
                    }
                    return;
                }
                Key::Named(NamedKey::Backspace) => {
                    self.input.delete_backward();
                    self.update_search_matches();
                    self.renderer.snap_input_scroll_to_cursor();
                    return;
                }
                Key::Named(NamedKey::ArrowLeft) => {
                    self.input.move_left();
                    self.renderer.snap_input_scroll_to_cursor();
                    return;
                }
                Key::Named(NamedKey::ArrowRight) => {
                    self.input.move_right();
                    self.renderer.snap_input_scroll_to_cursor();
                    return;
                }
                Key::Named(NamedKey::ArrowUp) => {
                    self.prev_match();
                    return;
                }
                Key::Named(NamedKey::ArrowDown) => {
                    self.next_match();
                    return;
                }
                Key::Character(s) => {
                    for ch in s.chars() {
                        self.input.insert(ch);
                    }
                    self.update_search_matches();
                    self.renderer.snap_input_scroll_to_cursor();
                    return;
                }
                Key::Named(NamedKey::Space) => {
                    self.input.insert(' ');
                    self.update_search_matches();
                    self.renderer.snap_input_scroll_to_cursor();
                    return;
                }
                _ => {}
            }
        }

        // Cmd+V (macOS) / Ctrl+V (other) — paste from clipboard.
        // Handled before TUI forwarding so it works in both the input box and
        // TUI apps (nvim, claude, htop). TUI path uses bracketed-paste protocol.
        let is_paste = if cfg!(target_os = "macos") {
            self.modifiers.super_key()
        } else {
            self.modifiers.control_key()
        };
        if is_paste {
            if let Key::Character(s) = &event.logical_key {
                if s.as_str() == "v" {
                    if let Ok(mut cb) = arboard::Clipboard::new() {
                        if let Ok(text) = cb.get_text() {
                            if self.term_grid.tui_active()
                                || self.block_builder.is_running_command()
                            {
                                // Bracketed paste: wraps pasted text so TUI apps
                                // (nvim insert mode, etc.) receive it correctly.
                                let payload = format!("\x1b[200~{text}\x1b[201~");
                                self.write_to_pty(&payload);
                            } else {
                                self.input.insert_text(&text);
                                self.renderer.snap_input_scroll_to_cursor();
                            }
                        }
                    }
                    return;
                }
            }
        }

        // Tab management shortcuts (macOS Cmd, other platforms Ctrl).
        // Intercept BEFORE TUI forwarding so tabs work even inside nvim / claude.
        let super_or_ctrl_tab = if cfg!(target_os = "macos") {
            self.modifiers.super_key()
        } else {
            self.modifiers.control_key()
        };
        if super_or_ctrl_tab {
            if let Key::Character(s) = &event.logical_key {
                match s.as_str() {
                    "t" => {
                        self.new_tab();
                        return;
                    }
                    "w" => {
                        self.close_tab();
                        return;
                    }
                    "[" => {
                        self.prev_tab();
                        return;
                    }
                    "]" => {
                        self.next_tab();
                        return;
                    }
                    _ => {}
                }
            }
        }

        // TUI/interactive mode: forward all keys directly to PTY as raw bytes.
        // Covers both alt-screen TUIs (vim, htop) and raw-mode apps (claude, React Ink).
        if self.term_grid.tui_active() || self.block_builder.is_running_command() {
            if let Some(bytes) = self.key_to_pty_bytes(event) {
                // Any keypress snaps the view back to the live screen — matches
                // xterm/iTerm behaviour where typing cancels scrollback.
                self.term_grid.scroll_to_bottom();
                if matches!(event.logical_key, Key::Named(NamedKey::Enter)) {
                    self.renderer.viewport.scroll_to_bottom();
                }
                if let Some(pty) = &mut self.pty {
                    let _ = pty.write(&bytes);
                }
            }
            return;
        }
        // Cmd+* shortcuts (macOS) / Ctrl+* shortcuts (other) for the input box.
        let super_or_ctrl = if cfg!(target_os = "macos") {
            self.modifiers.super_key()
        } else {
            self.modifiers.control_key()
        };
        if super_or_ctrl {
            if let Key::Character(s) = &event.logical_key {
                match s.as_str() {
                    "c" => {
                        self.copy_selection();
                        return;
                    }
                    "a" => {
                        self.input.select_all();
                        self.renderer.snap_input_scroll_to_cursor();
                        return;
                    }
                    "x" => {
                        // Cut: copy all input text to clipboard then clear.
                        if !self.input.text.is_empty() {
                            if let Ok(mut cb) = arboard::Clipboard::new() {
                                let _ = cb.set_text(self.input.text.clone());
                            }
                            self.input.select_all();
                            self.input.delete_backward();
                        }
                        self.renderer.snap_input_scroll_to_cursor();
                        return;
                    }
                    _ => {}
                }
            }
            // Cmd/Ctrl+Left = home, Cmd/Ctrl+Right = end.
            match &event.logical_key {
                Key::Named(NamedKey::ArrowLeft) => {
                    self.input.move_home();
                    self.renderer.snap_input_scroll_to_cursor();
                    return;
                }
                Key::Named(NamedKey::ArrowRight) => {
                    self.input.move_end();
                    self.renderer.snap_input_scroll_to_cursor();
                    return;
                }
                _ => {}
            }
        }

        // Alt/Option+Left/Right — word navigation.
        if self.modifiers.alt_key() {
            match &event.logical_key {
                Key::Named(NamedKey::ArrowLeft) => {
                    self.input.word_left();
                    self.renderer.snap_input_scroll_to_cursor();
                    return;
                }
                Key::Named(NamedKey::ArrowRight) => {
                    self.input.word_right();
                    self.renderer.snap_input_scroll_to_cursor();
                    return;
                }
                _ => {}
            }
        }

        // Readline-style Ctrl+* shortcuts (always active, both platforms).
        if self.modifiers.control_key() {
            if let Key::Character(s) = &event.logical_key {
                match s.as_str() {
                    "a" => {
                        self.input.move_home();
                        self.renderer.snap_input_scroll_to_cursor();
                        return;
                    }
                    "e" => {
                        self.input.move_end();
                        self.renderer.snap_input_scroll_to_cursor();
                        return;
                    }
                    "k" => {
                        self.input.kill_to_end();
                        self.renderer.snap_input_scroll_to_cursor();
                        return;
                    }
                    "u" => {
                        self.input.kill_to_start();
                        self.renderer.snap_input_scroll_to_cursor();
                        return;
                    }
                    "w" => {
                        self.input.delete_word_backward();
                        self.renderer.snap_input_scroll_to_cursor();
                        return;
                    }
                    _ => {}
                }
            }
        }

        match &event.logical_key {
            Key::Named(NamedKey::Enter) => {
                if self.modifiers.shift_key() {
                    // Shift+Enter: insert a newline into the input (multi-line).
                    self.input.insert('\n');
                } else if !self.input.is_empty() {
                    let text = self.input.submit();
                    self.input.push_history(text.clone());
                    self.route_input(text).await;
                }
            }
            Key::Named(NamedKey::Tab) => {
                if self.modifiers.shift_key() {
                    // Shift+Tab during an active cycle steps backwards through candidates.
                    // Otherwise, cycles input mode forward: Auto → Cmd → Agent → Auto.
                    if self.completion_cycle.is_some() {
                        self.try_autocomplete_step(true);
                    } else {
                        self.app_mode = match self.app_mode {
                            AppMode::Auto => AppMode::Cmd,
                            AppMode::Cmd => AppMode::Agent,
                            AppMode::Agent => AppMode::Auto,
                        };
                    }
                } else {
                    // Plain Tab: autocomplete the current input.
                    self.try_autocomplete_step(false);
                }
            }
            Key::Named(NamedKey::Backspace) => {
                self.input.delete_backward();
            }
            Key::Named(NamedKey::Delete) => {
                self.input.delete_forward();
            }
            Key::Named(NamedKey::ArrowUp) => {
                self.input.history_prev();
            }
            Key::Named(NamedKey::ArrowDown) => {
                self.input.history_next();
            }
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
        // After any cursor-moving or editing key, snap the input viewport so the cursor stays visible.
        self.renderer.snap_input_scroll_to_cursor();
    }

    fn copy_selection(&self) {
        // Drag-selected text (shell output cells or agent message buffer) takes priority
        // over block-level selection. Falls back to whole-block copy when no range is set.
        if let Some(text) = self.renderer.selected_text() {
            if !text.is_empty() {
                if let Ok(mut clipboard) = arboard::Clipboard::new() {
                    let _ = clipboard.set_text(text);
                }
                return;
            }
        }
        let Some(idx) = self.selected_block else {
            return;
        };
        let Some(block) = self.blocks.get(idx) else {
            return;
        };
        let text = match &block.content {
            BlockContent::ShellCommand { input, output, .. } => {
                if self.selected_sub_output {
                    // Copy only the output rows.
                    output
                        .rows
                        .iter()
                        .map(|row| {
                            row.cells
                                .iter()
                                .map(|c| c.grapheme.as_str())
                                .collect::<String>()
                        })
                        .collect::<Vec<_>>()
                        .join("\n")
                } else {
                    // Copy only the command.
                    input.clone()
                }
            }
            _ => block_to_text(block),
        };
        if text.is_empty() {
            return;
        }
        if let Ok(mut clipboard) = arboard::Clipboard::new() {
            let _ = clipboard.set_text(text);
        }
    }

    // ── Scrollback search ─────────────────────────────────────────────────────

    fn block_search_text(block: &Block) -> String {
        match &block.content {
            BlockContent::ShellCommand { input, output, .. } => {
                let mut s = input.clone();
                s.push('\n');
                for row in &output.rows {
                    for cell in &row.cells {
                        s.push_str(cell.grapheme.as_str());
                    }
                    s.push('\n');
                }
                s
            }
            BlockContent::AgentMessage { content_blocks, .. } => content_blocks
                .iter()
                .map(|cb| match cb {
                    beyonder_core::ContentBlock::Text { text } => text.clone(),
                    beyonder_core::ContentBlock::Code { code, .. } => code.clone(),
                    beyonder_core::ContentBlock::Thinking { thinking } => thinking.clone(),
                })
                .collect::<Vec<_>>()
                .join("\n"),
            BlockContent::Text { text } => text.clone(),
            BlockContent::ToolCall {
                tool_name,
                input,
                output,
                streaming_text,
                error,
                ..
            } => {
                let mut s = tool_name.clone();
                s.push(' ');
                s.push_str(&input.to_string());
                if let Some(o) = output {
                    s.push('\n');
                    s.push_str(o);
                }
                if let Some(st) = streaming_text {
                    s.push('\n');
                    s.push_str(st);
                }
                if let Some(e) = error {
                    s.push('\n');
                    s.push_str(e);
                }
                s
            }
            _ => String::new(),
        }
    }

    pub fn enter_search_mode(&mut self, prefill: &str) {
        if self.search_pattern.is_none() {
            self.search_saved_input = self.input.text.clone();
        }
        self.search_pattern = Some(prefill.to_string());
        self.input.select_all();
        self.input.delete_backward();
        for ch in prefill.chars() {
            self.input.insert(ch);
        }
        self.update_search_matches();
    }

    pub fn exit_search_mode(&mut self) {
        if self.search_pattern.is_none() {
            return;
        }
        self.search_pattern = None;
        self.search_matches.clear();
        self.search_current = None;
        self.input.select_all();
        self.input.delete_backward();
        let saved = std::mem::take(&mut self.search_saved_input);
        for ch in saved.chars() {
            self.input.insert(ch);
        }
        self.renderer.snap_input_scroll_to_cursor();
    }

    fn update_search_matches(&mut self) {
        let pat = self.input.text.clone();
        self.search_pattern = Some(pat.clone());
        self.search_matches.clear();
        if pat.is_empty() {
            self.search_current = None;
            return;
        }
        let Some(re) = regex::RegexBuilder::new(&pat)
            .case_insensitive(true)
            .build()
            .ok()
        else {
            self.search_current = None;
            return;
        };
        for (i, block) in self.blocks.iter().enumerate() {
            let text = Self::block_search_text(block);
            if re.is_match(&text) {
                self.search_matches.push(i);
            }
        }
        self.search_current = if self.search_matches.is_empty() {
            None
        } else {
            Some(0)
        };
        self.focus_current_match();
    }

    fn focus_current_match(&mut self) {
        let Some(cur) = self.search_current else {
            return;
        };
        let Some(&block_idx) = self.search_matches.get(cur) else {
            return;
        };
        if let Some(y) = self.renderer.block_top_y(block_idx) {
            let pad = 16.0 * self.renderer.scale_factor;
            self.renderer.viewport.scroll_to((y - pad).max(0.0));
        }
    }

    pub fn next_match(&mut self) {
        if self.search_matches.is_empty() {
            return;
        }
        let n = self.search_matches.len();
        self.search_current = Some(match self.search_current {
            None => 0,
            Some(i) => (i + 1) % n,
        });
        self.focus_current_match();
    }

    pub fn prev_match(&mut self) {
        if self.search_matches.is_empty() {
            return;
        }
        let n = self.search_matches.len();
        self.search_current = Some(match self.search_current {
            None => n - 1,
            Some(i) => (i + n - 1) % n,
        });
        self.focus_current_match();
    }

    async fn route_input(&mut self, text: String) {
        // Bare `exit` / `quit` quits the app (matches shell intuition).
        let trimmed = text.trim();
        if trimmed == "exit" || trimmed == "quit" {
            self.should_quit = true;
            return;
        }
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

    /// Tab handler. If a completion cycle is already active, step to the next
    /// (or previous, for `reverse`) candidate. Otherwise, start fresh: try slash
    /// command, agent mention, then shell-style completion.
    fn try_autocomplete_step(&mut self, reverse: bool) {
        if let Some(cycle) = self.completion_cycle.clone() {
            self.advance_cycle(cycle, reverse);
            return;
        }
        self.try_autocomplete();
    }

    /// Replace the cycle's token with the next/previous candidate and persist state.
    fn advance_cycle(&mut self, mut cycle: CompletionCycle, reverse: bool) {
        if cycle.candidates.is_empty() {
            return;
        }
        let n = cycle.candidates.len();
        let next = match cycle.index {
            None => {
                if reverse {
                    n - 1
                } else {
                    0
                }
            }
            Some(i) => {
                if reverse {
                    (i + n - 1) % n
                } else {
                    (i + 1) % n
                }
            }
        };
        cycle.index = Some(next);
        let replacement = cycle.candidates[next].clone();
        // Splice replacement in place of whatever is between token_start and current cursor.
        let cursor = self.input.cursor.min(self.input.text.len());
        let token_start = cycle.token_start.min(self.input.text.len());
        if cursor < token_start {
            return;
        }
        let mut new_text = String::with_capacity(self.input.text.len() + replacement.len());
        new_text.push_str(&self.input.text[..token_start]);
        new_text.push_str(&replacement);
        new_text.push_str(&self.input.text[cursor..]);
        self.input.text = new_text;
        self.input.cursor = token_start + replacement.len();
        self.completion_cycle = Some(cycle);
        self.renderer.snap_input_scroll_to_cursor();
    }

    /// Tab autocomplete: slash commands (/clea → /clear), agent mentions (@oll → @ollama).
    /// Single match → complete + trailing space; multiple matches → extend to longest common prefix.
    fn try_autocomplete(&mut self) {
        let text = self.input.text.clone();
        if let Some(rest) = text.strip_prefix('/') {
            let token: String = rest.chars().take_while(|c| !c.is_whitespace()).collect();
            if token.len() != rest.len() {
                return;
            } // already past the command word
            let matches = commands::filter(&token);
            if matches.is_empty() {
                return;
            }
            let names: Vec<&str> = matches.iter().map(|c| c.name).collect();
            let lcp = longest_common_prefix(&names);
            if matches.len() == 1 {
                self.input.text = format!("/{} ", names[0]);
            } else if lcp.len() > token.len() {
                self.input.text = format!("/{lcp}");
            } else {
                return;
            }
            self.input.cursor = self.input.text.len();
            self.renderer.snap_input_scroll_to_cursor();
            return;
        }
        if let Some(rest) = text.strip_prefix('@') {
            let token: String = rest.chars().take_while(|c| !c.is_whitespace()).collect();
            if token.len() != rest.len() {
                return;
            }
            let agents = self.supervisor.list_agents();
            let names: Vec<String> = agents
                .into_iter()
                .map(|a| a.name.clone())
                .filter(|n| n.starts_with(&token))
                .collect();
            if names.is_empty() {
                return;
            }
            let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
            let lcp = longest_common_prefix(&refs);
            if names.len() == 1 {
                self.input.text = format!("@{} ", names[0]);
            } else if lcp.len() > token.len() {
                self.input.text = format!("@{lcp}");
            } else {
                return;
            }
            self.input.cursor = self.input.text.len();
            self.renderer.snap_input_scroll_to_cursor();
            return;
        }
        // Shell-style completion: complete the token at the cursor as a path or command.
        self.shell_autocomplete();
    }

    /// File-path / command completion at the token under the cursor.
    /// Mirrors what zsh/bash do for the most common case: command in command position,
    /// path elsewhere. Multi-match extends to the longest common prefix; single match
    /// appends `/` for dirs or ` ` for files/commands.
    fn shell_autocomplete(&mut self) {
        let text = self.input.text.clone();
        let cursor = self.input.cursor.min(text.len());
        let token_start = text[..cursor]
            .rfind(|c: char| {
                c.is_whitespace() || matches!(c, '|' | ';' | '&' | '`' | '$' | '(' | '<' | '>')
            })
            .map(|i| i + text[i..].chars().next().unwrap().len_utf8())
            .unwrap_or(0);
        let token = &text[token_start..cursor];
        let prior = &text[..token_start];
        let is_command_pos = prior.trim().is_empty()
            || prior
                .trim_end()
                .ends_with(|c: char| matches!(c, '|' | ';' | '&'));

        let cwd = self.block_builder.cwd.clone();
        let candidates: Vec<String> = if !token.starts_with('~')
            && !token.starts_with('/')
            && !token.contains('/')
            && is_command_pos
        {
            command_completions(token)
        } else {
            path_completions(token, &cwd)
        };

        if candidates.is_empty() {
            return;
        }
        let refs: Vec<&str> = candidates.iter().map(|s| s.as_str()).collect();
        let lcp = longest_common_prefix(&refs);

        if candidates.len() == 1 {
            let only = &candidates[0];
            let replacement = if only.ends_with('/') {
                only.clone()
            } else {
                format!("{only} ")
            };
            self.splice_at(token_start, cursor, &replacement);
            return;
        }

        // Multiple matches.
        if lcp.len() > token.len() {
            // Extend to LCP first; arm cycle so the next Tab steps through full candidates.
            self.splice_at(token_start, cursor, &lcp);
            self.completion_cycle = Some(CompletionCycle {
                token_start,
                candidates,
                index: None,
            });
        } else {
            // LCP didn't advance — start cycling immediately on this Tab.
            let first = candidates[0].clone();
            self.splice_at(token_start, cursor, &first);
            self.completion_cycle = Some(CompletionCycle {
                token_start,
                candidates,
                index: Some(0),
            });
        }
    }

    /// Replace text[from..to] with `replacement` and place the cursor at the end of it.
    fn splice_at(&mut self, from: usize, to: usize, replacement: &str) {
        let to = to.min(self.input.text.len());
        if from > to {
            return;
        }
        let mut new_text = String::with_capacity(self.input.text.len() + replacement.len());
        new_text.push_str(&self.input.text[..from]);
        new_text.push_str(replacement);
        new_text.push_str(&self.input.text[to..]);
        self.input.text = new_text;
        self.input.cursor = from + replacement.len();
        self.renderer.snap_input_scroll_to_cursor();
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
        self.renderer.viewport.scroll_to_bottom();
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
        let existing_id = agents.iter().find(|a| a.name == name).map(|a| a.id.clone());

        if let Some(agent_id) = existing_id {
            self.push_pending_agent_block(agent_id.clone());
            if let Err(e) = self.supervisor.prompt_agent(&agent_id, &full_prompt) {
                error!("Failed to prompt agent {name}: {e}");
            }
        } else {
            let caps = CapabilitySet::default_coding_agent(self.session.working_directory.clone());
            let kind = match &self.config.provider {
                ProviderConfig::Ollama {
                    base_url,
                    api_key_env,
                } => {
                    // Env var takes precedence over config for cloud detection.
                    let (base_url, api_key_env) = if std::env::var("OLLAMA_API_KEY").is_ok() {
                        (
                            "https://ollama.com".to_string(),
                            Some("OLLAMA_API_KEY".to_string()),
                        )
                    } else {
                        (base_url.clone(), api_key_env.clone())
                    };
                    AgentKind::Ollama {
                        base_url,
                        model: self.active_model.clone(),
                        api_key_env,
                    }
                }
                ProviderConfig::LlamaCpp {
                    base_url,
                    api_key_env,
                } => AgentKind::LlamaCpp {
                    base_url: base_url.clone(),
                    model: self.active_model.clone(),
                    api_key_env: api_key_env.clone(),
                },
                ProviderConfig::Mlx {
                    base_url,
                    api_key_env,
                } => AgentKind::Mlx {
                    base_url: base_url.clone(),
                    model: self.active_model.clone(),
                    api_key_env: api_key_env.clone(),
                },
            };
            match self.supervisor.spawn_agent(name, kind, caps).await {
                Ok(agent_id) => {
                    self.capability_broker.register_agent(
                        agent_id.clone(),
                        CapabilitySet::default_coding_agent(self.session.working_directory.clone()),
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
            ["/find", rest @ ..] => {
                let pat = rest.join(" ");
                self.enter_search_mode(&pat);
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
                    let lines: Vec<String> = agents
                        .iter()
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
                let ids: Vec<_> = self
                    .supervisor
                    .list_agents()
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
                    "cmd" | "shell" => AppMode::Cmd,
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
            ["/theme"] => {
                let current = self.config.theme.clone();
                let mut lines = vec![
                    format!("Current theme: {}", current),
                    "Available:".to_string(),
                ];
                for name in beyonder_config::BUILTIN_THEMES {
                    lines.push(format!("  - {}", name));
                }
                self.push_text_block(lines.join("\n"));
            }
            ["/phone"] | ["/phone", "status"] => {
                let line = match &self.remote {
                    None => "phone bridge: off".to_string(),
                    Some(hub) => format!(
                        "phone bridge: on\nlisten :{}\nconnected: {}\npair: {}",
                        hub.port,
                        hub.is_connected(),
                        hub.pairing_url
                    ),
                };
                self.push_text_block(line);
            }
            ["/phone", "on"] => {
                if self.remote.is_some() {
                    self.push_text_block("phone bridge already running".into());
                } else {
                    match beyonder_remote::RemoteHub::start(
                        format!("{:?}", self.session.id),
                        self.active_model.clone(),
                        self.active_provider.clone(),
                    )
                    .await
                    {
                        Ok(hub) => {
                            let header =
                                format!("phone bridge started on :{}\nScan to pair:", hub.port);
                            let footer = format!("Or enter URL manually: {}", hub.pairing_url);
                            let bitmap = hub.qr_bitmap.clone();
                            self.remote = Some(hub);
                            self.remote_cursor = 0;
                            self.push_text_block(header);
                            self.push_qr_block(&bitmap);
                            self.push_text_block(footer);
                        }
                        Err(e) => {
                            error!("phone bridge: {e}");
                            self.push_text_block(format!("phone bridge failed: {e}"));
                        }
                    }
                }
            }
            ["/phone", "off"] => {
                if self.remote.take().is_some() {
                    self.push_text_block("phone bridge stopped".into());
                } else {
                    self.push_text_block("phone bridge was not running".into());
                }
            }
            ["/phone", "pair"] => {
                if let Some(hub) = &self.remote {
                    let header = format!("Scan to pair ({}):", hub.endpoint_label);
                    let footer = format!("Or enter URL manually: {}", hub.pairing_url);
                    let bitmap = hub.qr_bitmap.clone();
                    self.push_text_block(header);
                    self.push_qr_block(&bitmap);
                    self.push_text_block(footer);
                } else {
                    self.push_text_block("phone bridge off — run /phone on first".into());
                }
            }
            ["/phone", "tailscale"] => {
                if let Some(hub) = self.remote.as_mut() {
                    match hub.use_tailscale() {
                        Some(host) => {
                            let header = format!(
                                "phone bridge now advertised via Tailscale: {}",
                                host
                            );
                            let footer = hub.pairing_url.clone();
                            let bitmap = hub.qr_bitmap.clone();
                            self.push_text_block(header);
                            self.push_qr_block(&bitmap);
                            self.push_text_block(footer);
                        }
                        None => self.push_text_block(
                            "tailscale not installed or not logged in — `tailscale status` must work".into(),
                        ),
                    }
                } else {
                    self.push_text_block("phone bridge off — run /phone on first".into());
                }
            }
            ["/phone", "ngrok"] => {
                if let Some(hub) = self.remote.as_mut() {
                    match hub.use_ngrok().await {
                        Ok(host) => {
                            let header = format!("phone bridge tunneled via ngrok: wss://{}", host);
                            let footer = hub.pairing_url.clone();
                            let bitmap = hub.qr_bitmap.clone();
                            self.push_text_block(header);
                            self.push_qr_block(&bitmap);
                            self.push_text_block(footer);
                        }
                        Err(e) => self.push_text_block(format!("ngrok failed: {e}")),
                    }
                } else {
                    self.push_text_block("phone bridge off — run /phone on first".into());
                }
            }

            ["/theme", name] => {
                let theme = beyonder_config::theme_by_name(name);
                self.config.theme = theme.name.to_string();
                if let Err(e) = self.config.save() {
                    warn!("Failed to save config: {e}");
                }
                self.renderer.set_theme(theme);
                self.push_text_block(format!("Theme: {}", theme.name));
            }

            _ => {
                warn!("Unknown command: {text}");
                self.push_text_block(format!(
                    "Unknown command: {text}\nType /help for a list of commands."
                ));
            }
        }
    }

    /// Push a Human/User prompt echo block so the user can see what they sent.
    fn push_human_prompt_block(&mut self, text: String) {
        use beyonder_core::{
            BlockContent, BlockId, BlockKind, BlockStatus, ContentBlock, MessageRole,
            ProvenanceChain,
        };
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
        use beyonder_core::{
            BlockContent, BlockId, BlockKind, BlockStatus, MessageRole, ProvenanceChain,
        };
        let now = chrono::Utc::now();
        let block = Block {
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
    /// Re-read config.toml from disk and diff against the live config.
    /// Theme changes take effect immediately; model / provider changes apply
    /// on the next agent spawn (mirrors the `/model` and `/provider` commands).
    /// Font changes require a restart — a one-shot notice is posted.
    fn apply_config_reload(&mut self) {
        let new_config = BeyonderConfig::load_or_default();
        let mut notes: Vec<String> = Vec::new();

        if new_config.theme != self.config.theme {
            self.renderer.set_theme(new_config.resolved_theme());
            notes.push(format!("theme -> {}", new_config.theme));
        }
        if new_config.model != self.config.model {
            self.active_model = new_config.model.clone();
            notes.push(format!("model -> {} (next spawn)", new_config.model));
        }
        let new_provider_name = new_config.provider.name();
        let old_provider_name = self.config.provider.name();
        if new_provider_name != old_provider_name {
            self.active_provider = new_provider_name.to_string();
            notes.push(format!("provider -> {} (next spawn)", new_provider_name));
        }
        if new_config.font.family != self.config.font.family
            || (new_config.font.size - self.config.font.size).abs() > f32::EPSILON
        {
            notes.push("font change requires restart".to_string());
        }

        self.config = new_config;
        if !notes.is_empty() {
            self.push_text_block(format!("Config reloaded: {}", notes.join(", ")));
        }
    }

    /// Push an empty Text block that the renderer paints as a QR bitmap.
    /// Used for `/phone` pairing codes — in-line glyphs can't form scannable
    /// QR modules because glyphon line-height leaves gaps between rows.
    fn push_qr_block(&mut self, bitmap: &beyonder_remote::QrBitmap) {
        use beyonder_core::{BlockContent, BlockId, BlockKind, BlockStatus, ProvenanceChain};
        let now = chrono::Utc::now();
        let id = BlockId::new();
        let block = beyonder_core::Block {
            id: id.clone(),
            kind: BlockKind::System,
            parent_id: None,
            agent_id: None,
            session_id: self.session.id.clone(),
            status: BlockStatus::Completed,
            content: BlockContent::Text {
                text: String::new(),
            },
            created_at: now,
            updated_at: now,
            provenance: ProvenanceChain::default(),
        };
        self.blocks.push(block.clone());
        self.renderer.blocks.push(block);
        self.renderer.set_qr_block(
            id,
            beyonder_gpu::renderer::QrBitmap {
                width: bitmap.width,
                modules: bitmap.modules.clone(),
            },
        );
        if self.renderer.viewport.pinned_to_bottom {
            self.renderer.viewport.scroll_to_bottom();
        }
    }

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
        if self.renderer.viewport.pinned_to_bottom {
            self.renderer.viewport.scroll_to_bottom();
        }
    }

    /// Poll async channels and update state. Call on each event loop tick.
    /// Drain all async event sources. Returns `true` if any work was done that
    /// requires a redraw (PTY output, agent events, config reload, remote msgs).
    pub async fn tick(&mut self) -> bool {
        let mut had_work = false;

        // Config hot-reload: drain all pending file-watcher events (notify fires
        // several per save on most platforms). Apply at most one reload per tick.
        let mut config_changed = false;
        if let Some(rx) = &self.config_reload_rx {
            while rx.try_recv().is_ok() {
                config_changed = true;
            }
        }
        if config_changed {
            self.apply_config_reload();
            had_work = true;
        }

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
        if had_pty_output {
            had_work = true;
        }
        // Collect OSC 52 read-query responses before the mutable feed loop.
        let osc52_responses: Vec<String> = pty_output
            .iter()
            .filter_map(|b| self.handle_osc52(b))
            .collect();
        for bytes in pty_output {
            self.term_grid.feed(&bytes);
            for event in self.block_builder.feed(&bytes) {
                self.handle_build_event(event);
            }
        }
        for response in osc52_responses {
            self.write_to_pty(&response);
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
        // Treat name-detected interactive CLIs (claude) the same as alt-screen
        // TUIs — the renderer hides the input bar for them, so the PTY should
        // own the full window too.
        let interactive_cli = self
            .block_builder
            .running_command_name()
            .map(|name| matches!(name, "claude" | "claude-code"))
            .unwrap_or(false);
        let tui_now = self.term_grid.tui_active() || interactive_cli;
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
                            self.agent_running_tool
                                .insert(agent_id.clone(), name.clone());
                            // Finalize any in-flight agent text block (empty → removed).
                            self.finalize_agent_block(&agent_id);
                            // Show the tool call immediately so the user sees what's running.
                            self.push_tool_call_block(&agent_id, id, name, input);
                        }
                        AgentEvent::ToolResult {
                            id,
                            name: _,
                            output,
                            is_error,
                        } => {
                            self.complete_tool_call_block(&id, output, is_error);
                            // Push a new empty Running agent block so the spinner stays
                            // visible while we wait for the LLM to resume. The tool name
                            // in agent_running_tool (not cleared here) keeps the label
                            // showing. finalize_agent_block will drop this block if it's
                            // still empty when the next ToolCallRequested fires.
                            self.push_pending_agent_block(agent_id.clone());
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
            had_work = true;
            self.renderer.blocks = self.blocks.clone();
            self.renderer.agent_running_tool = self.agent_running_tool.clone();
        }

        // Drain broker events.
        while let Ok(event) = self.broker_rx.try_recv() {
            had_work = true;
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

        // /phone bridge: drain inbound commands, then push any new/changed blocks.
        // Architecture: the agent runs on the phone (local MLX or cloud). The
        // terminal is a pure executor — `RunCommand` writes to the shell PTY,
        // `Prompt` is kept as a legacy fallback that invokes the local agent.
        let mut remote_prompts: Vec<String> = vec![];
        let mut remote_shell: Vec<String> = vec![];
        let mut remote_pty_input: Vec<Vec<u8>> = vec![];
        let mut remote_pty_resize: Option<(u16, u16)> = None;
        let mut remote_interrupt = false;
        let mut remote_switch_tab: Option<usize> = None;
        let mut remote_new_tab = false;
        let mut remote_close_tab = false;
        if let Some(hub) = &self.remote {
            let gen = hub.connect_generation();
            if gen != self.remote_connect_gen {
                self.remote_connect_gen = gen;
                self.remote_cursor = 0;
                self.remote_pty_dims = None;
                self.remote_prev_cells.clear();
                eprintln!(
                    "[remote] phone reconnected (gen={gen}), resending all {} blocks",
                    self.blocks.len()
                );
                self.broadcast_tab_list();
            }
            let mut inbox = vec![];
            hub.poll_inbound(&mut inbox);
            if !inbox.is_empty() {
                had_work = true;
            }
            for msg in inbox {
                match msg {
                    beyonder_remote::ClientMsg::Prompt { text } => remote_prompts.push(text),
                    beyonder_remote::ClientMsg::RunCommand { cmd } => remote_shell.push(cmd),
                    beyonder_remote::ClientMsg::PtyInput { bytes } => remote_pty_input.push(bytes),
                    beyonder_remote::ClientMsg::PtyResize { cols, rows } => {
                        eprintln!("[remote] PtyResize {cols}x{rows}");
                        remote_pty_resize = Some((cols, rows));
                    }
                    beyonder_remote::ClientMsg::Interrupt => remote_interrupt = true,
                    beyonder_remote::ClientMsg::SwitchTab { index } => {
                        remote_switch_tab = Some(index)
                    }
                    beyonder_remote::ClientMsg::NewTab => remote_new_tab = true,
                    beyonder_remote::ClientMsg::CloseTab => remote_close_tab = true,
                    beyonder_remote::ClientMsg::ToolHint { .. }
                    | beyonder_remote::ClientMsg::Auth { .. }
                    | beyonder_remote::ClientMsg::Ping { .. } => {}
                }
            }
            // Live PTY mirror — throttled to ~3fps to keep payload manageable.
            if self.remote_last_pty_frame.elapsed() >= std::time::Duration::from_millis(333) {
                self.remote_last_pty_frame = std::time::Instant::now();
                let cells: Vec<Vec<beyonder_remote::PtyCell>> = self
                    .term_grid
                    .cell_grid()
                    .into_iter()
                    .map(|row| {
                        row.into_iter()
                            .map(|c| {
                                let to_u8 = |f: f32| (f.clamp(0.0, 1.0) * 255.0) as u8;
                                beyonder_remote::PtyCell {
                                    g: c.grapheme,
                                    fg: [to_u8(c.fg[0]), to_u8(c.fg[1]), to_u8(c.fg[2])],
                                    bg: c.bg.map(|b| [to_u8(b[0]), to_u8(b[1]), to_u8(b[2])]),
                                    bold: c.bold,
                                }
                            })
                            .collect()
                    })
                    .collect();
                let (cur_row, cur_col) = self.term_grid.cursor_pos();

                // Diff against previous frame — send only changed cells.
                if let Some(changes) =
                    beyonder_remote::protocol::compute_frame_diff(&self.remote_prev_cells, &cells)
                {
                    let total_cells: usize = cells.iter().map(|r| r.len()).sum();
                    if !changes.is_empty() && changes.len() < total_cells / 2 {
                        let _ = hub.send(beyonder_remote::ServerMsg::PtyFrameDiff(
                            beyonder_remote::PtyFrameDiff {
                                cursor_col: cur_col as u16,
                                cursor_row: cur_row as u16,
                                changes,
                            },
                        ));
                    } else if changes.is_empty() {
                        // No changes — skip sending entirely.
                    } else {
                        let _ = hub.send(beyonder_remote::ServerMsg::PtyFrame(
                            beyonder_remote::PtyFrame {
                                cols: self.term_grid.cols as u16,
                                rows: self.term_grid.rows as u16,
                                cursor_col: cur_col as u16,
                                cursor_row: cur_row as u16,
                                cells: cells.clone(),
                            },
                        ));
                    }
                } else {
                    // First frame or grid resized — send full.
                    let _ = hub.send(beyonder_remote::ServerMsg::PtyFrame(
                        beyonder_remote::PtyFrame {
                            cols: self.term_grid.cols as u16,
                            rows: self.term_grid.rows as u16,
                            cursor_col: cur_col as u16,
                            cursor_row: cur_row as u16,
                            cells: cells.clone(),
                        },
                    ));
                }
                self.remote_prev_cells = cells;
            }

            // Broadcast any blocks we haven't sent yet. Cap per-tick to avoid
            // overflowing the broadcast channel during history replay.
            let len = self.blocks.len();
            let batch_end = len.min(self.remote_cursor + 200);
            for idx in self.remote_cursor..batch_end {
                let block = self.blocks[idx].clone();
                let _ = hub.send(beyonder_remote::ServerMsg::BlockAppended(block));
            }
            // Advance cursor past any completed blocks at the tail.
            while self.remote_cursor < batch_end {
                let b = &self.blocks[self.remote_cursor];
                if matches!(b.status, BlockStatus::Running) {
                    let _ = hub.send(beyonder_remote::ServerMsg::BlockAppended(b.clone()));
                    break;
                }
                self.remote_cursor += 1;
            }
        }
        if remote_new_tab {
            self.new_tab();
        }
        if remote_close_tab {
            self.close_tab();
        }
        if let Some(idx) = remote_switch_tab {
            self.switch_tab(idx);
        }
        if remote_interrupt {
            self.supervisor.reset_all_conversations();
        }
        if let Some((cols, rows)) = remote_pty_resize {
            self.remote_pty_dims = Some((cols, rows));
        }
        if let Some((cols, rows)) = self.remote_pty_dims {
            // Apply every tick — window resize events keep trying to override.
            if self.term_grid.cols != cols as usize || self.term_grid.rows != rows as usize {
                self.term_grid.resize(cols as usize, rows as usize);
                self.block_builder
                    .set_grid_size(cols as usize, rows as usize);
                if let Some(pty) = &self.pty {
                    let _ = pty.resize(rows, cols);
                }
            }
        }
        for bytes in remote_pty_input {
            if let Some(pty) = &mut self.pty {
                let _ = pty.write(&bytes);
            }
        }
        for cmd in remote_shell {
            if cmd.starts_with('/') {
                self.handle_command(&cmd).await;
            } else {
                self.send_to_shell(cmd).await;
            }
        }
        // Phone-side Prompt messages are ignored — the agent runs on the
        // phone, not the terminal. Only RunCommand (shell execution) is
        // accepted from the phone. This prevents accidental agent invocation
        // on the Mac when the phone sends legacy Prompt messages.
        for text in remote_prompts {
            if text.starts_with('/') {
                self.handle_command(&text).await;
            } else {
                eprintln!(
                    "[remote] ignoring Prompt from phone (agent runs on phone): {}",
                    &text[..text.len().min(80)]
                );
            }
        }

        had_work
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
                        self.renderer.viewport.pinned_to_bottom = true;
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
                    if let BlockContent::ShellCommand {
                        ref input,
                        exit_code: Some(127),
                        ..
                    } = content
                    {
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
        if self.renderer.viewport.pinned_to_bottom {
            self.renderer.scroll_to_bottom();
        }
    }

    fn append_agent_text(&mut self, agent_id: &beyonder_core::AgentId, text: &str) {
        // Find the most recent Running agent block for this agent_id.
        let idx = self.blocks.iter().rposition(|b| {
            b.agent_id.as_ref() == Some(agent_id) && matches!(b.status, BlockStatus::Running)
        });

        if let Some(idx) = idx {
            let block = &mut self.blocks[idx];
            if let BlockContent::AgentMessage { content_blocks, .. } = &mut block.content {
                if let Some(beyonder_core::ContentBlock::Text { text: t }) =
                    content_blocks.last_mut()
                {
                    t.push_str(text);
                } else {
                    content_blocks.push(beyonder_core::ContentBlock::Text {
                        text: text.to_string(),
                    });
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
            b.agent_id.as_ref() == Some(agent_id) && matches!(b.status, BlockStatus::Running)
        }) {
            let is_empty = match &self.blocks[idx].content {
                beyonder_core::BlockContent::AgentMessage { content_blocks, .. } => {
                    content_blocks.is_empty()
                }
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
            if let beyonder_core::BlockContent::ToolCall {
                tool_use_id: tid, ..
            } = &b.content
            {
                tid == tool_use_id
            } else {
                false
            }
        }) {
            if let beyonder_core::BlockContent::ToolCall {
                output: out, error, ..
            } = &mut block.content
            {
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
        self.command_running =
            self.block_builder.is_running_command() || self.term_grid.tui_active();

        // Interactive CLIs that take over the terminal but don't use alt-screen
        // (e.g. `claude`) should hide the input bar just like nvim/htop do.
        let interactive_cli = self
            .block_builder
            .running_command_name()
            .map(|name| matches!(name, "claude" | "claude-code"))
            .unwrap_or(false);

        // Full-screen TUI takeover: alt-screen apps OR known interactive CLIs.
        self.renderer.tui_active = self.term_grid.tui_active() || interactive_cli;
        self.renderer.tui_cursor_shape = self.term_grid.cursor_shape_code();

        // Tell renderer which block (if any) is the live running command.
        self.renderer.running_block_idx = if self.block_builder.is_running_command() {
            self.blocks
                .iter()
                .rposition(|b| b.status == beyonder_core::BlockStatus::Running)
        } else {
            None
        };

        // Input bar: always show normally — no running state or color change.
        self.renderer.input_text = self.input.text.clone();
        self.renderer.input_cursor = self.input.cursor;
        self.renderer.input_all_selected = self.input.all_selected;
        self.renderer.input_preedit = self.ime_preedit.clone();
        self.renderer.input_mode_prefix = if let Some(pat) = &self.search_pattern {
            let total = self.search_matches.len();
            let cur = self.search_current.map(|i| i + 1).unwrap_or(0);
            let _ = pat;
            format!("find ({}/{}) ", cur, total)
        } else {
            match detect_mode(&self.input.text) {
                InputMode::Shell => String::new(),
                InputMode::Agent { name } => format!("@{} ", name),
                InputMode::Command { .. } => String::new(),
            }
        };
        self.renderer.search_match_blocks = self.search_matches.clone();
        self.renderer.search_current_match = self.search_current;
        self.renderer.input_running = self.block_builder.is_running_command();

        // Command palette — filter commands by what's been typed after the leading /.
        self.renderer.command_palette =
            if let InputMode::Command { ref cmd } = detect_mode(&self.input.text) {
                let matches = commands::filter(cmd);
                if matches.is_empty() {
                    None
                } else {
                    Some(
                        matches
                            .iter()
                            .map(|c| (c.usage.to_string(), c.description.to_string()))
                            .collect(),
                    )
                }
            } else {
                None
            };

        // Sync mode switcher label.
        self.renderer.mode_label = self.app_mode.label().to_string();

        // Sync tab strip state.
        self.renderer.tab_labels = self.tab_titles.clone();
        self.renderer.active_tab = self.active_tab;

        // Sync active model name.
        self.renderer.agent_model = self.active_model.clone();

        // Sync context pill labels.
        self.renderer.context_pills = vec![
            format!("conda: {}", self.current_conda),
            format!("node: {}", self.current_node),
            self.block_builder
                .cwd
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("~")
                .to_string(),
        ];

        let result = self.renderer.render();

        // Push the current caret rect to the OS IME as the candidate anchor.
        // Coordinates from the renderer are physical pixels; winit expects
        // logical pixels so we divide by scale_factor.
        let [cx, cy, cw, ch] = self.renderer.input_caret_rect;
        if cw > 0.0 && ch > 0.0 {
            let sf = self.renderer.scale_factor as f64;
            let lx = cx as f64 / sf;
            let ly = cy as f64 / sf;
            let lw = cw as f64 / sf;
            let lh = ch as f64 / sf;
            self.window.set_ime_cursor_area(
                winit::dpi::LogicalPosition::new(lx, ly),
                winit::dpi::LogicalSize::new(lw.max(1.0), lh.max(1.0)),
            );
        }

        result
    }
}

/// SGR mouse button code: Left=0, Middle=1, Right=2, None=3 (release placeholder).
fn sgr_button_code(
    btn: Option<winit::event::MouseButton>,
    _wheel_up: bool,
    _wheel_down: bool,
) -> u32 {
    use winit::event::MouseButton;
    match btn {
        Some(MouseButton::Left) => 0,
        Some(MouseButton::Middle) => 1,
        Some(MouseButton::Right) => 2,
        _ => 3,
    }
}
