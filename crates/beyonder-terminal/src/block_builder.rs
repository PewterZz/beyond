//! Converts raw PTY byte stream into structured ShellCommand blocks.
//! This is the core transformation: unstructured text → structured data.

use beyonder_core::{
    Block, BlockContent, BlockId, BlockKind, BlockStatus, Color, SessionId, TerminalCell,
    TerminalOutput, TerminalRow,
};

/// Returned from feed() — a completed block or a live update to an in-progress one.
#[derive(Debug)]
pub enum BuildEvent {
    /// A new block (running or completed).
    Block(Block),
    /// The running block's output changed — re-render it.
    LiveUpdate {
        block_id: BlockId,
        content: BlockContent,
    },
}
use std::path::PathBuf;
use std::time::Instant;

use crate::shell_hooks::markers;

/// State machine tracking where we are in a command lifecycle.
#[derive(Debug, Clone, PartialEq)]
enum BuildState {
    /// Waiting for a command to start (at the prompt).
    AtPrompt,
    /// A command is running; accumulating output.
    RunningCommand {
        command: String,
        output_bytes: Vec<u8>,
        started_at: Instant,
    },
}

/// Parses the raw PTY byte stream and emits completed ShellCommand blocks.
pub struct BlockBuilder {
    session_id: SessionId,
    pub cwd: PathBuf,
    state: BuildState,
    pending_block_id: Option<BlockId>,
    /// PTY dimensions — used to size the temp TermGrid for color-preserving output parsing.
    grid_cols: usize,
    grid_rows: usize,
}

impl BlockBuilder {
    pub fn new(session_id: SessionId, cwd: PathBuf) -> Self {
        Self {
            session_id,
            cwd,
            state: BuildState::AtPrompt,
            pending_block_id: None,
            grid_cols: 220,
            grid_rows: 50,
        }
    }

    /// Feed raw PTY bytes. Returns build events (new blocks or live updates).
    pub fn feed(&mut self, bytes: &[u8]) -> Vec<BuildEvent> {
        let mut events = vec![];
        let mut i = 0;

        while i < bytes.len() {
            // OSC 133 — FinalTerm standard. Carries less info than 633, so it acts
            // as a fallback when 633 isn't present. Repeated transitions are no-ops
            // because the state machine has already advanced.
            if bytes[i..].starts_with(b"\x1b]133;") {
                if let Some((marker, consumed)) = parse_osc_133(&bytes[i..]) {
                    match marker {
                        Osc133Marker::PromptStart => {
                            self.state = BuildState::AtPrompt;
                        }
                        Osc133Marker::CmdExecStart => {
                            // 133;C means "command output begins" — no command text.
                            // If 633;E hasn't already kicked us into RunningCommand,
                            // start with an unknown command string.
                            if matches!(self.state, BuildState::AtPrompt) {
                                let id = BlockId::new();
                                self.pending_block_id = Some(id.clone());
                                let mut block = Block::new(
                                    BlockKind::Human,
                                    self.session_id.clone(),
                                    BlockContent::ShellCommand {
                                        input: String::new(),
                                        output: TerminalOutput { rows: vec![] },
                                        exit_code: None,
                                        cwd: self.cwd.clone(),
                                        duration_ms: None,
                                    },
                                );
                                block.id = id;
                                block.status = BlockStatus::Running;
                                self.state = BuildState::RunningCommand {
                                    command: String::new(),
                                    output_bytes: vec![],
                                    started_at: Instant::now(),
                                };
                                events.push(BuildEvent::Block(block));
                            }
                        }
                        Osc133Marker::CmdEnd(code) => {
                            if let BuildState::RunningCommand {
                                command,
                                output_bytes,
                                started_at,
                            } = std::mem::replace(&mut self.state, BuildState::AtPrompt)
                            {
                                let duration_ms = started_at.elapsed().as_millis() as u64;
                                let output = parse_ansi_output(
                                    &output_bytes,
                                    self.grid_cols,
                                    self.grid_rows,
                                );
                                let content = BlockContent::ShellCommand {
                                    input: command,
                                    output,
                                    exit_code: Some(code),
                                    cwd: self.cwd.clone(),
                                    duration_ms: Some(duration_ms),
                                };
                                if let Some(id) = self.pending_block_id.take() {
                                    events.push(BuildEvent::LiveUpdate {
                                        block_id: id,
                                        content,
                                    });
                                }
                            }
                        }
                        Osc133Marker::CmdLineReady => {} // ignored — between prompt and exec
                    }
                    i += consumed;
                    continue;
                }
            }

            // Check for OSC marker sequences.
            if bytes[i..].starts_with(b"\x1b]633;") {
                if let Some((marker, consumed, _payload)) = parse_osc_633(&bytes[i..]) {
                    match marker {
                        OscMarker::CmdStart => {}
                        OscMarker::CmdText(cmd) => {
                            let id = BlockId::new();
                            self.pending_block_id = Some(id.clone());
                            // Emit a Running block immediately so the UI shows feedback.
                            let mut block = Block::new(
                                BlockKind::Human,
                                self.session_id.clone(),
                                BlockContent::ShellCommand {
                                    input: cmd.clone(),
                                    output: TerminalOutput { rows: vec![] },
                                    exit_code: None,
                                    cwd: self.cwd.clone(),
                                    duration_ms: None,
                                },
                            );
                            block.id = id;
                            block.status = BlockStatus::Running;
                            self.state = BuildState::RunningCommand {
                                command: cmd,
                                output_bytes: vec![],
                                started_at: Instant::now(),
                            };
                            events.push(BuildEvent::Block(block));
                        }
                        OscMarker::CmdEnd(exit_code) => {
                            if let BuildState::RunningCommand {
                                command,
                                output_bytes,
                                started_at,
                            } = std::mem::replace(&mut self.state, BuildState::AtPrompt)
                            {
                                let duration_ms = started_at.elapsed().as_millis() as u64;
                                let output = parse_ansi_output(
                                    &output_bytes,
                                    self.grid_cols,
                                    self.grid_rows,
                                );
                                let content = BlockContent::ShellCommand {
                                    input: command,
                                    output,
                                    exit_code: Some(exit_code),
                                    cwd: self.cwd.clone(),
                                    duration_ms: Some(duration_ms),
                                };
                                if let Some(id) = self.pending_block_id.take() {
                                    // Update the running block in place.
                                    events.push(BuildEvent::LiveUpdate {
                                        block_id: id,
                                        content,
                                    });
                                }
                            }
                        }
                        OscMarker::PromptStart => {
                            self.state = BuildState::AtPrompt;
                        }
                        OscMarker::Cwd(path) => {
                            self.cwd = PathBuf::from(path);
                        }
                    }
                    i += consumed;
                    continue;
                }
            }

            // Not a marker — accumulate into current command output.
            if let BuildState::RunningCommand { output_bytes, .. } = &mut self.state {
                output_bytes.push(bytes[i]);
            }
            i += 1;
        }

        events
    }

    pub fn set_cwd(&mut self, cwd: PathBuf) {
        self.cwd = cwd;
    }

    /// Update grid dimensions so output parsing matches the actual PTY size.
    /// Call this on PTY spawn and on window resize.
    pub fn set_grid_size(&mut self, cols: usize, rows: usize) {
        self.grid_cols = cols.max(40);
        self.grid_rows = rows.max(10);
    }

    /// True while a command is actively running (between CmdStart and CmdEnd markers).
    pub fn is_running_command(&self) -> bool {
        matches!(self.state, BuildState::RunningCommand { .. })
    }

    /// The leading word of the currently running command, if any.
    /// Used to detect known interactive CLIs that don't use alt-screen.
    pub fn running_command_name(&self) -> Option<&str> {
        if let BuildState::RunningCommand { command, .. } = &self.state {
            command.split_whitespace().next()
        } else {
            None
        }
    }

    /// Force-complete the running command — used when the PTY process dies without
    /// emitting a CmdEnd marker. Returns a LiveUpdate event if a command was running.
    pub fn force_complete(&mut self, exit_code: Option<u32>) -> Option<BuildEvent> {
        if let BuildState::RunningCommand {
            command,
            output_bytes,
            started_at,
        } = std::mem::replace(&mut self.state, BuildState::AtPrompt)
        {
            let duration_ms = started_at.elapsed().as_millis() as u64;
            let output = parse_ansi_output(&output_bytes, self.grid_cols, self.grid_rows);
            let content = BlockContent::ShellCommand {
                input: command,
                output,
                exit_code: exit_code.map(|c| c as i32),
                cwd: self.cwd.clone(),
                duration_ms: Some(duration_ms),
            };
            if let Some(id) = self.pending_block_id.take() {
                return Some(BuildEvent::LiveUpdate {
                    block_id: id,
                    content,
                });
            }
        }
        None
    }
}

#[derive(Debug)]
enum Osc133Marker {
    PromptStart,
    CmdLineReady,
    CmdExecStart,
    CmdEnd(i32),
}

fn parse_osc_133(bytes: &[u8]) -> Option<(Osc133Marker, usize)> {
    let prefix = b"\x1b]133;";
    if !bytes.starts_with(prefix) {
        return None;
    }
    let rest = &bytes[prefix.len()..];
    let end = rest
        .iter()
        .position(|&b| b == markers::BEL || b == b'\x1b')?;
    let payload = &rest[..end];
    let consumed = prefix.len() + end + 1;
    let marker = match payload {
        b"A" => Osc133Marker::PromptStart,
        b"B" => Osc133Marker::CmdLineReady,
        b"C" => Osc133Marker::CmdExecStart,
        _ if payload == b"D" => Osc133Marker::CmdEnd(0),
        _ if payload.starts_with(b"D;") => {
            let code = String::from_utf8_lossy(&payload[2..])
                .trim()
                .parse()
                .unwrap_or(0);
            Osc133Marker::CmdEnd(code)
        }
        _ => return None,
    };
    Some((marker, consumed))
}

#[derive(Debug)]
enum OscMarker {
    CmdStart,
    CmdText(String),
    CmdEnd(i32),
    PromptStart,
    Cwd(String),
}

/// Parse an OSC 633 sequence starting at `bytes[0]`.
/// Returns (marker, bytes_consumed, payload) if recognized.
fn parse_osc_633(bytes: &[u8]) -> Option<(OscMarker, usize, &[u8])> {
    let prefix = b"\x1b]633;";
    if !bytes.starts_with(prefix) {
        return None;
    }
    let rest = &bytes[prefix.len()..];

    // Find the BEL (0x07) or ST (ESC \) terminator.
    let end = rest
        .iter()
        .position(|&b| b == markers::BEL || b == b'\x1b')?;
    let payload = &rest[..end];
    let consumed = prefix.len() + end + 1; // +1 for BEL

    match payload {
        b"A" => Some((OscMarker::CmdStart, consumed, payload)),
        b"P" => Some((OscMarker::PromptStart, consumed, payload)),
        _ if payload.starts_with(b"E;") => {
            let cmd = String::from_utf8_lossy(&payload[2..]).to_string();
            Some((OscMarker::CmdText(cmd), consumed, payload))
        }
        _ if payload.starts_with(b"B;") => {
            let code_str = String::from_utf8_lossy(&payload[2..]);
            let code: i32 = code_str.trim().parse().unwrap_or(0);
            Some((OscMarker::CmdEnd(code), consumed, payload))
        }
        _ if payload.starts_with(b"P;Cwd=") => {
            let path = String::from_utf8_lossy(&payload[6..]).to_string();
            Some((OscMarker::Cwd(path), consumed, payload))
        }
        _ => None,
    }
}

/// Parse raw ANSI bytes into a structured TerminalOutput, preserving per-cell colors.
/// Feeds the bytes into a temporary TermGrid (same dimensions as the real PTY) so
/// that ANSI color codes, bold, italic, and cursor positioning are all honoured.
fn parse_ansi_output(bytes: &[u8], cols: usize, rows: usize) -> TerminalOutput {
    use crate::term_grid::TermGrid;

    let mut grid = TermGrid::new(cols, rows);
    grid.feed(bytes);
    let cells = grid.cell_grid();

    // Trim trailing blank rows.
    let last_content = cells.iter().rposition(|row| {
        row.iter().any(|c| {
            let fc = c.first_char();
            fc != ' ' && fc != '\0'
        })
    });
    let trimmed = match last_content {
        Some(i) => &cells[..=i],
        None => return TerminalOutput { rows: vec![] },
    };

    let rows_out = trimmed
        .iter()
        .map(|row| {
            // Trim trailing whitespace cells per row (same as the GPU renderer does).
            let last_vis = row.iter().rposition(|c| {
                let fc = c.first_char();
                fc != ' ' && fc != '\0'
            });
            let end = match last_vis {
                Some(i) => i + 1,
                None => 0,
            };
            TerminalRow {
                cells: row[..end]
                    .iter()
                    .map(|c| TerminalCell {
                        grapheme: c.grapheme.clone(),
                        fg: Some(Color {
                            r: (c.fg[0] * 255.0) as u8,
                            g: (c.fg[1] * 255.0) as u8,
                            b: (c.fg[2] * 255.0) as u8,
                        }),
                        bg: c.bg.map(|bg| Color {
                            r: (bg[0] * 255.0) as u8,
                            g: (bg[1] * 255.0) as u8,
                            b: (bg[2] * 255.0) as u8,
                        }),
                        bold: c.bold,
                        italic: c.italic,
                        underline: c.underline,
                        strikethrough: c.strikethrough,
                        link: c.link.as_ref().map(|a| a.as_ref().clone()),
                    })
                    .collect(),
            }
        })
        .collect();

    TerminalOutput { rows: rows_out }
}
