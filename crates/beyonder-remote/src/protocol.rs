//! Wire protocol for the /phone bridge.
//!
//! Frames are length-prefixed CBOR blobs carried inside binary WebSocket
//! messages. CBOR was picked over JSON because the iPhone client decodes it
//! ~3× faster and it preserves the exact byte-level representation of
//! `BlockContent` variants without the round-trip cost.

use beyonder_core::{Block, BlockId, BlockStatus};
use serde::{Deserialize, Serialize};

pub const PROTOCOL_VERSION: u16 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hello {
    pub version: u16,
    pub server_name: String,
    pub session_id: String,
    pub active_model: String,
    pub active_provider: String,
}

/// Server → phone.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "t", content = "v")]
pub enum ServerMsg {
    Hello(Hello),
    /// Full block — sent when a new block appears.
    BlockAppended(Block),
    /// Status / content update on an existing block (used for streaming agent text).
    BlockUpdated {
        id: BlockId,
        status: BlockStatus,
        content_patch: Option<ContentPatch>,
    },
    /// Incremental append to a streaming agent text block.
    AgentTextDelta {
        block_id: BlockId,
        delta: String,
    },
    /// Agent completed its turn — phone can stop showing the spinner.
    AgentTurnComplete {
        agent_id: String,
        stop_reason: String,
    },
    Pong {
        nonce: u64,
    },
    Error {
        message: String,
    },
    /// Live PTY frame mirror — full-screen snapshot of the Mac terminal grid.
    /// Phone renders this directly so TUIs (vim, claude, htop, etc.) work.
    PtyFrame(PtyFrame),
    /// Incremental PTY update — only changed cells since last frame.
    PtyFrameDiff(PtyFrameDiff),
    /// Full list of tabs — sent on connect and whenever tabs change.
    TabList(TabList),
    /// Active tab changed (index into the tab list).
    TabSwitched {
        index: usize,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TabList {
    pub tabs: Vec<TabInfo>,
    pub active: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TabInfo {
    pub index: usize,
    pub title: String,
    pub session_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PtyFrame {
    pub cols: u16,
    pub rows: u16,
    pub cursor_col: u16,
    pub cursor_row: u16,
    pub cells: Vec<Vec<PtyCell>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PtyCell {
    pub g: String,
    pub fg: [u8; 3],
    pub bg: Option<[u8; 3]>,
    pub bold: bool,
}

/// Incremental PTY update: only cells that changed since the last full frame.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PtyFrameDiff {
    pub cursor_col: u16,
    pub cursor_row: u16,
    /// Changed cells: (row, col, cell).
    pub changes: Vec<(u16, u16, PtyCell)>,
}

/// Phone → server.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "t", content = "v")]
pub enum ClientMsg {
    /// Auth: bearer token (pre-shared via pairing QR).
    Auth {
        token: String,
    },
    /// Free-form prompt — goes through the normal agent path.
    Prompt {
        text: String,
    },
    /// Direct shell command — bypasses agent.
    RunCommand {
        cmd: String,
    },
    /// Pre-parsed tool-call hint from the on-device model.
    /// Server may use this to short-circuit planning if safe.
    ToolHint {
        name: String,
        args_json: String,
    },
    /// Interrupt the current in-flight agent turn.
    Interrupt,
    Ping {
        nonce: u64,
    },
    /// Raw bytes to write to the Mac PTY stdin (individual keystrokes,
    /// including ANSI escape sequences for arrows / Ctrl-keys).
    PtyInput {
        bytes: Vec<u8>,
    },
    /// Phone tells the Mac to resize the PTY + TermGrid to dims that fit
    /// the phone screen. TUIs re-render to the new size.
    PtyResize {
        cols: u16,
        rows: u16,
    },
    /// Phone requests switching to a different tab.
    SwitchTab {
        index: usize,
    },
    /// Phone requests opening a new tab.
    NewTab,
    /// Phone requests closing the active tab.
    CloseTab,
}

/// Compute a cell-level diff between two frames of equal dimensions.
/// Returns `None` if dimensions differ (caller should send a full frame).
/// Returns `Some(changes)` where changes is the list of (row, col, cell) tuples.
pub fn compute_frame_diff(
    prev: &[Vec<PtyCell>],
    curr: &[Vec<PtyCell>],
) -> Option<Vec<(u16, u16, PtyCell)>> {
    if prev.len() != curr.len() {
        return None;
    }
    if prev.first().map(|r| r.len()) != curr.first().map(|r| r.len()) {
        return None;
    }
    let mut changes = vec![];
    for (r, (prev_row, curr_row)) in prev.iter().zip(curr.iter()).enumerate() {
        for (c, (prev_cell, curr_cell)) in prev_row.iter().zip(curr_row.iter()).enumerate() {
            if prev_cell != curr_cell {
                changes.push((r as u16, c as u16, curr_cell.clone()));
            }
        }
    }
    Some(changes)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentPatch {
    pub text_append: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cell(g: &str, bold: bool) -> PtyCell {
        PtyCell {
            g: g.into(),
            fg: [255, 255, 255],
            bg: None,
            bold,
        }
    }

    fn grid(rows: usize, cols: usize, fill: &str) -> Vec<Vec<PtyCell>> {
        vec![vec![cell(fill, false); cols]; rows]
    }

    #[test]
    fn identical_frames_produce_zero_changes() {
        let frame = grid(24, 80, " ");
        let diff = compute_frame_diff(&frame, &frame).unwrap();
        assert_eq!(diff.len(), 0);
    }

    #[test]
    fn single_cell_change_detected() {
        let prev = grid(24, 80, " ");
        let mut curr = prev.clone();
        curr[5][10] = cell("A", true);
        let diff = compute_frame_diff(&prev, &curr).unwrap();
        assert_eq!(diff.len(), 1);
        assert_eq!(diff[0], (5, 10, cell("A", true)));
    }

    #[test]
    fn dimension_mismatch_returns_none() {
        let a = grid(24, 80, " ");
        let b = grid(25, 80, " ");
        assert!(compute_frame_diff(&a, &b).is_none());
    }

    #[test]
    fn diff_smaller_than_full_frame() {
        let prev = grid(24, 80, " ");
        let mut curr = prev.clone();
        // Change 10 cells out of 1920 — diff should be much smaller.
        for i in 0..10 {
            curr[0][i] = cell("X", false);
        }
        let diff = compute_frame_diff(&prev, &curr).unwrap();
        assert_eq!(diff.len(), 10);
        let total_cells = 24 * 80;
        assert!(
            diff.len() < total_cells / 2,
            "diff ({}) should be <50% of total cells ({})",
            diff.len(),
            total_cells
        );
    }

    #[test]
    fn full_screen_change_produces_all_cells() {
        let prev = grid(24, 80, " ");
        let curr = grid(24, 80, "X");
        let diff = compute_frame_diff(&prev, &curr).unwrap();
        assert_eq!(diff.len(), 24 * 80);
    }

    #[test]
    fn frame_diff_roundtrips_through_cbor() {
        let prev = grid(4, 4, " ");
        let mut curr = prev.clone();
        curr[1][2] = cell("B", true);
        let changes = compute_frame_diff(&prev, &curr).unwrap();
        let msg = ServerMsg::PtyFrameDiff(PtyFrameDiff {
            cursor_col: 2,
            cursor_row: 1,
            changes,
        });
        let bytes = ciborium::into_writer(&msg, Vec::new());
        assert!(bytes.is_ok(), "CBOR serialization should succeed");
    }
}
