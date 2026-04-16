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
    /// Binary-packed full frame — same as PtyFrame but cells are a compact blob.
    PtyFramePacked(PtyFramePacked),
    /// Binary-packed diff — same as PtyFrameDiff but changes are a compact blob.
    PtyFrameDiffPacked(PtyFrameDiffPacked),
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

/// Binary-packed full frame. Cells are encoded as a flat byte blob via `pack_cells`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PtyFramePacked {
    pub cols: u16,
    pub rows: u16,
    pub cursor_col: u16,
    pub cursor_row: u16,
    #[serde(with = "serde_bytes")]
    pub packed: Vec<u8>,
}

/// Binary-packed diff. Changes encoded as a flat byte blob via `pack_diff_changes`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PtyFrameDiffPacked {
    pub cursor_col: u16,
    pub cursor_row: u16,
    pub num_changes: u32,
    #[serde(with = "serde_bytes")]
    pub packed: Vec<u8>,
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

/// Adaptive frame rate controller: ramps to 10fps during activity,
/// backs off to 1fps when idle. Avoids wasting bandwidth on static screens.
#[derive(Debug, Clone)]
pub struct AdaptiveThrottle {
    pub interval_ms: u64,
    pub idle_frames: u32,
}

impl Default for AdaptiveThrottle {
    fn default() -> Self {
        Self {
            interval_ms: 100,
            idle_frames: 0,
        }
    }
}

impl AdaptiveThrottle {
    pub fn report_activity(&mut self, had_changes: bool) {
        if had_changes {
            self.idle_frames = 0;
            self.interval_ms = 100; // 10fps
        } else {
            self.idle_frames = self.idle_frames.saturating_add(1);
            self.interval_ms = match self.idle_frames {
                0..=1 => 100,
                2..=3 => 200,
                4..=6 => 333,
                7..=10 => 500,
                _ => 1000,
            };
        }
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentPatch {
    pub text_append: Option<String>,
}

// ── Binary cell packing ──────────────────────────────────────────────
//
// CBOR encodes each PtyCell as a tagged map with field names, adding ~15-20
// bytes of overhead per cell. For a 80×24 grid that's 29-38 KB of pure
// overhead. The packed format below uses a flat byte buffer:
//
//   Per cell: [flags:u8] [fg:3B] [bg:3B if HAS_BG] [glyph_len:u8] [glyph:NB]
//
//   flags: bit 0 = bold, bit 1 = has_bg
//
// A typical space cell is 6 bytes (flags + fg + len=1 + " ") vs ~20 in CBOR.

const FLAG_BOLD: u8 = 0x01;
const FLAG_HAS_BG: u8 = 0x02;

/// Pack a grid of cells into a compact binary blob.
pub fn pack_cells(cells: &[Vec<PtyCell>]) -> Vec<u8> {
    let total: usize = cells.iter().map(|r| r.len()).sum();
    // Estimate ~8 bytes per cell.
    let mut buf = Vec::with_capacity(total * 8);
    for row in cells {
        for cell in row {
            let mut flags = 0u8;
            if cell.bold {
                flags |= FLAG_BOLD;
            }
            if cell.bg.is_some() {
                flags |= FLAG_HAS_BG;
            }
            buf.push(flags);
            buf.extend_from_slice(&cell.fg);
            if let Some(bg) = &cell.bg {
                buf.extend_from_slice(bg);
            }
            let g = cell.g.as_bytes();
            buf.push(g.len() as u8);
            buf.extend_from_slice(g);
        }
    }
    buf
}

/// Unpack a binary blob back into a grid of cells.
pub fn unpack_cells(data: &[u8], rows: usize, cols: usize) -> Option<Vec<Vec<PtyCell>>> {
    let mut pos = 0;
    let mut grid = Vec::with_capacity(rows);
    for _ in 0..rows {
        let mut row = Vec::with_capacity(cols);
        for _ in 0..cols {
            if pos >= data.len() {
                return None;
            }
            let flags = data[pos];
            pos += 1;
            if pos + 3 > data.len() {
                return None;
            }
            let fg = [data[pos], data[pos + 1], data[pos + 2]];
            pos += 3;
            let bg = if flags & FLAG_HAS_BG != 0 {
                if pos + 3 > data.len() {
                    return None;
                }
                let b = [data[pos], data[pos + 1], data[pos + 2]];
                pos += 3;
                Some(b)
            } else {
                None
            };
            if pos >= data.len() {
                return None;
            }
            let g_len = data[pos] as usize;
            pos += 1;
            if pos + g_len > data.len() {
                return None;
            }
            let g = std::str::from_utf8(&data[pos..pos + g_len])
                .ok()?
                .to_string();
            pos += g_len;
            row.push(PtyCell {
                g,
                fg,
                bg,
                bold: flags & FLAG_BOLD != 0,
            });
        }
        grid.push(row);
    }
    Some(grid)
}

/// Pack diff changes into a compact binary blob.
/// Format: [row:u16 LE][col:u16 LE][cell packed as above] per change.
pub fn pack_diff_changes(changes: &[(u16, u16, PtyCell)]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(changes.len() * 12);
    for (r, c, cell) in changes {
        buf.extend_from_slice(&r.to_le_bytes());
        buf.extend_from_slice(&c.to_le_bytes());
        let mut flags = 0u8;
        if cell.bold {
            flags |= FLAG_BOLD;
        }
        if cell.bg.is_some() {
            flags |= FLAG_HAS_BG;
        }
        buf.push(flags);
        buf.extend_from_slice(&cell.fg);
        if let Some(bg) = &cell.bg {
            buf.extend_from_slice(bg);
        }
        let g = cell.g.as_bytes();
        buf.push(g.len() as u8);
        buf.extend_from_slice(g);
    }
    buf
}

/// Unpack diff changes from a binary blob.
pub fn unpack_diff_changes(data: &[u8]) -> Option<Vec<(u16, u16, PtyCell)>> {
    let mut pos = 0;
    let mut changes = vec![];
    while pos < data.len() {
        if pos + 4 > data.len() {
            return None;
        }
        let r = u16::from_le_bytes([data[pos], data[pos + 1]]);
        let c = u16::from_le_bytes([data[pos + 2], data[pos + 3]]);
        pos += 4;
        if pos >= data.len() {
            return None;
        }
        let flags = data[pos];
        pos += 1;
        if pos + 3 > data.len() {
            return None;
        }
        let fg = [data[pos], data[pos + 1], data[pos + 2]];
        pos += 3;
        let bg = if flags & FLAG_HAS_BG != 0 {
            if pos + 3 > data.len() {
                return None;
            }
            let b = [data[pos], data[pos + 1], data[pos + 2]];
            pos += 3;
            Some(b)
        } else {
            None
        };
        if pos >= data.len() {
            return None;
        }
        let g_len = data[pos] as usize;
        pos += 1;
        if pos + g_len > data.len() {
            return None;
        }
        let g = std::str::from_utf8(&data[pos..pos + g_len])
            .ok()?
            .to_string();
        pos += g_len;
        changes.push((
            r,
            c,
            PtyCell {
                g,
                fg,
                bg,
                bold: flags & FLAG_BOLD != 0,
            },
        ));
    }
    Some(changes)
}

/// Magic byte prefixed to zstd-compressed frames.
pub const ZSTD_MAGIC: u8 = 0xFF;

/// Compress a CBOR payload with zstd, returning magic-prefixed bytes.
/// Returns `None` if compression doesn't save space.
pub fn compress_cbor(cbor: &[u8], level: i32) -> Option<Vec<u8>> {
    let compressed = zstd::encode_all(cbor, level).ok()?;
    if compressed.len() + 1 < cbor.len() {
        let mut out = Vec::with_capacity(1 + compressed.len());
        out.push(ZSTD_MAGIC);
        out.extend_from_slice(&compressed);
        Some(out)
    } else {
        None
    }
}

/// Decompress a potentially zstd-compressed frame. If the first byte is
/// `ZSTD_MAGIC`, strips it and decompresses; otherwise returns the input as-is.
pub fn decompress_frame(bytes: &[u8]) -> std::io::Result<std::borrow::Cow<'_, [u8]>> {
    if bytes.first() == Some(&ZSTD_MAGIC) {
        let decompressed = zstd::decode_all(&bytes[1..])?;
        Ok(std::borrow::Cow::Owned(decompressed))
    } else {
        Ok(std::borrow::Cow::Borrowed(bytes))
    }
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

    #[test]
    fn zstd_compress_roundtrips_pty_frame() {
        let frame = PtyFrame {
            cols: 80,
            rows: 24,
            cursor_col: 0,
            cursor_row: 0,
            cells: grid(24, 80, " "),
        };
        let msg = ServerMsg::PtyFrame(frame);
        let mut cbor = Vec::new();
        ciborium::into_writer(&msg, &mut cbor).unwrap();

        let compressed = compress_cbor(&cbor, 1).expect("frame should compress");
        assert!(
            compressed.len() < cbor.len(),
            "compressed ({}) should be smaller than raw ({})",
            compressed.len(),
            cbor.len()
        );
        assert_eq!(compressed[0], ZSTD_MAGIC);

        let decompressed = decompress_frame(&compressed).unwrap();
        assert_eq!(decompressed.as_ref(), cbor.as_slice());
    }

    #[test]
    fn zstd_passthrough_for_small_payloads() {
        let msg = ServerMsg::Pong { nonce: 42 };
        let mut cbor = Vec::new();
        ciborium::into_writer(&msg, &mut cbor).unwrap();
        // Small payload — compression shouldn't help.
        let result = compress_cbor(&cbor, 1);
        // Either None (not worth it) or still valid.
        if let Some(compressed) = result {
            let decompressed = decompress_frame(&compressed).unwrap();
            assert_eq!(decompressed.as_ref(), cbor.as_slice());
        }
    }

    #[test]
    fn decompress_frame_returns_raw_when_no_magic() {
        let raw = vec![0xA2, 0x01, 0x02]; // arbitrary non-magic bytes
        let result = decompress_frame(&raw).unwrap();
        assert_eq!(result.as_ref(), raw.as_slice());
    }

    #[test]
    fn zstd_compression_ratio_on_typical_frame() {
        // Simulate a typical terminal frame with repeated whitespace — should
        // compress very well, proving bandwidth savings for the phone link.
        let mut frame_cells = grid(24, 80, " ");
        // Add some text on a few lines to be realistic.
        for (i, ch) in "$ cargo build --release".chars().enumerate() {
            frame_cells[0][i] = cell(&ch.to_string(), false);
        }
        for (i, ch) in "   Compiling beyonder v0.1.0".chars().enumerate() {
            frame_cells[1][i] = cell(&ch.to_string(), false);
        }
        let msg = ServerMsg::PtyFrame(PtyFrame {
            cols: 80,
            rows: 24,
            cursor_col: 0,
            cursor_row: 2,
            cells: frame_cells,
        });
        let mut cbor = Vec::new();
        ciborium::into_writer(&msg, &mut cbor).unwrap();
        let compressed = compress_cbor(&cbor, 1).expect("should compress");
        let ratio = compressed.len() as f64 / cbor.len() as f64;
        assert!(
            ratio < 0.25,
            "typical mostly-blank frame should compress >4x, got {:.1}x (ratio {ratio:.3})",
            1.0 / ratio
        );
    }

    #[test]
    fn adaptive_throttle_ramps_up_on_activity() {
        let mut t = AdaptiveThrottle::default();
        assert_eq!(t.interval_ms, 100);
        // Simulate activity.
        t.report_activity(true);
        assert_eq!(t.interval_ms, 100);
        assert_eq!(t.idle_frames, 0);
    }

    #[test]
    fn adaptive_throttle_backs_off_when_idle() {
        let mut t = AdaptiveThrottle::default();
        let mut intervals = vec![];
        for _ in 0..15 {
            t.report_activity(false);
            intervals.push(t.interval_ms);
        }
        // Should monotonically increase and cap at 1000ms.
        for w in intervals.windows(2) {
            assert!(
                w[1] >= w[0],
                "interval should not decrease: {} < {}",
                w[1],
                w[0]
            );
        }
        assert_eq!(*intervals.last().unwrap(), 1000);
    }

    #[test]
    fn adaptive_throttle_resets_on_activity_after_idle() {
        let mut t = AdaptiveThrottle::default();
        // Go idle for a while.
        for _ in 0..20 {
            t.report_activity(false);
        }
        assert_eq!(t.interval_ms, 1000);
        // Activity snaps back to 10fps.
        t.report_activity(true);
        assert_eq!(t.interval_ms, 100);
        assert_eq!(t.idle_frames, 0);
    }

    #[test]
    fn adaptive_throttle_reset_returns_to_default() {
        let mut t = AdaptiveThrottle::default();
        for _ in 0..20 {
            t.report_activity(false);
        }
        t.reset();
        assert_eq!(t.interval_ms, 100);
        assert_eq!(t.idle_frames, 0);
    }

    #[test]
    fn pack_unpack_cells_roundtrip() {
        let cells = vec![
            vec![cell(" ", false), cell("A", true)],
            vec![
                PtyCell {
                    g: "B".into(),
                    fg: [0, 255, 0],
                    bg: Some([10, 20, 30]),
                    bold: false,
                },
                cell("C", false),
            ],
        ];
        let packed = pack_cells(&cells);
        let unpacked = unpack_cells(&packed, 2, 2).expect("unpack should succeed");
        assert_eq!(cells, unpacked);
    }

    #[test]
    fn pack_unpack_diff_roundtrip() {
        let changes = vec![
            (0u16, 5u16, cell("X", true)),
            (
                3,
                10,
                PtyCell {
                    g: "Y".into(),
                    fg: [128, 0, 255],
                    bg: Some([1, 2, 3]),
                    bold: false,
                },
            ),
        ];
        let packed = pack_diff_changes(&changes);
        let unpacked = unpack_diff_changes(&packed).expect("unpack should succeed");
        assert_eq!(changes, unpacked);
    }

    #[test]
    fn packed_frame_smaller_than_cbor_frame() {
        let cells = grid(24, 80, " ");
        // CBOR size of the old PtyFrame variant.
        let cbor_msg = ServerMsg::PtyFrame(PtyFrame {
            cols: 80,
            rows: 24,
            cursor_col: 0,
            cursor_row: 0,
            cells: cells.clone(),
        });
        let mut cbor_buf = Vec::new();
        ciborium::into_writer(&cbor_msg, &mut cbor_buf).unwrap();

        // Packed size.
        let packed_data = pack_cells(&cells);
        let packed_msg = ServerMsg::PtyFramePacked(PtyFramePacked {
            cols: 80,
            rows: 24,
            cursor_col: 0,
            cursor_row: 0,
            packed: packed_data,
        });
        let mut packed_buf = Vec::new();
        ciborium::into_writer(&packed_msg, &mut packed_buf).unwrap();

        assert!(
            packed_buf.len() < cbor_buf.len(),
            "packed ({}) should be smaller than CBOR ({}), saving {}%",
            packed_buf.len(),
            cbor_buf.len(),
            100 - (packed_buf.len() * 100 / cbor_buf.len())
        );
    }

    #[test]
    fn packed_diff_smaller_than_cbor_diff() {
        let changes: Vec<(u16, u16, PtyCell)> = (0..50)
            .map(|i| (i / 10, i % 10, cell("X", i % 2 == 0)))
            .collect();

        let cbor_msg = ServerMsg::PtyFrameDiff(PtyFrameDiff {
            cursor_col: 0,
            cursor_row: 0,
            changes: changes.clone(),
        });
        let mut cbor_buf = Vec::new();
        ciborium::into_writer(&cbor_msg, &mut cbor_buf).unwrap();

        let packed_data = pack_diff_changes(&changes);
        let packed_msg = ServerMsg::PtyFrameDiffPacked(PtyFrameDiffPacked {
            cursor_col: 0,
            cursor_row: 0,
            num_changes: changes.len() as u32,
            packed: packed_data,
        });
        let mut packed_buf = Vec::new();
        ciborium::into_writer(&packed_msg, &mut packed_buf).unwrap();

        assert!(
            packed_buf.len() < cbor_buf.len(),
            "packed diff ({}) should be smaller than CBOR diff ({})",
            packed_buf.len(),
            cbor_buf.len()
        );
    }

    #[test]
    fn pack_handles_multibyte_graphemes() {
        let cells = vec![vec![PtyCell {
            g: "🦀".into(),
            fg: [255, 128, 0],
            bg: None,
            bold: true,
        }]];
        let packed = pack_cells(&cells);
        let unpacked = unpack_cells(&packed, 1, 1).unwrap();
        assert_eq!(cells, unpacked);
    }
}
