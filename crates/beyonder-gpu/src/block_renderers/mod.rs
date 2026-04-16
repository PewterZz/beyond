pub mod agent_message;
pub mod approval;
pub mod shell_block;

use crate::pipeline::RectInstance;
use beyonder_core::{Block, BlockContent};

/// Trait implemented by each block type's renderer.
pub trait BlockRenderer {
    /// Compute the height this block will occupy at the given width.
    fn measure_height(&self, block: &Block, width: f32, font_size: f32) -> f32;

    /// Emit rectangles for this block at the given position.
    fn render_rects(
        &self,
        block: &Block,
        x: f32,
        y: f32,
        width: f32,
        rects: &mut Vec<RectInstance>,
    );
}

/// Dispatch to the correct renderer based on block content.
/// `font_size` here is the physical font size (already multiplied by scale_factor).
pub fn measure_block_height(block: &Block, width: f32, font_size: f32) -> f32 {
    let cmd_bar_h = font_size * 2.8; // two-row command bar (meta + command)
    let inner_gap = font_size * 0.4; // visible gap between cmd bar and output panel
    let header_h = font_size * 1.8; // header row for non-shell blocks
    let line_h = font_size * 1.45;
    let v_pad = font_size * 0.6; // bottom padding for output panel
    match &block.content {
        BlockContent::ShellCommand { output, .. } => {
            // Count up to the last non-blank row — avoids huge blocks from TUI snapshots
            // that include trailing empty rows from the original terminal grid.
            let last_content = output
                .rows
                .iter()
                .rposition(|row| {
                    row.cells.iter().any(|c| {
                        let fc = c.grapheme.chars().next().unwrap_or('\0');
                        fc != ' ' && fc != '\0'
                    })
                })
                .map(|i| i + 1)
                .unwrap_or(0);
            if last_content == 0 {
                // No output — show only the command bar.
                cmd_bar_h
            } else {
                cmd_bar_h + inner_gap + last_content as f32 * line_h + v_pad
            }
        }
        BlockContent::AgentMessage { content_blocks, .. } => {
            // Estimate wrapped line count using char_w ≈ font_size * 0.6.
            // The renderer uses content_pad * 2 ≈ font_size * 1.0 total horizontal inset.
            let effective_w = (width - font_size * 1.0).max(1.0);
            let chars_per_line = ((effective_w / (font_size * 0.6)).floor() as usize).max(1);

            let visual_lines: f32 = content_blocks
                .iter()
                .map(|cb| match cb {
                    beyonder_core::ContentBlock::Text { text } => text
                        .lines()
                        .map(|line| {
                            let stripped = strip_md_markers(line);
                            let chars = stripped.chars().count().max(1);
                            chars.div_ceil(chars_per_line) as f32
                        })
                        .sum::<f32>()
                        .max(1.0),
                    beyonder_core::ContentBlock::Code { code, .. } => {
                        // +1 for the fence line
                        code.lines()
                            .map(|line| {
                                let chars = line.chars().count().max(1);
                                chars.div_ceil(chars_per_line) as f32
                            })
                            .sum::<f32>()
                            .max(1.0)
                            + 1.0
                    }
                    _ => 1.0,
                })
                .sum::<f32>()
                .max(1.0);
            // Running blocks get an extra row reserved for the tool/spinner indicator.
            let extra = if matches!(block.status, beyonder_core::BlockStatus::Running) {
                line_h * 1.5
            } else {
                0.0
            };
            // No header — just top padding + text + bottom padding.
            v_pad + visual_lines * line_h + v_pad + extra
        }
        BlockContent::ApprovalRequest { .. } => font_size * 10.0,
        BlockContent::ToolCall { output, error, .. } => {
            let text = output.as_deref().or(error.as_deref()).unwrap_or("");
            let lines = if text.is_empty() {
                1.0
            } else {
                text.lines().count() as f32
            };
            header_h + lines * line_h + v_pad
        }
        BlockContent::Text { text } => {
            let lines = text.lines().count().max(1) as f32;
            v_pad + lines * line_h + v_pad
        }
        _ => font_size * 6.0,
    }
}

/// Strip leading markdown markers from a line so char-count reflects visible text width.
fn strip_md_markers(line: &str) -> String {
    let s = line.trim_start_matches('#').trim_start();
    let s = if s.starts_with("- ") || s.starts_with("* ") {
        &s[2..]
    } else {
        s
    };
    // Strip bold/italic markers (**, *, __) — rough, good enough for width estimation.
    s.replace("**", "")
        .replace('*', "")
        .replace("__", "")
        .replace('`', "")
}

/// Emit rectangle draw calls for a block's background and border.
pub fn render_block_background(
    _block: &Block,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    rects: &mut Vec<RectInstance>,
) {
    // Flat terminal background — no tint, no border.
    rects.push(
        RectInstance::filled(x, y, width, height, [0.118, 0.118, 0.180, 1.0]).with_radius(3.0),
    );
}
