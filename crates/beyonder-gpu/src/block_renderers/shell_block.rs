//! Renderer for ShellCommand blocks.
//! Two-rect layout: [cmd bar] gap [output panel]
//! The command bar and output panel are visually distinct — different backgrounds,
//! a clear gap between them — so `$ ls` and the directory listing look separate.

use beyonder_core::{Block, BlockContent, BlockKind, BlockStatus};

use crate::pipeline::RectInstance;

#[allow(clippy::too_many_arguments)]
pub fn render_shell_block(
    block: &Block,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    font_size: f32,
    scale: f32,
    rects: &mut Vec<RectInstance>,
) {
    let BlockContent::ShellCommand {
        output, exit_code, ..
    } = &block.content
    else {
        return;
    };

    let cmd_bar_h = font_size * 2.8;
    let inner_gap = font_size * 0.4;

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
    let has_output = last_content > 0;

    // --- Cmd bar (Rect 1) ---
    // Flat terminal background — no tint.
    rects.push(
        RectInstance::filled(x, y, width, cmd_bar_h, [0.118, 0.118, 0.180, 1.0])
            .with_radius(3.0 * scale),
    );

    // Left accent stripe (kind-colored) — replaces the running mauve stripe when done.
    let accent = match block.kind {
        BlockKind::Human => [0.271, 0.278, 0.353, 0.7],
        BlockKind::Agent => [0.537, 0.706, 0.980, 0.85],
        BlockKind::Approval => [0.976, 0.886, 0.686, 0.90],
        BlockKind::System => [0.271, 0.278, 0.353, 0.4],
        BlockKind::Tool => [0.580, 0.886, 0.835, 0.75],
    };
    let stripe_w = 3.0 * scale;
    rects.push(RectInstance::filled(x, y, stripe_w, cmd_bar_h, accent).with_radius(scale));

    // Running indicator — Mauve pulse strip (overrides accent when actively running).
    if block.status == BlockStatus::Running {
        rects.push(
            RectInstance::filled(x, y, stripe_w, cmd_bar_h, [0.796, 0.651, 0.969, 0.95]) // Mauve #cba6f7
                .with_radius(scale),
        );
    }

    // Exit-code right-edge indicator on the cmd bar.
    if let Some(code) = exit_code {
        let indicator = if *code == 0 {
            [0.651, 0.890, 0.631, 0.80] // Green #a6e3a1 — success
        } else {
            [0.953, 0.545, 0.659, 0.85] // Red #f38ba8 — non-zero exit
        };
        rects.push(
            RectInstance::filled(x + width - stripe_w, y, stripe_w, cmd_bar_h, indicator)
                .with_radius(scale),
        );
    }

    // --- Output panel (Rect 2) — only when there's output or command is still running ---
    if has_output || block.status == BlockStatus::Running {
        let out_y = y + cmd_bar_h + inner_gap;
        let out_h = height - cmd_bar_h - inner_gap;
        if out_h > 1.0 {
            // Output panel — same flat terminal background.
            rects.push(
                RectInstance::filled(x, out_y, width, out_h, [0.118, 0.118, 0.180, 1.0])
                    .with_radius(3.0 * scale),
            );
        }
    }
}
