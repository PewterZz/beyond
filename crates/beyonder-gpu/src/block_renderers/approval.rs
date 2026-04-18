//! Renderer for ApprovalRequest blocks.
//! These are the security dialogs — rendered with attention-grabbing amber accent.

use super::render_block_background;
use crate::pipeline::RectInstance;
use beyonder_core::Block;

#[allow(clippy::too_many_arguments)]
pub fn render_approval_block(
    block: &Block,
    x: f32,
    y: f32,
    width: f32,
    height: f32,
    scale: f32,
    rects: &mut Vec<RectInstance>,
    button_rects: &mut Vec<([f32; 4], String, bool)>,
    text_labels: &mut Vec<(String, [f32; 4], [u8; 3])>,
) {
    render_block_background(block, x, y, width, height, rects);

    rects.push(
        RectInstance::filled(x, y, 4.0 * scale, height, [0.976, 0.886, 0.686, 1.0])
            .with_radius(scale * 2.0),
    );

    rects.push(
        RectInstance::filled(
            x + scale,
            y + scale,
            width - scale * 2.0,
            26.0 * scale,
            [0.200, 0.170, 0.088, 1.0],
        )
        .with_radius(2.0 * scale),
    );

    let btn_y = y + height - 32.0 * scale;
    let btn_w = 90.0 * scale;
    let btn_h = 24.0 * scale;
    let btn_r = 4.0 * scale;
    let gap = 10.0 * scale;
    let approve_x = x + gap;
    let deny_x = x + gap + btn_w + gap;

    rects.push(
        RectInstance::filled(approve_x, btn_y, btn_w, btn_h, [0.210, 0.500, 0.245, 1.0])
            .with_radius(btn_r),
    );
    rects.push(
        RectInstance::filled(deny_x, btn_y, btn_w, btn_h, [0.530, 0.165, 0.220, 1.0])
            .with_radius(btn_r),
    );
    button_rects.push(([approve_x, btn_y, btn_w, btn_h], block.id.0.clone(), true));
    button_rects.push(([deny_x, btn_y, btn_w, btn_h], block.id.0.clone(), false));
    text_labels.push((
        "Approve".to_string(),
        [approve_x, btn_y, btn_w, btn_h],
        [255, 255, 255],
    ));
    text_labels.push((
        "Deny".to_string(),
        [deny_x, btn_y, btn_w, btn_h],
        [255, 255, 255],
    ));
}
