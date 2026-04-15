//! Viewport and scroll management for the block stream.

/// Tracks the visible portion of the block stream.
#[derive(Debug, Clone)]
pub struct Viewport {
    pub width: f32,
    pub height: f32,
    pub scroll_offset: f32,
    pub total_content_height: f32,
    /// Physical-pixel offset from the top of the window where the block stream begins.
    /// Lets a tab strip (or other chrome) render above the blocks without overlap.
    pub top_offset: f32,
    /// True when the user is at (or near) the bottom and auto-scroll should track new content.
    /// Set to false when the user explicitly scrolls up; restored when they return to the bottom.
    pub pinned_to_bottom: bool,
}

impl Viewport {
    pub fn new(width: f32, height: f32) -> Self {
        Self {
            width,
            height,
            scroll_offset: 0.0,
            total_content_height: 0.0,
            top_offset: 0.0,
            pinned_to_bottom: true,
        }
    }

    /// Scroll by delta pixels (positive = down).
    pub fn scroll(&mut self, delta: f32) {
        let max_scroll = (self.total_content_height - self.height).max(0.0);
        self.scroll_offset = (self.scroll_offset + delta).clamp(0.0, max_scroll);
        self.pinned_to_bottom = (max_scroll - self.scroll_offset) < 1.0;
    }

    /// Snap to the top.
    pub fn scroll_to_top(&mut self) {
        self.scroll_offset = 0.0;
        self.pinned_to_bottom = false;
    }

    /// Snap to the bottom (show latest content).
    pub fn scroll_to_bottom(&mut self) {
        let max_scroll = (self.total_content_height - self.height).max(0.0);
        self.scroll_offset = max_scroll;
        self.pinned_to_bottom = true;
    }

    pub fn resize(&mut self, width: f32, height: f32) {
        self.width = width;
        self.height = height;
    }

    /// Is a y-coordinate (in content space) visible?
    pub fn is_visible(&self, y: f32, h: f32) -> bool {
        let top = self.scroll_offset;
        let bottom = self.scroll_offset + self.height;
        y + h > top && y < bottom
    }

    /// Convert content-space y to screen-space y.
    pub fn content_to_screen_y(&self, content_y: f32) -> f32 {
        content_y - self.scroll_offset + self.top_offset
    }
}
