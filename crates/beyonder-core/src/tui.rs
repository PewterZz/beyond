//! TUI cell type shared between terminal emulation and GPU rendering.

use serde::{Deserialize, Serialize};

/// Underline rendering style for a cell.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnderlineStyle {
    #[default]
    None,
    Single,
    Double,
    Curly,
    Dotted,
    Dashed,
}

/// A single rendered cell in a TUI screen grid.
#[derive(Debug, Clone)]
pub struct TuiCell {
    /// Full grapheme cluster (base + combining/ZWJ sequence). A single terminal
    /// cell may contain multiple codepoints, e.g. `👨‍👩‍👧` (ZWJ family),
    /// `👋🏽` (skin-tone modifier), `1️⃣` (keycap), `🇯🇵` (regional-indicator flag).
    pub grapheme: String,
    /// Foreground RGB color (linear, 0.0–1.0).
    pub fg: [f32; 3],
    /// Background RGB color. `None` means default background (transparent/Base).
    pub bg: Option<[f32; 3]>,
    pub bold: bool,
    pub italic: bool,
    pub underline: UnderlineStyle,
    pub strikethrough: bool,
    /// OSC 8 hyperlink URI (shared via `Arc` because runs of cells share the same link).
    pub link: Option<std::sync::Arc<String>>,
}

impl TuiCell {
    /// First codepoint of the grapheme, or `\0` if empty. Useful for
    /// single-codepoint classification (whitespace/null/box-drawing).
    pub fn first_char(&self) -> char {
        self.grapheme.chars().next().unwrap_or('\0')
    }

    /// True if the cell is a null spacer (wide-char follow-up) or blank.
    pub fn is_null(&self) -> bool {
        self.grapheme.is_empty() || self.grapheme == "\0"
    }
}
