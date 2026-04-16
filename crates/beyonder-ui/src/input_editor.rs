//! Single-line input editor with cursor and submission history.
//! Handles keyboard events and produces text for routing by mode_detector.

use std::path::PathBuf;

/// Maximum number of history entries persisted to disk.
const MAX_HISTORY: usize = 5000;

#[derive(Debug, Clone, Default)]
pub struct InputEditor {
    pub text: String,
    pub cursor: usize,
    /// True when Cmd+A / select-all was pressed. The next insert or delete
    /// replaces the entire contents. Cleared by any cursor movement or edit.
    pub all_selected: bool,
    history: Vec<String>,
    /// None = live input; Some(i) = browsing history at index i (0 = most recent).
    history_idx: Option<usize>,
    /// Stashed live text while browsing history.
    draft: String,
}

impl InputEditor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, ch: char) {
        self.history_idx = None;
        if self.all_selected {
            self.text.clear();
            self.cursor = 0;
            self.all_selected = false;
        }
        self.text.insert(self.cursor, ch);
        self.cursor += ch.len_utf8();
    }

    /// Insert a multi-character string at the cursor (used for paste).
    pub fn insert_text(&mut self, s: &str) {
        self.history_idx = None;
        if self.all_selected {
            self.text.clear();
            self.cursor = 0;
            self.all_selected = false;
        }
        self.text.insert_str(self.cursor, s);
        self.cursor += s.len();
    }

    pub fn delete_backward(&mut self) {
        self.history_idx = None;
        if self.all_selected {
            self.text.clear();
            self.cursor = 0;
            self.all_selected = false;
            return;
        }
        if self.cursor == 0 {
            return;
        }
        let prev = self.prev_char_boundary();
        self.text.drain(prev..self.cursor);
        self.cursor = prev;
    }

    pub fn delete_forward(&mut self) {
        self.history_idx = None;
        if self.all_selected {
            self.text.clear();
            self.cursor = 0;
            self.all_selected = false;
            return;
        }
        if self.cursor >= self.text.len() {
            return;
        }
        let next = self.next_char_boundary();
        self.text.drain(self.cursor..next);
    }

    /// Ctrl+K — delete from cursor to end of line.
    pub fn kill_to_end(&mut self) {
        self.history_idx = None;
        self.all_selected = false;
        self.text.truncate(self.cursor);
    }

    /// Ctrl+U — delete from start of line to cursor.
    pub fn kill_to_start(&mut self) {
        self.history_idx = None;
        self.all_selected = false;
        self.text.drain(..self.cursor);
        self.cursor = 0;
    }

    /// Ctrl+W / Alt+Backspace — delete one word backward (stops at whitespace boundary).
    pub fn delete_word_backward(&mut self) {
        self.history_idx = None;
        self.all_selected = false;
        if self.cursor == 0 {
            return;
        }
        // Skip trailing spaces, then skip the word.
        let mut i = self.cursor;
        while i > 0 && self.text[..i].chars().next_back() == Some(' ') {
            i -= 1;
        }
        while i > 0 && self.text[..i].chars().next_back() != Some(' ') {
            i -= self.text[..i]
                .chars()
                .next_back()
                .map(|c| c.len_utf8())
                .unwrap_or(1);
        }
        self.text.drain(i..self.cursor);
        self.cursor = i;
    }

    /// Alt+Left / Opt+Left — move one word left.
    pub fn word_left(&mut self) {
        self.all_selected = false;
        if self.cursor == 0 {
            return;
        }
        let mut i = self.cursor;
        while i > 0 && self.text[..i].chars().next_back() == Some(' ') {
            i -= self.text[..i]
                .chars()
                .next_back()
                .map(|c| c.len_utf8())
                .unwrap_or(1);
        }
        while i > 0 && self.text[..i].chars().next_back() != Some(' ') {
            i -= self.text[..i]
                .chars()
                .next_back()
                .map(|c| c.len_utf8())
                .unwrap_or(1);
        }
        self.cursor = i;
    }

    /// Alt+Right / Opt+Right — move one word right.
    pub fn word_right(&mut self) {
        self.all_selected = false;
        let len = self.text.len();
        if self.cursor >= len {
            return;
        }
        let mut i = self.cursor;
        while i < len && self.text[i..].chars().next() == Some(' ') {
            i += 1;
        }
        while i < len && self.text[i..].chars().next() != Some(' ') {
            i += self.text[i..]
                .chars()
                .next()
                .map(|c| c.len_utf8())
                .unwrap_or(1);
        }
        self.cursor = i;
    }

    /// Cmd+A — mark all text as selected. Next edit replaces everything.
    pub fn select_all(&mut self) {
        self.all_selected = true;
        self.cursor = self.text.len();
    }

    pub fn move_left(&mut self) {
        self.all_selected = false;
        self.cursor = self.prev_char_boundary();
    }

    pub fn move_right(&mut self) {
        self.all_selected = false;
        self.cursor = self.next_char_boundary();
    }

    pub fn move_home(&mut self) {
        self.all_selected = false;
        self.cursor = 0;
    }

    pub fn move_end(&mut self) {
        self.all_selected = false;
        self.cursor = self.text.len();
    }

    /// Walk backward through history (older). Returns true if the display changed.
    pub fn history_prev(&mut self) -> bool {
        if self.history.is_empty() {
            return false;
        }
        let next_idx = match self.history_idx {
            None => {
                self.draft = self.text.clone();
                0
            }
            Some(i) if i + 1 < self.history.len() => i + 1,
            _ => return false,
        };
        self.history_idx = Some(next_idx);
        let entry = self.history[self.history.len() - 1 - next_idx].clone();
        self.text = entry;
        self.cursor = self.text.len();
        true
    }

    /// Walk forward through history (newer / back to draft). Returns true if the display changed.
    pub fn history_next(&mut self) -> bool {
        match self.history_idx {
            None => false,
            Some(0) => {
                self.history_idx = None;
                self.text = self.draft.clone();
                self.cursor = self.text.len();
                true
            }
            Some(i) => {
                let next_idx = i - 1;
                self.history_idx = Some(next_idx);
                let entry = self.history[self.history.len() - 1 - next_idx].clone();
                self.text = entry;
                self.cursor = self.text.len();
                true
            }
        }
    }

    /// Push a submitted entry into history (dedup consecutive identical entries)
    /// and append to the on-disk history file.
    pub fn push_history(&mut self, text: String) {
        if text.is_empty() {
            return;
        }
        if self.history.last().map(|s| s == &text).unwrap_or(false) {
            return;
        }
        self.history.push(text.clone());
        // Append to disk — fire-and-forget, never block the UI.
        Self::append_to_disk(&text);
    }

    /// Populate in-memory history from the on-disk file.
    pub fn load_history_from_disk(&mut self) {
        let path = Self::history_path();
        if let Ok(contents) = std::fs::read_to_string(&path) {
            self.history = contents
                .lines()
                .filter(|l| !l.is_empty())
                .map(String::from)
                .collect();
            // Trim to last MAX_HISTORY entries if the file has grown.
            if self.history.len() > MAX_HISTORY {
                let excess = self.history.len() - MAX_HISTORY;
                self.history.drain(..excess);
            }
        }
    }

    fn history_path() -> PathBuf {
        beyonder_config::beyonder_dir().join("history")
    }

    fn append_to_disk(line: &str) {
        use std::io::Write;
        let path = Self::history_path();
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
        {
            let _ = writeln!(f, "{}", line);
        }
    }

    /// Replace the editor content and move cursor to end.
    pub fn set_text(&mut self, text: String) {
        self.history_idx = None;
        self.cursor = text.len();
        self.text = text;
    }

    /// Take the current text and clear the editor (submit).
    pub fn submit(&mut self) -> String {
        self.history_idx = None;
        self.all_selected = false;
        self.draft.clear();
        let text = std::mem::take(&mut self.text);
        self.cursor = 0;
        text
    }

    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    fn prev_char_boundary(&self) -> usize {
        if self.cursor == 0 {
            return 0;
        }
        let mut i = self.cursor - 1;
        while !self.text.is_char_boundary(i) {
            i -= 1;
        }
        i
    }

    fn next_char_boundary(&self) -> usize {
        if self.cursor >= self.text.len() {
            return self.text.len();
        }
        let mut i = self.cursor + 1;
        while i < self.text.len() && !self.text.is_char_boundary(i) {
            i += 1;
        }
        i
    }

    /// Return the suffix of the best matching history entry, if any.
    /// Searches from most recent to oldest for an entry that starts with
    /// the current input text (case-sensitive prefix match).
    pub fn ghost_suggestion(&self) -> Option<&str> {
        let prefix = &self.text;
        if prefix.is_empty() || self.history_idx.is_some() {
            return None;
        }
        // Walk history backwards (most recent first).
        for entry in self.history.iter().rev() {
            if entry.len() > prefix.len() && entry.starts_with(prefix.as_str()) {
                return Some(&entry[prefix.len()..]);
            }
        }
        None
    }

    /// Accept the ghost suggestion: replace text with the full history entry.
    /// Returns true if a suggestion was accepted.
    pub fn accept_suggestion(&mut self) -> bool {
        if let Some(suffix) = self.ghost_suggestion().map(String::from) {
            self.text.push_str(&suffix);
            self.cursor = self.text.len();
            true
        } else {
            false
        }
    }
}
