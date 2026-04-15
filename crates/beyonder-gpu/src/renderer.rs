//! Main GPU renderer — wgpu pipeline + glyphon text rendering.

use anyhow::{Context, Result};
use tracing::debug;
use beyonder_core::{AgentId, Block, BlockContent, BlockKind, BlockStatus, TuiCell};
use glyphon::{
    Attrs, Buffer as GlyphBuffer, Cache, Color as GlyphColor, ColorMode, Family, FontSystem,
    Metrics, Resolution, Shaping, SwashCache, TextArea, TextAtlas, TextBounds,
    TextRenderer, Viewport as GlyphViewport,
};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;
use winit::window::Window;

use crate::block_renderers::{
    approval::render_approval_block, agent_message::render_agent_message,
    measure_block_height, render_block_background, shell_block::render_shell_block,
};
use crate::pipeline::{RectInstance, RectPipeline};
use crate::viewport::Viewport;

/// Minimum height of the input bar (one text line + pills + mode pill + padding).
const INPUT_BAR_HEIGHT: f32 = 120.0;
/// Maximum number of visible lines in the input text area before it scrolls.
const MAX_INPUT_LINES: usize = 4;
/// Logical height of the tab strip when visible (>= 2 tabs).
const TAB_BAR_HEIGHT: f32 = 28.0;
/// Horizontal padding around the block stream.
const PADDING: f32 = 4.0;
/// Vertical gap between blocks.
const GAP: f32 = 2.0;

pub struct Renderer {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub surface_config: wgpu::SurfaceConfiguration,
    pub rect_pipeline: RectPipeline,

    // Text rendering
    pub font_system: FontSystem,
    pub swash_cache: SwashCache,
    pub glyph_cache: Cache,
    pub glyph_viewport: GlyphViewport,
    pub text_atlas: TextAtlas,
    pub text_renderer: TextRenderer,

    pub viewport: Viewport,
    /// Logical font size in points/px. Multiply by scale_factor for physical pixels.
    pub font_size: f32,
    /// HiDPI scale factor — 2.0 on Retina, 1.0 on standard displays.
    pub scale_factor: f32,
    /// Measured cell dimensions (cell_w, cell_h) derived from actual font metrics.
    /// cell_h = floor(max_ascent + max_descent) — matches swash raster height exactly.
    /// Cached so terminal_cell_size() doesn't reshape every frame.
    measured_cell_size: (f32, f32),
    /// Exact (non-floored) line height for GlyphBuffer Metrics.
    /// = max_ascent + max_descent as a float, so centering_offset is 0 and glyphs
    /// sit at the top of each cell without a 1px downward shift from rounding.
    measured_metrics_line_h: f32,
    pub blocks: Vec<Block>,

    // Input bar state (synced from App before each render)
    pub input_text: String,
    pub input_cursor: usize,
    pub input_all_selected: bool,
    pub input_mode_prefix: String,

    /// Cached dynamic bar height in physical pixels — updated once per frame.
    computed_bar_h: f32,
    /// Pixel scroll offset applied to the input text (keeps cursor visible when > MAX_INPUT_LINES).
    input_scroll_px: f32,

    /// Selected block index — highlighted on screen, text copyable via Cmd+C.
    pub selected_block: Option<usize>,
    /// For ShellCommand blocks: true = output panel selected, false = cmd bar selected.
    pub selected_sub_output: bool,
    /// True while a command is running — input bar shows a "running" indicator.
    pub input_running: bool,

    pub tui_active: bool,
    pub tui_cells: Vec<Vec<TuiCell>>,
    pub tui_cursor: (usize, usize),
    /// Cursor shape requested by the TUI app (0=block, 1=beam, 2=underline).
    pub tui_cursor_shape: u8,
    /// Index of the currently running ShellCommand block (if any). Its content
    /// area renders live TermGrid cells instead of stored output.
    pub running_block_idx: Option<usize>,

    // Cursor blink state
    cursor_blink_on: bool,
    cursor_last_toggle: Instant,

    // Spinner for streaming agent blocks
    pub spinner_frame: u8,
    spinner_last_tick: Instant,

    // Context pills (synced from App before each render)
    /// Labels for the 3 context pills: [conda, node, dir].
    pub context_pills: Vec<String>,
    /// Currently open dropdown: (pill_idx, items, hovered_idx).
    pub open_dropdown: Option<(usize, Vec<String>, Option<usize>)>,
    /// Bounding rects [x, y, w, h] for each pill (written during layout).
    pub pill_rects: Vec<[f32; 4]>,
    /// Bounding rects [x, y, w, h] per dropdown item (written during layout).
    pub dropdown_item_rects: Vec<[f32; 4]>,

    // Command palette — shown when input starts with /
    /// Filtered command list: (usage, description). Set by App each frame.
    pub command_palette: Option<Vec<(String, String)>>,
    pub cmd_palette_hovered: Option<usize>,
    /// Hit-test rects for each palette row [x, y, w, h].
    pub cmd_palette_rects: Vec<[f32; 4]>,

    // Mode switcher — bottom-left of the input bar.
    pub mode_label: String,
    pub mode_pill_rect: [f32; 4],

    /// Active agent model name — shown as a pill in the top-right of the input bar.
    pub agent_model: String,

    /// GlyphBuffer cache: block_id → (content_len, buf_w_bits, font_bits, viewport_h_bits, buffer).
    /// Re-shaping is skipped when content and layout params are unchanged.
    glyph_buf_cache: HashMap<beyonder_core::BlockId, (u64, u32, u32, u32, GlyphBuffer)>,

    /// Currently executing tool per agent: agent_id → tool_name.
    /// Set by App when ToolCallRequested arrives; cleared on TextDelta or TurnComplete.
    pub agent_running_tool: HashMap<beyonder_core::AgentId, String>,

    /// Blocks whose default-collapsed state has been overridden by the user.
    /// A block_id in this set is always shown expanded regardless of collapsed_default.
    pub user_expanded: HashSet<beyonder_core::BlockId>,

    // Tab strip (synced from App before each render).
    /// Labels for each tab. Strip is shown only when `tab_labels.len() >= 2`.
    pub tab_labels: Vec<String>,
    /// Index of the currently-active tab.
    pub active_tab: usize,
    /// Hit rects [x, y, w, h] per tab, written during tab-strip layout.
    pub tab_rects: Vec<[f32; 4]>,
}

/// Accumulates GlyphBuffer entries for a frame. Parallel `keys` vec records
/// optional cache metadata so shaped buffers can be returned to the cache
/// after `TextRenderer::prepare()` consumes the TextArea borrows.
struct TextBufList {
    entries: Vec<(GlyphBuffer, f32, f32, f32, f32, GlyphColor)>,
    /// (block_id, content_len, buf_w_bits, pf_bits, viewport_h_bits)
    keys: Vec<Option<(beyonder_core::BlockId, u64, u32, u32, u32)>>,
    /// Per-entry explicit clip rect override (clip_top, clip_bottom). When Some,
    /// overrides the default derivation from (y, y+h). Used for scrolled buffers
    /// where TextArea.top is shifted but the visible clip window differs.
    clip_overrides: Vec<Option<(i32, i32)>>,
}

impl TextBufList {
    fn new() -> Self {
        Self { entries: vec![], keys: vec![], clip_overrides: vec![] }
    }

    /// Push a non-cached entry (tuple matches Vec::push call sites unchanged).
    fn push(&mut self, entry: (GlyphBuffer, f32, f32, f32, f32, GlyphColor)) {
        self.entries.push(entry);
        self.keys.push(None);
        self.clip_overrides.push(None);
    }

    /// Push with an explicit clip rect (clip_top, clip_bottom) in physical pixels.
    fn push_clipped(&mut self, entry: (GlyphBuffer, f32, f32, f32, f32, GlyphColor), clip: (i32, i32)) {
        self.entries.push(entry);
        self.keys.push(None);
        self.clip_overrides.push(Some(clip));
    }

    /// Push a cacheable entry alongside its invalidation key.
    fn push_cached(
        &mut self,
        entry: (GlyphBuffer, f32, f32, f32, f32, GlyphColor),
        key: (beyonder_core::BlockId, u64, u32, u32, u32),
    ) {
        self.entries.push(entry);
        self.keys.push(Some(key));
        self.clip_overrides.push(None);
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
}

impl Renderer {
    pub async fn new(window: Arc<Window>) -> Result<Self> {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance
            .create_surface(Arc::clone(&window))
            .context("Failed to create surface")?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .context("No suitable GPU adapter")?;

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("beyonder"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                },
                None,
            )
            .await
            .context("Failed to get GPU device")?;

        let caps = surface.get_capabilities(&adapter);
        // Prefer non-sRGB so we can use hex values (sRGB) directly without
        // linear conversion. With a sRGB surface the GPU would gamma-correct
        // our values making all colours appear wrong.
        let surface_format = caps
            .formats
            .iter()
            .copied()
            .find(|f| !f.is_srgb())
            .unwrap_or(wgpu::TextureFormat::Bgra8Unorm);

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        let rect_pipeline = RectPipeline::new(&device, surface_format);
        rect_pipeline.update_screen_size(&queue, size.width as f32, size.height as f32);

        // Glyphon 0.8 setup
        let mut font_system = FontSystem::new();
        let swash_cache = SwashCache::new();
        let glyph_cache = Cache::new(&device);
        let mut glyph_viewport = GlyphViewport::new(&device, &glyph_cache);
        glyph_viewport.update(
            &queue,
            Resolution { width: size.width.max(1), height: size.height.max(1) },
        );
        // Use Web color mode: our surface is Bgra8Unorm (non-sRGB) and all colour
        // values (palette, rects, glyph attrs) are already in sRGB space.
        // Accurate mode would gamma-decode text colours to linear before upload,
        // making them darker than rect colours which go through unmodified.
        let mut text_atlas =
            TextAtlas::with_color_mode(&device, &queue, &glyph_cache, surface_format, ColorMode::Web);
        let text_renderer = TextRenderer::new(
            &mut text_atlas,
            &device,
            wgpu::MultisampleState::default(),
            None,
        );

        let font_size = 16.0; // logical pixels
        let scale_factor = window.scale_factor() as f32;
        let input_bar_h = INPUT_BAR_HEIGHT * scale_factor;
        // Viewport covers only the block-stream area; input bar at bottom is excluded.
        let viewport = Viewport::new(size.width as f32, size.height as f32 - input_bar_h);

        let (cell_w, cell_h, metrics_line_h) = Self::measure_cell_size_static(&mut font_system, font_size, scale_factor);
        let measured_cell_size = (cell_w, cell_h);
        let measured_metrics_line_h = metrics_line_h;

        Ok(Self {
            device,
            queue,
            surface,
            surface_config,
            rect_pipeline,
            font_system,
            swash_cache,
            glyph_cache,
            glyph_viewport,
            text_atlas,
            text_renderer,
            viewport,
            font_size,
            scale_factor,
            measured_cell_size,
            measured_metrics_line_h,
            blocks: vec![],
            input_text: String::new(),
            input_cursor: 0,
            input_all_selected: false,
            input_mode_prefix: "> ".to_string(),
            computed_bar_h: INPUT_BAR_HEIGHT * scale_factor,
            input_scroll_px: 0.0,

            selected_block: None,
            selected_sub_output: false,
            input_running: false,
            tui_active: false,
            tui_cells: vec![],
            tui_cursor: (0, 0),
            tui_cursor_shape: 0,
            running_block_idx: None,
            cursor_blink_on: true,
            cursor_last_toggle: Instant::now(),
            spinner_frame: 0,
            spinner_last_tick: Instant::now(),
            context_pills: vec![],
            open_dropdown: None,
            pill_rects: vec![],
            dropdown_item_rects: vec![],
            command_palette: None,
            cmd_palette_hovered: None,
            cmd_palette_rects: vec![],
            mode_label: "auto".to_string(),
            mode_pill_rect: [0.0; 4],
            agent_model: String::new(),
            glyph_buf_cache: HashMap::new(),
            agent_running_tool: HashMap::new(),
            user_expanded: HashSet::new(),
            tab_labels: vec![],
            active_tab: 0,
            tab_rects: vec![],
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.surface_config.width = width;
        self.surface_config.height = height;
        self.surface.configure(&self.device, &self.surface_config);
        let tab_h = self.tab_bar_height_phys();
        self.viewport.resize(width as f32, height as f32 - self.computed_bar_h - tab_h);
        self.viewport.top_offset = tab_h;
        self.rect_pipeline.update_screen_size(&self.queue, width as f32, height as f32);
        self.glyph_viewport.update(
            &self.queue,
            Resolution { width, height },
        );
    }

    /// Physical-pixel height of the tab strip. Zero when fewer than 2 tabs or when a TUI app is active.
    pub fn tab_bar_height_phys(&self) -> f32 {
        if self.tab_labels.len() >= 2 && !self.tui_active {
            TAB_BAR_HEIGHT * self.scale_factor
        } else {
            0.0
        }
    }

    /// Returns the tab index that was clicked at (px, py), if any.
    pub fn tab_hit(&self, px: f32, py: f32) -> Option<usize> {
        for (i, rect) in self.tab_rects.iter().enumerate() {
            let [x, y, w, h] = *rect;
            if px >= x && px < x + w && py >= y && py < y + h {
                return Some(i);
            }
        }
        None
    }

    pub fn set_scale_factor(&mut self, scale_factor: f64) {
        self.scale_factor = scale_factor as f32;
        let (cell_w, cell_h, metrics_line_h) = Self::measure_cell_size_static(&mut self.font_system, self.font_size, self.scale_factor);
        self.measured_cell_size = (cell_w, cell_h);
        self.measured_metrics_line_h = metrics_line_h;
    }

    /// Measure actual cell dimensions from font metrics using cosmic-text layout.
    /// Uses `max_ascent + max_descent` from a shaped reference string so that
    /// box-drawing and block-element characters (which are designed to fill the
    /// exact ascent+descent range) tile without gaps or overlaps between rows.
    /// Returns (cell_w, cell_h_display, metrics_line_h).
    /// cell_h_display = floor(max_ascent + max_descent) — matches swash's integer pixel height.
    /// metrics_line_h = exact float, so GlyphBuffer centering_offset = 0 (glyphs sit at cell top).
    fn measure_cell_size_static(font_system: &mut FontSystem, font_size: f32, scale_factor: f32) -> (f32, f32, f32) {
        let phys = font_size * scale_factor;
        let metrics = Metrics::new(phys, phys * 2.0);
        let mut buf = GlyphBuffer::new(font_system, metrics);
        buf.set_size(font_system, None, None);
        buf.set_text(
            font_system,
            "Mg",
            Attrs::new().family(Family::Name("JetBrainsMono Nerd Font")),
            Shaping::Basic,
        );
        buf.shape_until_scroll(font_system, false);

        let mut cell_w = (phys * 0.6).round();
        let mut metrics_line_h = phys * 1.2; // fallback
        let mut cell_h = metrics_line_h.floor();

        if let Some(buf_line) = buf.lines.first() {
            if let Some(layout_lines) = buf_line.layout_opt() {
                if let Some(ll) = layout_lines.first() {
                    metrics_line_h = ll.max_ascent + ll.max_descent;
                    // floor() matches the integer pixel height swash rasterizes.
                    // Using ceil() would make rows 1px taller than glyphs, causing
                    // 1px gaps between adjacent box-drawing / block characters.
                    cell_h = metrics_line_h.floor();
                }
            }
        }
        for run in buf.layout_runs() {
            if let Some(last) = run.glyphs.last() {
                let total_w = last.x + last.w;
                let n = run.glyphs.len() as f32;
                cell_w = (total_w / n).round();
            }
            break;
        }

        (cell_w, cell_h, metrics_line_h)
    }

    /// Return the block and sub-region hit by a click at screen_y.
    /// Returns (block_index, is_output_panel).
    /// For ShellCommand blocks the cmd bar and output panel are separate hit regions.
    pub fn block_hit_at(&self, screen_y: f32) -> Option<(usize, bool)> {
        let sc = self.scale_factor;
        let padding = PADDING * sc;
        let gap = GAP * sc;
        let phys_font = self.font_size * sc;
        let content_w = self.viewport.width - padding * 2.0;
        let inner_gap = phys_font * 0.4;
        let mut y = padding;
        for (i, block) in self.blocks.iter().enumerate() {
            let h = self.block_height(i, block, content_w, phys_font);
            let sy = self.viewport.content_to_screen_y(y);
            if screen_y >= sy && screen_y < sy + h {
                let shell_cmd_bar_h = phys_font * 2.8;
                let is_output = matches!(block.content, BlockContent::ShellCommand { .. })
                    && screen_y >= sy + shell_cmd_bar_h + inner_gap;
                return Some((i, is_output));
            }
            y += h + gap;
        }
        None
    }

    /// Plain block index hit-test (ignores sub-regions).
    pub fn block_index_at(&self, screen_y: f32) -> Option<usize> {
        self.block_hit_at(screen_y).map(|(i, _)| i)
    }

    /// Returns true if the mode switcher pill at the bottom-left was clicked.
    pub fn mode_pill_hit(&self, px: f32, py: f32) -> bool {
        let [x, y, w, h] = self.mode_pill_rect;
        w > 0.0 && px >= x && px < x + w && py >= y && py < y + h
    }

    /// Returns the index of the pill that was clicked (0=conda, 1=node, 2=dir).
    pub fn pill_hit(&self, px: f32, py: f32) -> Option<usize> {
        for (i, rect) in self.pill_rects.iter().enumerate() {
            let [x, y, w, h] = *rect;
            if px >= x && px < x + w && py >= y && py < y + h {
                return Some(i);
            }
        }
        None
    }

    /// Returns the index of the command palette row that was clicked.
    pub fn cmd_palette_hit(&self, px: f32, py: f32) -> Option<usize> {
        for (i, rect) in self.cmd_palette_rects.iter().enumerate() {
            let [x, y, w, h] = *rect;
            if px >= x && px < x + w && py >= y && py < y + h {
                return Some(i);
            }
        }
        None
    }

    /// Returns the index of the dropdown item that was clicked.
    pub fn dropdown_hit(&self, px: f32, py: f32) -> Option<usize> {
        for (i, rect) in self.dropdown_item_rects.iter().enumerate() {
            let [x, y, w, h] = *rect;
            if px >= x && px < x + w && py >= y && py < y + h {
                return Some(i);
            }
        }
        None
    }

    /// Returns the index of the dropdown item hovered at the given position.
    pub fn dropdown_hover_at(&self, px: f32, py: f32) -> Option<usize> {
        self.dropdown_hit(px, py)
    }

    pub fn scroll(&mut self, delta: f32) {
        self.viewport.scroll(delta);
    }

    pub fn scroll_to_bottom(&mut self) {
        self.viewport.scroll_to_bottom();
    }

    /// Physical size of one terminal cell based on actual font metrics.
    /// Derived from `max_ascent + max_descent` of shaped text so that box-drawing
    /// and block-element characters tile without gaps between rows.
    pub fn terminal_cell_size(&self) -> (f32, f32) {
        self.measured_cell_size
    }

    /// Physical height of the input bar in pixels.
    pub fn bar_height_phys(&self) -> f32 {
        self.computed_bar_h
    }

    /// Physical (width, height) of the surface in pixels.
    pub fn surface_size(&self) -> (f32, f32) {
        (self.surface_config.width as f32, self.surface_config.height as f32)
    }

    /// Scroll the input text viewport by `delta` physical pixels (positive = down / toward newer text).
    /// Clamped between 0 and max scroll. Call this when the user scrolls over the input bar.
    pub fn scroll_input(&mut self, delta: f32) {
        let sc = self.scale_factor;
        let phys_font = self.font_size * sc;
        let win_w = self.surface_config.width as f32;
        let h_pad = 14.0 * sc;
        let text_w = (win_w - h_pad * 2.0).max(1.0);
        let line_h = phys_font * 1.4;
        let (total_lines, _) = self.measure_input_lines(text_w, phys_font);
        let visible_lines = total_lines.min(MAX_INPUT_LINES);
        let max_scroll = ((total_lines as f32 - visible_lines as f32) * line_h).max(0.0);
        self.input_scroll_px = (self.input_scroll_px + delta).clamp(0.0, max_scroll);
    }

    /// Snap input scroll back so the cursor is visible (call after any cursor-moving keystroke).
    pub fn snap_input_scroll_to_cursor(&mut self) {
        let sc = self.scale_factor;
        let phys_font = self.font_size * sc;
        let win_w = self.surface_config.width as f32;
        let h_pad = 14.0 * sc;
        let text_w = (win_w - h_pad * 2.0).max(1.0);
        let line_h = phys_font * 1.4;
        let (total_lines, cursor_line) = self.measure_input_lines(text_w, phys_font);
        let visible_lines = total_lines.min(MAX_INPUT_LINES);
        let viewport_h = visible_lines as f32 * line_h;
        let cursor_top = cursor_line as f32 * line_h;
        let cursor_bot = cursor_top + line_h;
        if cursor_top < self.input_scroll_px {
            self.input_scroll_px = cursor_top;
        }
        if cursor_bot > self.input_scroll_px + viewport_h {
            self.input_scroll_px = cursor_bot - viewport_h;
        }
        let max_scroll = ((total_lines as f32 - visible_lines as f32) * line_h).max(0.0);
        self.input_scroll_px = self.input_scroll_px.clamp(0.0, max_scroll);
    }

    /// Measure how many visual lines the current input text produces when wrapped to text_w,
    /// and which visual line the cursor is on. Returns (total_visual_lines, cursor_visual_line).
    fn measure_input_lines(&mut self, text_w: f32, phys_font: f32) -> (usize, usize) {
        if self.input_text.is_empty() || self.input_running {
            return (1, 0);
        }
        let cursor = self.input_cursor.min(self.input_text.len());
        let before = &self.input_text[..cursor];
        let after = &self.input_text[cursor..];
        let text = if self.input_all_selected {
            format!("{}█{}", self.input_mode_prefix, self.input_text)
        } else {
            format!("{}{}▌{}", self.input_mode_prefix, before, after)
        };

        // Byte position of the caret (▌ is 3 UTF-8 bytes; █ is 3 bytes too)
        let prefix_len = self.input_mode_prefix.len();
        let caret_byte_end = if self.input_all_selected {
            // select-all: cursor conceptually at the end
            text.len()
        } else {
            prefix_len + cursor + "▌".len()
        };

        let col = GlyphColor::rgb(205, 214, 244);
        let buf = self.make_buffer(&text, text_w, phys_font, col);

        let mut total_lines = 0usize;
        let mut cursor_line = 0usize;

        for run in buf.layout_runs() {
            let run_end = run.glyphs.last().map(|g| g.end).unwrap_or(0);
            // If this run ends before the caret, the cursor is on a later line.
            if run_end < caret_byte_end {
                cursor_line = total_lines + 1;
            }
            total_lines += 1;
        }
        // Clamp cursor_line in case it went one past due to the loop logic
        let cursor_line = cursor_line.min(total_lines.saturating_sub(1));

        (total_lines.max(1), cursor_line)
    }

    /// Recompute `computed_bar_h` and `input_scroll_px` based on current input state.
    /// Call once per frame before `append_bar_rects` and `build_bar_text_buffers`.
    fn compute_bar_state(&mut self) {
        let sc = self.scale_factor;
        let phys_font = self.font_size * sc;
        let win_w = self.surface_config.width as f32;
        let h_pad = 14.0 * sc;
        let text_w = (win_w - h_pad * 2.0).max(1.0);
        let line_h = phys_font * 1.4;

        let (total_lines, cursor_line) = self.measure_input_lines(text_w, phys_font);
        let visible_lines = total_lines.min(MAX_INPUT_LINES);

        // Bar height = base 1-line height + extra lines.
        // Base (120 logical) fits exactly 1 line with centering margins.
        let line_h_logical = self.font_size * 1.4;
        let extra_lines = (visible_lines as f32 - 1.0).max(0.0);
        let bar_h_logical = INPUT_BAR_HEIGHT + extra_lines * line_h_logical;
        self.computed_bar_h = bar_h_logical * sc;

        // Clamp scroll to valid range (content may have shrunk).
        let max_scroll = ((total_lines as f32 - visible_lines as f32) * line_h).max(0.0);
        self.input_scroll_px = self.input_scroll_px.clamp(0.0, max_scroll);

        if total_lines > 1 {
            tracing::info!(
                total_lines,
                cursor_line,
                visible_lines,
                input_scroll_px = self.input_scroll_px,
                "input bar state"
            );
        }
    }

    /// Terminal grid dimensions for TUI fullscreen mode (bar hidden — full window).
    pub fn tui_grid_size(&self) -> (u16, u16) {
        let (cell_w, cell_h) = self.terminal_cell_size();
        let full_w = self.surface_config.width as f32;
        let full_h = self.surface_config.height as f32;
        let cols = (full_w / cell_w).floor().max(40.0) as u16;
        let rows = (full_h / cell_h).floor().max(10.0) as u16;
        (cols, rows)
    }

    /// Terminal grid dimensions (cols × rows) that fit in the usable area above the input bar.
    /// This is the source of truth for PTY sizing on spawn and resize.
    /// NOTE: viewport.height is already set to (surface_height - bar_height_phys) by resize(),
    /// so we use it directly — do NOT subtract bar_height again here.
    pub fn terminal_grid_size(&self) -> (u16, u16) {
        let (cell_w, cell_h) = self.terminal_cell_size();
        let sc = self.scale_factor;
        let content_w = (self.viewport.width - PADDING * sc * 2.0).max(cell_w);
        let content_h = self.viewport.height.max(cell_h);
        let cols = (content_w / cell_w).floor().max(40.0) as u16;
        let rows = (content_h / cell_h).floor().max(10.0) as u16;
        (cols, rows)
    }

    pub fn render(&mut self) -> Result<()> {
        // Advance cursor blink — toggle every 530 ms.
        let now = Instant::now();
        if now.duration_since(self.cursor_last_toggle).as_millis() >= 530 {
            self.cursor_blink_on = !self.cursor_blink_on;
            self.cursor_last_toggle = now;
        }
        // Advance spinner — advance every 80 ms through 10 braille frames.
        if now.duration_since(self.spinner_last_tick).as_millis() >= 80 {
            self.spinner_frame = (self.spinner_frame + 1) % 10;
            self.spinner_last_tick = now;
        }

        // Recompute dynamic bar height and scroll offset.
        let old_bar_h = self.computed_bar_h;
        let old_tab_h = self.viewport.top_offset;
        self.compute_bar_state();
        let tab_h = self.tab_bar_height_phys();
        if (self.computed_bar_h - old_bar_h).abs() > 0.5 || (tab_h - old_tab_h).abs() > 0.5 {
            let w = self.surface_config.width as f32;
            let h = self.surface_config.height as f32;
            self.viewport.resize(w, h - self.computed_bar_h - tab_h);
            self.viewport.top_offset = tab_h;
        }

        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                self.surface.configure(&self.device, &self.surface_config);
                return Ok(());
            }
            Err(e) => return Err(e.into()),
        };

        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("beyonder_frame"),
                });

        // --- Rect layout ---
        let mut rects = if !self.tui_active {
            let (mut r, total_h) = self.layout_blocks();
            self.viewport.total_content_height = total_h;
            if self.running_block_idx.is_some() && self.viewport.pinned_to_bottom {
                self.viewport.scroll_to_bottom();
            }
            r
        } else {
            let mut r = vec![];
            self.layout_tui(&mut r);
            r
        };
        // Only hide the input bar for full-screen TUI apps (nvim, htop) that take over
        // the alt-screen — not for every running shell command. Regular commands keep
        // the bar visible; input_running shows a "running…" indicator instead.
        let bar_hidden = self.tui_active;
        if !bar_hidden {
            self.append_bar_rects(&mut rects);
        }
        self.append_tab_bar_rects(&mut rects);
        self.rect_pipeline.upload_instances(&self.queue, &rects);

        // --- Text layout ---
        // Buffers must outlive the TextArea slices, so we build them here.
        // Each entry: (buffer, x, y, w, h, color)
        debug!(blocks = self.blocks.len(), "render: building text buffers");
        let (mut buf_list, block_entry_count) = if self.tui_active {
            let texts = self.build_tui_text_buffers();
            let count = texts.len();
            (texts, count)
        } else {
            self.build_text_buffers()
        };
        if !bar_hidden {
            let bar_texts = self.build_bar_text_buffers();
            buf_list.entries.extend(bar_texts.entries);
            buf_list.keys.extend(bar_texts.keys);
            buf_list.clip_overrides.extend(bar_texts.clip_overrides);
        }
        self.build_tab_bar_text_buffers(&mut buf_list);
        debug!(entries = buf_list.entries.len(), "render: text buffers built");
        let win_h = self.surface_config.height as f32;
        // When the bar is hidden, text fills the full window.
        // When visible, text must not render over the input bar.
        let text_clip_bottom = if bar_hidden {
            win_h
        } else {
            win_h - self.computed_bar_h
        };
        // Block-stream text must not bleed above the tab strip.
        let block_clip_top_min = self.tab_bar_height_phys() as i32;
        let text_areas: Vec<TextArea> = buf_list.entries
            .iter()
            .enumerate()
            .map(|(i, (buf, x, y, w, h, color))| {
                // Block stream entries are clipped at the bar boundary.
                // Bar text uses (y, y+h) unless a clip override was provided
                // (e.g. scrolled input text where TextArea.top is shifted).
                let (clip_top, clip_bottom) = if let Some((ct, cb)) = buf_list.clip_overrides[i] {
                    (ct, cb)
                } else if i < block_entry_count {
                    ((*y as i32).max(block_clip_top_min), ((*y + *h) as i32).min(text_clip_bottom as i32))
                } else {
                    ((*y as i32).max(0), (*y + *h) as i32)
                };
                TextArea {
                    buffer: buf,
                    left: *x,
                    top: *y,
                    scale: 1.0,
                    bounds: TextBounds {
                        left: (*x as i32).max(0),
                        top: clip_top,
                        right: (*x + *w) as i32,
                        bottom: clip_bottom,
                    },
                    default_color: *color,
                    custom_glyphs: &[],
                }
            })
            .collect();

        debug!("render: calling text_renderer.prepare");
        self.text_renderer
            .prepare(
                &self.device,
                &self.queue,
                &mut self.font_system,
                &mut self.text_atlas,
                &self.glyph_viewport,
                text_areas,
                &mut self.swash_cache,
            )
            .ok();
        debug!("render: text_renderer.prepare done");

        // Re-insert shaped buffers that have cache keys back into the cache.
        // text_areas was consumed by prepare() so buf_list.entries is free to move.
        for ((buf, ..), key) in buf_list.entries.into_iter().zip(buf_list.keys.into_iter()) {
            if let Some((id, len, bw, pf, vh)) = key {
                self.glyph_buf_cache.insert(id, (len, bw, pf, vh, buf));
            }
        }

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("beyonder_main"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            // Catppuccin Mocha — Base #1e1e2e
                            r: 0.118,
                            g: 0.118,
                            b: 0.180,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            self.rect_pipeline.draw(&mut pass, rects.len() as u32);
            self.text_renderer
                .render(&self.text_atlas, &self.glyph_viewport, &mut pass)
                .ok();
        }

        debug!("render: submitting GPU commands");
        self.queue.submit([encoder.finish()]);
        output.present();
        debug!("render: frame presented");

        // Free atlas glyphs that are no longer needed this frame.
        self.text_atlas.trim();
        Ok(())
    }

    // -------------------------------------------------------------------------
    // Layout helpers
    // -------------------------------------------------------------------------

    /// Height for the running block — sized to fit up to the last non-blank
    /// TermGrid row so simple commands don't get a massive 30-row block.
    fn live_block_height(&self, phys_font: f32) -> f32 {
        let last_content = self.tui_cells
            .iter()
            .rposition(|row| row.iter().any(|c| c.ch != ' ' && c.ch != '\0'))
            .map(|i| i + 1)
            .unwrap_or(1);
        let (_, cell_h) = self.terminal_cell_size();
        let cmd_bar_h = phys_font * 2.8;
        let inner_gap = phys_font * 0.4;
        cmd_bar_h + inner_gap + last_content as f32 * cell_h + cell_h * 0.5
    }

    /// Height to use for block `idx` — overrides stored-output measurement
    /// Returns true if this block is currently shown collapsed (header only).
    fn is_collapsed(&self, block: &Block) -> bool {
        match &block.content {
            BlockContent::ToolCall { collapsed_default, output, .. } =>
                output.is_some() && *collapsed_default && !self.user_expanded.contains(&block.id),
            _ => false,
        }
    }

    /// Toggle a tool block open/closed.
    pub fn toggle_collapsed(&mut self, block_id: &beyonder_core::BlockId) {
        if self.user_expanded.contains(block_id) {
            self.user_expanded.remove(block_id);
        } else {
            self.user_expanded.insert(block_id.clone());
        }
    }

    /// for the currently running block.
    fn block_height(&self, idx: usize, block: &Block, content_w: f32, phys_font: f32) -> f32 {
        if self.is_collapsed(block) {
            phys_font * 1.8 // header bar only
        } else if self.running_block_idx == Some(idx) && !self.tui_cells.is_empty() {
            self.live_block_height(phys_font)
        } else {
            measure_block_height(block, content_w, phys_font)
        }
    }

    fn layout_blocks(&self) -> (Vec<RectInstance>, f32) {
        let mut rects = vec![];
        let sc = self.scale_factor;
        let padding = PADDING * sc;
        let gap = GAP * sc;
        let phys_font = self.font_size * sc;
        let content_w = self.viewport.width - padding * 2.0;
        let mut y = padding;

        for (i, block) in self.blocks.iter().enumerate() {
            let h = self.block_height(i, block, content_w, phys_font);
            let sy = self.viewport.content_to_screen_y(y);

            if self.viewport.is_visible(y, h) {
                let x = padding;
                match &block.content {
                    BlockContent::ShellCommand { .. } => {
                        render_shell_block(block, x, sy, content_w, h, phys_font, sc, &mut rects);
                    }
                    BlockContent::AgentMessage { .. } => {
                        render_agent_message(block, x, sy, content_w, h, sc, &mut rects);
                    }
                    BlockContent::ApprovalRequest { .. } => {
                        render_approval_block(block, x, sy, content_w, h, sc, &mut rects);
                    }
                    _ => {
                        render_block_background(block, x, sy, content_w, h, &mut rects);
                    }
                }
                // Cell background rects — live (TermGrid) and completed (stored output).
                let cmd_bar_h = phys_font * 2.8;
                let inner_gap = phys_font * 0.4;
                let output_pad_x = 4.0 * sc;
                let content_x = x + output_pad_x;
                let content_y = sy + cmd_bar_h + inner_gap;
                let (cell_w, cell_h) = self.terminal_cell_size();
                let rect_h = cell_h.ceil();
                let rect_w = cell_w.ceil();
                if self.running_block_idx == Some(i) && !self.tui_cells.is_empty() {
                    for (row_idx, row) in self.tui_cells.iter().enumerate() {
                        let ry = (content_y + row_idx as f32 * cell_h).floor();
                        if ry > sy + h { break; }
                        for (col_idx, cell) in row.iter().enumerate() {
                            if let Some(bg) = cell.bg {
                                let rx = (content_x + col_idx as f32 * cell_w).floor();
                                rects.push(RectInstance::filled(rx, ry, rect_w, rect_h, [bg[0], bg[1], bg[2], 1.0]));
                            }
                        }
                    }
                    // Cursor
                    let (cur_row, cur_col) = self.tui_cursor;
                    let cx = (content_x + cur_col as f32 * cell_w).floor();
                    let cy = (content_y + cur_row as f32 * cell_h).floor();
                    rects.push(RectInstance::filled(cx, cy, rect_w, rect_h, [0.804, 0.835, 0.918, 0.55]));
                } else if let BlockContent::ShellCommand { output, .. } = &block.content {
                    for (row_idx, row) in output.rows.iter().enumerate() {
                        let ry = (content_y + row_idx as f32 * cell_h).floor();
                        if ry > sy + h { break; }
                        for (col_idx, cell) in row.cells.iter().enumerate() {
                            if let Some(bg) = cell.bg {
                                let rx = (content_x + col_idx as f32 * cell_w).floor();
                                rects.push(RectInstance::filled(rx, ry, rect_w, rect_h,
                                    [bg.r as f32 / 255.0, bg.g as f32 / 255.0, bg.b as f32 / 255.0, 1.0]));
                            }
                        }
                    }
                }
                // Selection highlight — for ShellCommand only the clicked sub-rect lights up.
                if self.selected_block == Some(i) {
                    let cmd_bar_h = phys_font * 2.8;
                    let inner_gap = phys_font * 0.4;
                    let (hl_y, hl_h) = if matches!(block.content, BlockContent::ShellCommand { .. }) {
                        if self.selected_sub_output {
                            let out_y = sy + cmd_bar_h + inner_gap;
                            (out_y, (h - cmd_bar_h - inner_gap).max(1.0))
                        } else {
                            (sy, cmd_bar_h)
                        }
                    } else {
                        (sy, h)
                    };
                    rects.push(
                        RectInstance::filled(x, hl_y, content_w, hl_h, [0.30, 0.55, 0.90, 0.12])
                            .with_radius(3.0)
                            .with_border(1.0, [0.40, 0.65, 1.0, 0.6]),
                    );
                }
            }
            y += h + gap;
        }

        (rects, y)
    }

    /// Draw the input bar chrome (background, separator, pills, mode pill, dropdown,
    /// command palette). Always called regardless of tui_active so the bar is always
    /// visible — even when a full-screen TUI app like nvim is running.
    /// Also updates self.pill_rects, self.mode_pill_rect, self.dropdown_item_rects,
    /// and self.cmd_palette_rects so that text and hit-testing stay in sync.
    fn append_bar_rects(&mut self, rects: &mut Vec<RectInstance>) {
        let win_w = self.surface_config.width as f32;
        let win_h = self.surface_config.height as f32;
        let bar_h = self.computed_bar_h;
        let bar_y = win_h - bar_h;
        let sc = self.scale_factor;
        let phys_font = self.font_size * sc;

        // Bar background + separator.
        let bar_bg = if self.input_running {
            [0.065, 0.065, 0.100, 1.0_f32]
        } else {
            [0.094, 0.094, 0.145, 1.0]
        };
        rects.push(RectInstance::filled(0.0, bar_y, win_w, bar_h, bar_bg));
        rects.push(RectInstance::filled(0.0, bar_y, win_w, sc.ceil(), [0.271, 0.278, 0.353, 0.5]));

        // Context pills.
        let pill_hpad = 12.0 * sc;
        let pill_gap = 8.0 * sc;
        let pill_char_w = phys_font * 0.6 * 0.75;
        let pill_h = 22.0 * sc;
        let pill_top = bar_y + 14.0 * sc;
        let pill_bgs: [[f32; 4]; 3] = [
            [0.155, 0.138, 0.068, 1.0],
            [0.080, 0.148, 0.080, 1.0],
            [0.108, 0.108, 0.198, 1.0],
        ];
        let pill_borders: [[f32; 4]; 3] = [
            [0.976, 0.886, 0.686, 0.75],
            [0.651, 0.890, 0.631, 0.75],
            [0.706, 0.745, 0.996, 0.75],
        ];
        let pill_icons = ['\u{e73c}', '\u{e718}', '\u{f07c}'];
        let mut new_pill_rects: Vec<[f32; 4]> = Vec::new();
        let mut pill_x = 14.0 * sc;
        let pills = self.context_pills.clone();
        for (i, label) in pills.iter().enumerate() {
            let icon = pill_icons.get(i).copied().unwrap_or(' ');
            let full_label = format!("{} {}", icon, label);
            let pill_w = full_label.chars().count() as f32 * pill_char_w + 2.0 * pill_hpad;
            let bg = pill_bgs.get(i).copied().unwrap_or([0.192, 0.196, 0.267, 1.0]);
            let border = pill_borders.get(i).copied().unwrap_or([0.345, 0.357, 0.439, 0.6]);
            rects.push(
                RectInstance::filled(pill_x, pill_top, pill_w, pill_h, bg)
                    .with_radius(4.0)
                    .with_border(1.0, border),
            );
            new_pill_rects.push([pill_x, pill_top, pill_w, pill_h]);
            pill_x += pill_w + pill_gap;
        }
        self.pill_rects = new_pill_rects;

        // Model name pill — top-right of the input bar.
        if !self.agent_model.is_empty() {
            let model_label = format!("\u{f135}  {}", self.agent_model); // rocket icon
            let model_font = phys_font * 0.75;
            let model_char_w = model_font * 0.6;
            let model_w = model_label.chars().count() as f32 * model_char_w + 2.0 * pill_hpad;
            let model_x = win_w - model_w - 14.0 * sc;
            rects.push(
                RectInstance::filled(model_x, pill_top, model_w, pill_h, [0.090, 0.065, 0.130, 1.0])
                    .with_radius(4.0)
                    .with_border(1.0, [0.722, 0.561, 0.957, 0.7]),
            );
        }

        // Mode switcher pill.
        {
            let mode_text = format!("◈  {}", self.mode_label);
            let mode_w = mode_text.chars().count() as f32 * pill_char_w + 2.0 * pill_hpad;
            let mode_h = 20.0 * sc;
            let mode_x = 14.0 * sc;
            let mode_y = bar_y + bar_h - mode_h - 8.0 * sc;
            let mode_bg = match self.mode_label.as_str() {
                "cmd"   => [0.065, 0.095, 0.155, 1.0],
                "agent" => [0.095, 0.065, 0.155, 1.0],
                _       => [0.098, 0.098, 0.118, 1.0],
            };
            let mode_border = match self.mode_label.as_str() {
                "cmd"   => [0.537, 0.706, 0.980, 0.8],
                "agent" => [0.792, 0.651, 0.988, 0.8],
                _       => [0.345, 0.357, 0.439, 0.6],
            };
            rects.push(
                RectInstance::filled(mode_x, mode_y, mode_w, mode_h, mode_bg)
                    .with_radius(4.0)
                    .with_border(1.0, mode_border),
            );
            self.mode_pill_rect = [mode_x, mode_y, mode_w, mode_h];
        }

        // Dropdown rects.
        let mut new_dropdown_rects: Vec<[f32; 4]> = Vec::new();
        if let Some((pill_idx, ref items, ref hovered)) = self.open_dropdown.clone() {
            if let Some(&[px, _py, pw, _ph]) = self.pill_rects.get(pill_idx) {
                let item_h = 22.0 * sc;
                let dd_w = pw.max(120.0 * sc);
                let n = items.len();
                let dd_h_total = n as f32 * item_h;
                let dd_y_start = bar_y - dd_h_total;
                let dd_border = pill_borders.get(pill_idx).copied()
                    .unwrap_or([0.345, 0.357, 0.439, 0.7]);
                rects.push(
                    RectInstance::filled(px, dd_y_start, dd_w, dd_h_total, [0.118, 0.118, 0.180, 1.0])
                        .with_radius(4.0)
                        .with_border(1.0, dd_border),
                );
                for (i, _item) in items.iter().enumerate() {
                    let iy = dd_y_start + i as f32 * item_h;
                    let is_hovered = hovered.map(|h| h == i).unwrap_or(false);
                    let item_bg = if is_hovered {
                        pill_bgs.get(pill_idx).copied().unwrap_or([0.192, 0.196, 0.267, 1.0])
                    } else {
                        [0.118, 0.118, 0.180, 0.0]
                    };
                    rects.push(RectInstance::filled(px, iy, dd_w, item_h, item_bg));
                    new_dropdown_rects.push([px, iy, dd_w, item_h]);
                }
            }
        }
        self.dropdown_item_rects = new_dropdown_rects;

        // Command palette.
        let mut new_palette_rects: Vec<[f32; 4]> = Vec::new();
        if let Some(ref cmds) = self.command_palette.clone() {
            if !cmds.is_empty() {
                let item_h = 28.0 * sc;
                let pal_w = (win_w * 0.6).min(600.0 * sc).max(300.0 * sc);
                let pal_x = 14.0 * sc;
                let n = cmds.len().min(8);
                let pal_h = n as f32 * item_h;
                let pal_y = bar_y - pal_h - 4.0 * sc;
                rects.push(
                    RectInstance::filled(pal_x, pal_y, pal_w, pal_h, [0.094, 0.094, 0.145, 1.0])
                        .with_radius(6.0)
                        .with_border(1.0, [0.271, 0.278, 0.353, 0.8]),
                );
                for i in 0..n {
                    let iy = pal_y + i as f32 * item_h;
                    if self.cmd_palette_hovered == Some(i) {
                        rects.push(RectInstance::filled(pal_x, iy, pal_w, item_h, [0.192, 0.196, 0.267, 1.0]));
                    }
                    new_palette_rects.push([pal_x, iy, pal_w, item_h]);
                }
            }
        }
        self.cmd_palette_rects = new_palette_rects;
    }

    fn tui_cell_size(&self) -> (f32, f32) {
        self.terminal_cell_size()
    }

    fn layout_tui(&self, rects: &mut Vec<RectInstance>) {
        let (cell_w, cell_h) = self.tui_cell_size();
        // Bar is hidden in TUI mode — fill the full window.
        let bar_y = self.surface_config.height as f32;

        for (row_idx, row) in self.tui_cells.iter().enumerate() {
            let row_y = (row_idx as f32 * cell_h).floor();
            if row_y >= bar_y { break; }
            // cell_h is measured to the exact font ascent+descent — no extra pixel.
            let rect_h = cell_h.ceil();
            if row.is_empty() { continue; }
            for (col_idx, cell) in row.iter().enumerate() {
                if let Some(bg) = cell.bg {
                    let col_x = (col_idx as f32 * cell_w).floor();
                    let rect_w = cell_w.ceil() + 1.0;
                    rects.push(RectInstance::filled(col_x, row_y, rect_w, rect_h, [bg[0], bg[1], bg[2], 1.0]));
                }
            }
        }
        // Cursor — shape depends on what the TUI app requested.
        let (cur_row, cur_col) = self.tui_cursor;
        let cx = (cur_col as f32 * cell_w).floor();
        let cy = (cur_row as f32 * cell_h).floor();
        if cy < bar_y {
            let cursor_color = [0.804, 0.835, 0.918, 0.55_f32];
            match self.tui_cursor_shape {
                1 => {
                    // Beam: 2 logical-px wide bar at left edge of cell.
                    let beam_w = (2.0 * self.scale_factor).max(2.0);
                    rects.push(RectInstance::filled(cx, cy, beam_w, cell_h.ceil(), cursor_color));
                }
                2 => {
                    // Underline: thin bar at bottom of cell.
                    let ul_h = (2.0 * self.scale_factor).max(2.0);
                    rects.push(RectInstance::filled(cx, cy + cell_h.ceil() - ul_h, cell_w.ceil(), ul_h, cursor_color));
                }
                _ => {
                    // Block (default).
                    rects.push(RectInstance::filled(cx, cy, cell_w.ceil(), cell_h.ceil(), cursor_color));
                }
            }
        }
    }

    fn build_tui_text_buffers(&mut self) -> TextBufList {
        let (cell_w, cell_h) = self.tui_cell_size();
        // Bar is hidden in TUI mode — fill the full window.
        let bar_y = self.surface_config.height as f32;
        let sc = self.scale_factor;
        let phys_font = self.font_size * sc;
        let mut results = TextBufList::new();

        let cells = self.tui_cells.clone();
        for (row_idx, row) in cells.iter().enumerate() {
            if row.is_empty() { continue; }
            // Match the snapped y from layout_tui so text sits exactly on its bg rect.
            let y = (row_idx as f32 * cell_h).floor();
            if y >= bar_y { break; }

            // Per-run rendering: each color run is positioned at its exact column pixel
            // with a fixed width, so advance-width mismatches for special chars (box-drawing,
            // Nerd Font icons) are contained and don't shift subsequent text rightward.
            let runs = self.make_tui_row_runs(&row, cell_w, phys_font);
            for (buf, x, w, color) in runs {
                results.push((buf, x, y, w, cell_h, color));
            }
        }
        results
    }

    /// Build glyphon Buffers for visible blocks plus bar/pill/dropdown text.
    /// Returns (entries, block_entry_count) — only the first `block_entry_count`
    /// entries should be clipped to `bar_y`.
    fn build_text_buffers(&mut self) -> (TextBufList, usize) {
        let sc = self.scale_factor;
        let padding = PADDING * sc;
        let gap = GAP * sc;
        let phys_font = self.font_size * sc;
        let content_w = self.viewport.width - padding * 2.0;
        let _line_h = phys_font * 1.4;
        let mut results = TextBufList::new();
        let mut y = padding;

        let blocks = self.blocks.clone();
        for (block_idx, block) in blocks.iter().enumerate() {
            let block_t0 = std::time::Instant::now();
            let h = self.block_height(block_idx, block, content_w, phys_font);
            let sy = self.viewport.content_to_screen_y(y);

            if self.viewport.is_visible(y, h) {
                let x = padding;
                // Layout constants (physical px).
                let is_shell = matches!(block.content, BlockContent::ShellCommand { .. });
                let is_agent = matches!(block.content, BlockContent::AgentMessage { .. });
                // Agent messages have no header bar — content starts with padding only.
                let cmd_bar_h = if is_shell { phys_font * 2.8 } else if is_agent { 0.0 } else { phys_font * 1.6 };
                let inner_gap = if is_agent { phys_font * 0.6 } else { phys_font * 0.4 };
                let hdr_pad = 10.0 * sc;
                let content_pad = 8.0 * sc;

                if let BlockContent::ShellCommand { input, cwd, duration_ms, .. } = &block.content {
                    // Meta row: full ~-abbreviated cwd + execution time, above the command.
                    let meta_font = phys_font * 0.88;
                    let meta_line_h = meta_font * 1.4;
                    let meta_y = sy + 4.0 * sc;
                    let home = std::env::var("HOME").unwrap_or_default();
                    let cwd_str = cwd.to_str().unwrap_or("~");
                    let dir_display = if !home.is_empty() && cwd_str.starts_with(home.as_str()) {
                        format!("~{}", &cwd_str[home.len()..])
                    } else {
                        cwd_str.to_string()
                    };
                    let dur_str = duration_ms.map(format_duration);
                    let meta_text = match &dur_str {
                        Some(d) => format!("{}  {}", dir_display, d),
                        None => dir_display,
                    };
                    let meta_color = GlyphColor::rgb(166, 173, 200); // between Overlay2 and Subtext1
                    let meta_buf = self.make_buffer(&meta_text, content_w - hdr_pad * 2.0, meta_font, meta_color);
                    results.push((meta_buf, x + hdr_pad, meta_y, content_w - hdr_pad * 2.0, meta_line_h, meta_color));

                    // Command row (normal font): the shell command itself.
                    let cmd_text_y = meta_y + meta_line_h + 3.0 * sc;
                    let cmd_color = GlyphColor::rgb(205, 214, 244); // Text
                    let cmd_buf = self.make_buffer(input, content_w - hdr_pad * 2.0, phys_font, cmd_color);
                    results.push((cmd_buf, x + hdr_pad, cmd_text_y, content_w - hdr_pad * 2.0, phys_font * 1.4, cmd_color));
                } else if !is_agent {
                    // Non-shell, non-agent blocks: single centered header line.
                    let raw_label = block_header_label(block);
                    // Only output-bearing ToolCall blocks get a collapse chevron.
                    let has_tool_output = matches!(&block.content,
                        BlockContent::ToolCall { output, .. } if output.is_some());
                    let header_label = if has_tool_output {
                        let chevron = if self.is_collapsed(block) { "▶ " } else { "▼ " };
                        format!("{}{}", chevron, raw_label)
                    } else {
                        raw_label
                    };
                    let header_color = match block.kind {
                        BlockKind::Agent    => GlyphColor::rgb(137, 180, 250),
                        BlockKind::Approval => GlyphColor::rgb(249, 226, 175),
                        BlockKind::Tool     => GlyphColor::rgb(148, 226, 213),
                        _                   => GlyphColor::rgb(186, 194, 222),
                    };
                    let header_buf = self.make_buffer(
                        &header_label,
                        content_w - hdr_pad * 2.0,
                        phys_font,
                        header_color,
                    );
                    let hdr_text_y = sy + (cmd_bar_h - phys_font * 1.4) * 0.5;
                    results.push((header_buf, x + hdr_pad, hdr_text_y, content_w - hdr_pad * 2.0, phys_font * 1.4, header_color));
                }

                // Output area starts below the cmd bar + gap.
                let output_top = sy + cmd_bar_h + inner_gap;

                // Live running block: render TermGrid rows inside the output panel.
                let output_pad_x = 4.0 * sc;
                if self.running_block_idx == Some(block_idx) && !self.tui_cells.is_empty() {
                    let (cell_w, cell_h) = self.terminal_cell_size();
                    let tui_cells = self.tui_cells.clone();
                    for (row_idx, row) in tui_cells.iter().enumerate() {
                        if row.is_empty() { continue; }
                        let ry = (output_top + row_idx as f32 * cell_h).floor();
                        if ry > sy + h { break; }
                        let runs = self.make_tui_row_runs(&row, cell_w, phys_font);
                        for (buf, rx, w, color) in runs {
                            results.push((buf, x + output_pad_x + rx, ry, w, cell_h, color));
                        }
                    }
                } else {
                    match &block.content {
                        BlockContent::ShellCommand { output, .. } => {
                            // Render per-cell with stored colors (preserves ANSI/TUI colors).
                            let (cell_w, cell_h) = self.terminal_cell_size();
                            let content_x = x + output_pad_x;
                            let base_y = output_top + 2.0 * sc;
                            for (row_idx, row) in output.rows.iter().enumerate() {
                                let row_y = (base_y + row_idx as f32 * cell_h).floor();
                                if row_y > sy + h { break; }
                                let any_visible = row.cells.iter().any(|c| c.character != ' ' && c.character != '\0');
                                if !any_visible { continue; }
                                let tui_row: Vec<TuiCell> = row.cells.iter().map(|c| TuiCell {
                                    ch: c.character,
                                    fg: c.fg.map(|col| [col.r as f32 / 255.0, col.g as f32 / 255.0, col.b as f32 / 255.0])
                                           .unwrap_or([0.804, 0.835, 0.918]),
                                    bg: c.bg.map(|col| [col.r as f32 / 255.0, col.g as f32 / 255.0, col.b as f32 / 255.0]),
                                    bold: c.bold,
                                    italic: c.italic,
                                }).collect();
                                let runs = self.make_tui_row_runs(&tui_row, cell_w, phys_font);
                                for (buf, rx, w, color) in runs {
                                    results.push((buf, content_x + rx, row_y, w, cell_h, color));
                                }
                            }
                        }
                        _ => if !self.is_collapsed(block) {
                            let content_text = block_content_text(block);
                            let hdr_h = cmd_bar_h;
                            let content_h = h - hdr_h - content_pad;
                            let buf_w = content_w - content_pad * 2.0;
                            let text_y = sy + hdr_h + 4.0 * sc;

                            // Spinner for streaming agent blocks.
                            let is_running_agent = is_agent
                                && matches!(block.status, BlockStatus::Running);
                            let running_tool = block.agent_id.as_ref()
                                .and_then(|id| self.agent_running_tool.get(id))
                                .cloned();
                            const FRAMES: [&str; 10] = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"];
                            let spin_color = GlyphColor::rgb(137, 180, 250); // Catppuccin Blue
                            if is_running_agent && content_text.is_empty() {
                                // No text yet — centred spinner (or "tool_name..." label).
                                let frame = FRAMES[self.spinner_frame as usize];
                                let label = if let Some(ref tname) = running_tool {
                                    format!("{} {}…", frame, tname)
                                } else {
                                    frame.to_string()
                                };
                                let spin_buf = self.make_buffer(&label, buf_w, phys_font, spin_color);
                                let spin_h = phys_font * 1.4;
                                let spin_x = x + content_pad;
                                let spin_y = sy + (h - spin_h) * 0.5;
                                results.push((spin_buf, spin_x, spin_y, buf_w, spin_h, spin_color));
                            } else if is_running_agent && running_tool.is_some() {
                                // Content already rendered + tool is executing — show
                                // a compact tool indicator at the bottom of the block.
                                let frame = FRAMES[self.spinner_frame as usize];
                                let tname = running_tool.unwrap();
                                let label = format!("{} {}…", frame, tname);
                                let spin_buf = self.make_buffer(&label, buf_w, phys_font * 0.9, spin_color);
                                let spin_h = phys_font * 1.3;
                                let spin_x = x + content_pad;
                                // Anchor just above the block bottom edge.
                                let spin_y = sy + h - spin_h - 4.0 * sc;
                                results.push((spin_buf, spin_x, spin_y, buf_w, spin_h, spin_color));
                            }

                            if !content_text.is_empty() {
                                let is_user_msg = matches!(&block.content,
                                    BlockContent::AgentMessage { role: beyonder_core::MessageRole::User, .. });
                                let fallback_color = GlyphColor::rgb(205, 214, 244);
                                // Cache key: (content_len, buf_w bits, phys_font bits, viewport_h bits).
                                let content_len = content_text.len() as u64;
                                let bw_bits = buf_w.to_bits();
                                let pf_bits = phys_font.to_bits();
                                let vh_bits = self.viewport.height.to_bits();
                                if is_agent && !is_user_msg {
                                    // Try to reuse a previously shaped markdown buffer.
                                    let cached = self.glyph_buf_cache.remove(&block.id);
                                    let (buf, skipped, cache_len) = match cached {
                                        Some((len, bw, pf, vh, b))
                                            if len == content_len
                                               && bw == bw_bits && pf == pf_bits && vh == vh_bits =>
                                        {
                                            // Content unchanged — reuse shaped buffer as-is.
                                            let line_h = phys_font * 1.4;
                                            let max_vis = ((self.viewport.height / line_h).ceil() as usize + 30).max(50);
                                            let total = content_text.lines().count();
                                            (b, total.saturating_sub(max_vis), len)
                                        }
                                        _ => {
                                            let (b, s) = self.make_markdown_buffer(&content_text, buf_w, phys_font);
                                            (b, s, content_len)
                                        }
                                    };
                                    // Offset text_y down by the skipped lines so the
                                    // visible tail renders at the correct screen position.
                                    let line_h = phys_font * 1.4;
                                    let adjusted_text_y = text_y + skipped as f32 * line_h;
                                    results.push_cached(
                                        (buf, x + content_pad, adjusted_text_y, buf_w, content_h.max(1.0), fallback_color),
                                        (block.id.clone(), cache_len, bw_bits, pf_bits, vh_bits),
                                    );
                                } else {
                                    let content_buf = if is_user_msg {
                                        let col = GlyphColor::rgb(166, 173, 200); // dimmer — user echo
                                        self.make_buffer(&content_text, buf_w, phys_font, col)
                                    } else {
                                        let text_color = match block.kind {
                                            BlockKind::Approval => GlyphColor::rgb(250, 179, 135),
                                            BlockKind::Tool     => GlyphColor::rgb(137, 220, 235),
                                            _                   => GlyphColor::rgb(205, 214, 244),
                                        };
                                        self.make_buffer(&content_text, buf_w, phys_font, text_color)
                                    };
                                    results.push((content_buf, x + content_pad, text_y, buf_w, content_h.max(1.0), fallback_color));
                                }
                            }
                        }
                    }
                }
            }
            let block_ms = block_t0.elapsed().as_millis();
            if block_ms > 5 {
                debug!(
                    block_idx,
                    kind = ?block.kind,
                    status = ?block.status,
                    elapsed_ms = block_ms,
                    "build_text_buffers: slow block"
                );
            }
            y += h + gap;
        }

        // Block entries end here. Bar text is appended separately via build_bar_text_buffers.
        let block_entry_count = results.len();
        (results, block_entry_count)
    }

    /// Build GlyphBuffers for the input bar chrome: input field, context pills,
    /// mode switcher, dropdown items, and command palette.
    /// All entries are unclamped (rendered on top of the bar, not clipped at bar_y).
    /// MUST be called after append_bar_rects so pill_rects/mode_pill_rect are current.
    fn build_bar_text_buffers(&mut self) -> TextBufList {
        let sc = self.scale_factor;
        let phys_font = self.font_size * sc;
        let win_w = self.surface_config.width as f32;
        let win_h = self.surface_config.height as f32;
        let bar_h = self.computed_bar_h;
        let bar_y = win_h - bar_h;
        let h_pad = 14.0 * sc;
        let text_x = h_pad;
        let text_w = win_w - h_pad * 2.0;
        let line_h = phys_font * 1.4;
        let text_zone_top = bar_y + 14.0 * sc + 22.0 * sc + 7.0 * sc;
        let mode_zone_h = 20.0 * sc + 8.0 * sc;
        let remaining_h = bar_h - (text_zone_top - bar_y) - mode_zone_h;
        let text_y = text_zone_top + (remaining_h - line_h) * 0.5;

        let mut results = TextBufList::new();

        if self.input_running {
            let text = "running…".to_string();
            let col = GlyphColor::rgb(108, 112, 134);
            let buf = self.make_buffer(&text, text_w, phys_font, col);
            results.push((buf, text_x, text_y, text_w, line_h, col));
        } else if self.input_text.is_empty() {
            let caret = if self.cursor_blink_on { "▌" } else { " " };
            let caret_col = GlyphColor::rgb(205, 214, 244);
            let caret_w = (phys_font * 0.6).round();
            let caret_buf = self.make_buffer(caret, caret_w * 2.0, phys_font, caret_col);
            results.push((caret_buf, text_x, text_y, caret_w * 2.0, line_h, caret_col));
            let ph = "Type anything, beyonder will pick up whether it's a command or prompt";
            let ph_col = GlyphColor::rgb(108, 112, 134);
            let ph_x = text_x + caret_w;
            let ph_w = (text_w - caret_w).max(1.0);
            let ph_buf = self.make_buffer(ph, ph_w, phys_font, ph_col);
            results.push((ph_buf, ph_x, text_y, ph_w, line_h, ph_col));
        } else {
            let cursor = self.input_cursor.min(self.input_text.len());
            let before = &self.input_text[..cursor];
            let after = &self.input_text[cursor..];
            let (text, col) = if self.input_all_selected {
                // Render the whole line in accent colour with a block cursor to
                // signal "all selected — next keystroke replaces everything".
                let t = format!("{}█{}", self.input_mode_prefix, self.input_text);
                (t, GlyphColor::rgb(137, 180, 250)) // Catppuccin Blue
            } else {
                let caret = if self.cursor_blink_on { "▌" } else { " " };
                let t = format!("{}{}{}{}", self.input_mode_prefix, before, caret, after);
                (t, GlyphColor::rgb(205, 214, 244))
            };

            let phys_font_local = self.font_size * sc;
            let line_h_local = phys_font_local * 1.4;
            let visible_lines = self.measure_input_lines(text_w, phys_font_local).0
                .min(MAX_INPUT_LINES);
            let text_area_h = visible_lines as f32 * line_h_local;

            // Multi-line centering: place the text block vertically centred in the
            // remaining space between the pills row and the mode pill.
            let remaining_h = bar_h - (text_zone_top - bar_y) - mode_zone_h;
            let text_block_y = text_zone_top + (remaining_h - text_area_h).max(0.0) * 0.5;

            // Bounded buffer + set_scroll: layout_runs() subtracts scroll.vertical
            // from each run's Y, so visible lines have Y ∈ [0, text_area_h] and
            // pre-scroll lines have negative Y. TextArea.top = text_block_y maps
            // visible lines to screen [text_block_y, text_block_y+text_area_h].
            // Clip tightly to that range so negative-Y runs don't bleed into pills.
            let metrics = glyphon::Metrics::new(phys_font, phys_font * 1.4);
            let mut buf = GlyphBuffer::new(&mut self.font_system, metrics);
            buf.set_size(&mut self.font_system, Some(text_w), Some(text_area_h));
            buf.set_text(
                &mut self.font_system,
                &text,
                glyphon::Attrs::new()
                    .family(glyphon::Family::Name("JetBrainsMono Nerd Font"))
                    .color(col),
                glyphon::Shaping::Advanced,
            );
            buf.set_scroll(glyphon::cosmic_text::Scroll {
                line: 0,
                vertical: self.input_scroll_px,
                horizontal: 0.0,
            });
            buf.shape_until_scroll(&mut self.font_system, false);
            let run_tops: Vec<f32> = buf.layout_runs().map(|r| r.line_top).collect();
            let clip_top = text_block_y as i32;
            let clip_bottom = (text_block_y + text_area_h) as i32;
            tracing::info!(
                input_scroll_px = self.input_scroll_px,
                text_block_y,
                text_area_h,
                clip_top,
                clip_bottom,
                visible_lines,
                ?run_tops,
                "bar text layout"
            );
            results.push_clipped((buf, text_x, text_block_y, text_w, text_area_h, col), (clip_top, clip_bottom));
        }

        // Pill text — one entry per pill.
        let pill_top = bar_y + 14.0 * sc;
        let pill_h = 22.0 * sc;
        let pill_hpad = 12.0 * sc;
        let pill_gap = 8.0 * sc;
        let pill_char_w = phys_font * 0.6 * 0.75;
        let pill_font_size = phys_font * 0.75;
        let pill_line_h = pill_font_size * 1.4;
        let pill_text_y = pill_top + (pill_h - pill_line_h) * 0.5;
        let pill_icons = ['\u{e73c}', '\u{e718}', '\u{f07c}'];
        let pill_text_colors = [
            GlyphColor::rgb(249, 226, 175),
            GlyphColor::rgb(166, 227, 161),
            GlyphColor::rgb(180, 190, 254),
        ];
        let pills = self.context_pills.clone();
        let mut pill_x = 14.0 * sc;
        for (i, label) in pills.iter().enumerate() {
            let icon = pill_icons.get(i).copied().unwrap_or(' ');
            let full_label = format!("{} {}", icon, label);
            let pill_w = full_label.chars().count() as f32 * pill_char_w + 2.0 * pill_hpad;
            let color = pill_text_colors.get(i).copied()
                .unwrap_or(GlyphColor::rgb(186, 194, 222));
            let pill_buf = self.make_pill_buffer(&full_label, pill_w - 2.0 * pill_hpad, pill_font_size, color);
            results.push((pill_buf, pill_x + pill_hpad, pill_text_y, pill_w - 2.0 * pill_hpad, pill_line_h, color));
            pill_x += pill_w + pill_gap;
        }

        // Model name pill text — top-right.
        if !self.agent_model.is_empty() {
            let model_label = format!("\u{f135}  {}", self.agent_model);
            let model_font = phys_font * 0.75;
            let model_char_w = model_font * 0.6;
            let pill_hpad = 12.0 * sc;
            let model_w = model_label.chars().count() as f32 * model_char_w + 2.0 * pill_hpad;
            let model_x = win_w - model_w - 14.0 * sc;
            let pill_top = bar_y + 14.0 * sc;
            let pill_h = 22.0 * sc;
            let model_line_h = model_font * 1.4;
            let model_ty = pill_top + (pill_h - model_line_h) * 0.5;
            let model_color = GlyphColor::rgb(203, 143, 247);
            let model_buf = self.make_pill_buffer(&model_label, model_w - 2.0 * pill_hpad, model_font, model_color);
            results.push((model_buf, model_x + pill_hpad, model_ty, model_w - 2.0 * pill_hpad, model_line_h, model_color));
        }

        // Mode switcher text.
        {
            let [mode_x, mode_y, mode_w, mode_h] = self.mode_pill_rect;
            if mode_w > 0.0 {
                let mode_text = format!("◈  {}", self.mode_label);
                let mode_font = phys_font * 0.75;
                let mode_line_h = mode_font * 1.4;
                let mode_color = match self.mode_label.as_str() {
                    "cmd"   => GlyphColor::rgb(137, 180, 250),
                    "agent" => GlyphColor::rgb(203, 166, 247),
                    _       => GlyphColor::rgb(147, 153, 178),
                };
                let hpad = 12.0 * sc;
                let mode_buf = self.make_pill_buffer(&mode_text, mode_w - 2.0 * hpad, mode_font, mode_color);
                let ty = mode_y + (mode_h - mode_line_h) * 0.5;
                results.push((mode_buf, mode_x + hpad, ty, mode_w - 2.0 * hpad, mode_line_h, mode_color));
            }
        }

        // Dropdown text.
        if let Some((pill_idx, ref items, _)) = self.open_dropdown.clone() {
            if let Some(&[px, _py, pw, _ph]) = self.pill_rects.get(pill_idx) {
                let item_h = 22.0 * sc;
                let item_v_pad = 3.0 * sc;
                let dd_w = pw.max(120.0 * sc);
                let n = items.len();
                let dd_y_start = bar_y - n as f32 * item_h;
                let dd_text_colors = [
                    GlyphColor::rgb(249, 226, 175),
                    GlyphColor::rgb(166, 227, 161),
                    GlyphColor::rgb(180, 190, 254),
                ];
                let dd_text_color = dd_text_colors.get(pill_idx).copied()
                    .unwrap_or(GlyphColor::rgb(205, 214, 244));
                for (i, item) in items.iter().enumerate() {
                    let iy = dd_y_start + i as f32 * item_h + item_v_pad;
                    let item_buf = self.make_buffer(item, dd_w - pill_hpad * 2.0, phys_font, dd_text_color);
                    results.push((item_buf, px + pill_hpad, iy, dd_w - pill_hpad * 2.0, item_h, dd_text_color));
                }
            }
        }

        // Command palette text.
        if let Some(ref cmds) = self.command_palette.clone() {
            let n = cmds.len().min(8);
            if n > 0 {
                let item_h = 28.0 * sc;
                let pal_w = (win_w * 0.6).min(600.0 * sc).max(300.0 * sc);
                let pal_x = 14.0 * sc;
                let pal_y = bar_y - n as f32 * item_h - 4.0 * sc;
                let usage_col = GlyphColor::rgb(180, 190, 254);
                let desc_col  = GlyphColor::rgb(108, 112, 134);
                let pal_font  = phys_font * 0.88;
                let pal_line_h = pal_font * 1.4;
                let v_pad = (item_h - pal_line_h) * 0.5;
                let h_pad = 10.0 * sc;
                let usage_w = pal_w * 0.38;
                let desc_x = pal_x + h_pad + usage_w + 8.0 * sc;
                let desc_w = pal_w - usage_w - h_pad * 2.0 - 8.0 * sc;
                for (i, (usage, desc)) in cmds.iter().take(n).enumerate() {
                    let iy = pal_y + i as f32 * item_h + v_pad;
                    let usage_buf = self.make_buffer(usage, usage_w, pal_font, usage_col);
                    results.push((usage_buf, pal_x + h_pad, iy, usage_w, pal_line_h, usage_col));
                    let desc_buf = self.make_buffer(desc, desc_w, pal_font, desc_col);
                    results.push((desc_buf, desc_x, iy, desc_w, pal_line_h, desc_col));
                }
            }
        }

        results
    }

    /// Draw tab-strip backgrounds + updates `self.tab_rects` for hit-testing.
    /// No-ops when fewer than 2 tabs or TUI active (tab_bar_height_phys() == 0).
    fn append_tab_bar_rects(&mut self, rects: &mut Vec<RectInstance>) {
        let tab_h = self.tab_bar_height_phys();
        if tab_h <= 0.0 || self.tab_labels.is_empty() {
            self.tab_rects.clear();
            return;
        }
        let sc = self.scale_factor;
        let win_w = self.surface_config.width as f32;
        // Strip background (Catppuccin Mantle).
        rects.push(RectInstance::filled(0.0, 0.0, win_w, tab_h, [0.094, 0.094, 0.145, 1.0]));
        // Bottom separator.
        rects.push(RectInstance::filled(0.0, tab_h - sc.ceil(), win_w, sc.ceil(), [0.271, 0.278, 0.353, 0.6]));

        let pad_x = 8.0 * sc;
        let gap = 4.0 * sc;
        let inner_pad = 12.0 * sc;
        let phys_font = self.font_size * sc * 0.85;
        let char_w = phys_font * 0.6;
        let tab_inner_h = tab_h - 6.0 * sc;
        let tab_y = 3.0 * sc;

        let mut x = pad_x;
        let labels = self.tab_labels.clone();
        let active = self.active_tab;
        let mut new_rects: Vec<[f32; 4]> = Vec::with_capacity(labels.len());
        for (i, label) in labels.iter().enumerate() {
            let tab_w = label.chars().count() as f32 * char_w + inner_pad * 2.0;
            let is_active = i == active;
            let (bg, border) = if is_active {
                ([0.192, 0.196, 0.267, 1.0_f32], [0.537, 0.706, 0.980, 0.9_f32])
            } else {
                ([0.125, 0.125, 0.180, 1.0_f32], [0.271, 0.278, 0.353, 0.5_f32])
            };
            rects.push(
                RectInstance::filled(x, tab_y, tab_w, tab_inner_h, bg)
                    .with_radius(4.0)
                    .with_border(1.0, border),
            );
            new_rects.push([x, tab_y, tab_w, tab_inner_h]);
            x += tab_w + gap;
        }
        self.tab_rects = new_rects;
    }

    /// Append tab-strip label text to the given TextBufList.
    fn build_tab_bar_text_buffers(&mut self, results: &mut TextBufList) {
        let tab_h = self.tab_bar_height_phys();
        if tab_h <= 0.0 || self.tab_rects.is_empty() {
            return;
        }
        let sc = self.scale_factor;
        let phys_font = self.font_size * sc * 0.85;
        let line_h = phys_font * 1.4;
        let labels = self.tab_labels.clone();
        let active = self.active_tab;
        let rects = self.tab_rects.clone();
        let inner_pad = 12.0 * sc;
        for (i, label) in labels.iter().enumerate() {
            let Some(&[rx, ry, rw, rh]) = rects.get(i) else { continue };
            let color = if i == active {
                GlyphColor::rgb(205, 214, 244)
            } else {
                GlyphColor::rgb(147, 153, 178)
            };
            let ty = ry + (rh - line_h) * 0.5;
            let buf = self.make_pill_buffer(label, (rw - inner_pad * 2.0).max(1.0), phys_font, color);
            results.push((buf, rx + inner_pad, ty, rw - inner_pad * 2.0, line_h, color));
        }
    }

    fn make_pill_buffer(
        &mut self,
        text: &str,
        max_width: f32,
        size: f32,
        color: GlyphColor,
    ) -> GlyphBuffer {
        let metrics = Metrics::new(size, size * 1.4);
        let mut buf = GlyphBuffer::new(&mut self.font_system, metrics);
        buf.set_size(&mut self.font_system, Some(max_width), None);
        buf.set_text(
            &mut self.font_system,
            text,
            Attrs::new().family(Family::Name("JetBrainsMono Nerd Font")).color(color),
            Shaping::Advanced,
        );
        buf.shape_until_scroll(&mut self.font_system, false);
        buf
    }

    fn make_buffer(
        &mut self,
        text: &str,
        max_width: f32,
        size: f32,
        color: GlyphColor,
    ) -> GlyphBuffer {
        let metrics = Metrics::new(size, size * 1.4);
        let mut buf = GlyphBuffer::new(&mut self.font_system, metrics);
        buf.set_size(&mut self.font_system, Some(max_width), None);
        // Advanced required: spinner braille (⠋…), Nerd Font symbols, box-drawing.
        buf.set_text(
            &mut self.font_system,
            text,
            Attrs::new().family(Family::Name("JetBrainsMono Nerd Font")).color(color),
            Shaping::Advanced,
        );
        buf.shape_until_scroll(&mut self.font_system, false);
        buf
    }

    /// Build a GlyphBuffer for agent markdown text using per-span colors.
    /// Handles headings, **bold**, `inline code`, fenced code blocks, and list items.
    /// Returns `(buffer, skipped_lines)` — only the last `max_vis_lines` lines of
    /// `text` are shaped so reshape cost is O(viewport) not O(total_response).
    fn make_markdown_buffer(&mut self, text: &str, max_width: f32, size: f32) -> (GlyphBuffer, usize) {
        use glyphon::Weight;

        let line_h = size * 1.4;
        // Shape only the last viewport-height worth of lines + a small lookahead
        // buffer so scrolling slightly above the visible area still looks correct.
        let max_vis_lines = ((self.viewport.height / line_h).ceil() as usize + 30).max(50);
        let all_lines: Vec<&str> = text.lines().collect();
        let total_lines = all_lines.len();
        let skipped_lines = total_lines.saturating_sub(max_vis_lines);
        // Work on the visible tail only.
        let visible_text: std::borrow::Cow<str> = if skipped_lines > 0 {
            std::borrow::Cow::Owned(all_lines[skipped_lines..].join("\n"))
        } else {
            std::borrow::Cow::Borrowed(text)
        };
        let text = visible_text.as_ref();

        let base_color   = GlyphColor::rgb(205, 214, 244); // Text
        let heading_color= GlyphColor::rgb(180, 190, 254); // Lavender
        let code_color   = GlyphColor::rgb(137, 220, 235); // Teal
        let bold_color   = GlyphColor::rgb(255, 255, 255); // bright white
        let fence_color  = GlyphColor::rgb(166, 227, 161); // green

        let metrics = Metrics::new(size, size * 1.4);
        let mut buf = GlyphBuffer::new(&mut self.font_system, metrics);
        buf.set_size(&mut self.font_system, Some(max_width), None);

        let font_name = Family::Name("JetBrainsMono Nerd Font");
        let default_attrs = Attrs::new().family(font_name).color(base_color);

        // Build spans: (text, Attrs)
        let mut spans: Vec<(String, GlyphColor, bool)> = vec![]; // (text, color, bold)

        let mut in_fence = false;
        for (li, line) in text.lines().enumerate() {
            let nl = if li > 0 { "\n" } else { "" };

            // Fenced code block toggle.
            if line.starts_with("```") {
                in_fence = !in_fence;
                spans.push((format!("{}{}", nl, line), fence_color, false));
                continue;
            }
            if in_fence {
                spans.push((format!("{}{}", nl, line), fence_color, false));
                continue;
            }

            // Heading — strip markers, differentiate by color only.
            if let Some(rest) = line.strip_prefix("### ") {
                spans.push((format!("{}{}", nl, rest), heading_color, false));
                continue;
            }
            if let Some(rest) = line.strip_prefix("## ") {
                spans.push((format!("{}{}", nl, rest), heading_color, false));
                continue;
            }
            if let Some(rest) = line.strip_prefix("# ") {
                spans.push((format!("{}{}", nl, rest), heading_color, false));
                continue;
            }

            // List item bullet.
            let (parse_line, line_pfx) = if line.starts_with("- ") || line.starts_with("* ") {
                spans.push((format!("{}• ", nl), base_color, false));
                (&line[2..], "")
            } else {
                (line, nl)
            };

            // Inline spans: **bold** and `code`.
            parse_inline(line_pfx, parse_line, base_color, bold_color, code_color, &mut spans);
        }

        // Convert to rich-text spans for glyphon.
        let rich: Vec<(String, Attrs)> = spans
            .iter()
            .map(|(text, color, bold)| {
                let mut attrs = Attrs::new().family(font_name).color(*color);
                if *bold {
                    attrs = attrs.weight(Weight::BOLD);
                }
                (text.clone(), attrs)
            })
            .collect();

        buf.set_rich_text(
            &mut self.font_system,
            rich.iter().map(|(t, a)| (t.as_str(), *a)),
            default_attrs,
            Shaping::Advanced,
        );
        buf.shape_until_scroll(&mut self.font_system, false);
        (buf, skipped_lines)
    }

    /// Build a GlyphBuffer for a single terminal row with per-character colors.
    /// Uses `set_rich_text` so each span gets its own fg color, and disables
    /// word-wrap with `set_size(None, None)` so terminal rows never reflow.
    /// Build a single-color GlyphBuffer for a run of characters.
    /// No width constraint — wide/Nerd Font chars would wrap and disappear with a fixed width.
    /// The caller positions the buffer at the correct column and TextBounds clips at the run boundary.
    fn make_tui_run_buffer(&mut self, text: &str, color: GlyphColor, phys_font: f32, _width: f32, shaping: Shaping) -> GlyphBuffer {
        // Use exact (non-floored) line height so centering_offset = 0.
        // This places glyphs at the exact cell top — prevents 1px downward shift
        // that breaks tiling of box-drawing and block-element characters.
        let metrics = Metrics::new(phys_font, self.measured_metrics_line_h);
        let mut buf = GlyphBuffer::new(&mut self.font_system, metrics);
        // No width constraint — prevents wrapping. TextBounds.right clips overflow at the run boundary.
        buf.set_size(&mut self.font_system, None, None);
        let font_name = Family::Name("JetBrainsMono Nerd Font");
        buf.set_text(
            &mut self.font_system,
            text,
            Attrs::new().family(font_name).color(color),
            shaping,
        );
        buf.shape_until_scroll(&mut self.font_system, false);
        buf
    }

    /// Decompose a row of TuiCells into per-color-run buffers for column-exact rendering.
    /// Returns `(buf, x_offset, width, color)` for each non-whitespace run.
    /// Each run is positioned at `run_start_col * cell_w` and capped to `run_len * cell_w`,
    /// so special-char advance-width mismatch is contained within the run and never accumulates.
    fn make_tui_row_runs(&mut self, cells: &[TuiCell], cell_w: f32, phys_font: f32) -> Vec<(GlyphBuffer, f32, f32, GlyphColor)> {
        let mut result = Vec::new();
        let mut i = 0;
        while i < cells.len() {
            // Skip null chars — these are wide-char spacer cells that alacritty writes
            // in the column adjacent to a 2-cell-wide character. Including them as spaces
            // in a run produces a phantom space glyph next to the wide char.
            if cells[i].ch == '\0' {
                i += 1;
                continue;
            }
            let run_start = i;
            let run_fg = cells[i].fg;
            // End the run on fg change OR any null cell (spacer boundary).
            while i < cells.len() && cells[i].fg == run_fg && cells[i].ch != '\0' {
                i += 1;
            }
            let run_cells = &cells[run_start..i];
            let text: String = run_cells.iter().map(|c| {
                if (c.ch as u32) < 32 { ' ' } else { c.ch }
            }).collect();
            if text.trim().is_empty() { continue; }

            let color = GlyphColor::rgb(
                (run_fg[0] * 255.0) as u8,
                (run_fg[1] * 255.0) as u8,
                (run_fg[2] * 255.0) as u8,
            );
            let x = (run_start as f32 * cell_w).floor();
            // Extend width to cover any '\0' spacer cells immediately following this run.
            // Wide characters (e.g. Nerd Font icons) occupy 2 columns — alacritty writes
            // a '\0' spacer in the adjacent column. Without this, the icon is clipped at
            // the run boundary and the spacer cell appears as a blank colored strip.
            let mut end_col = i;
            while end_col < cells.len() && cells[end_col].ch == '\0' {
                end_col += 1;
            }
            let col_span = end_col - run_start;
            let w = (col_span as f32 * cell_w).ceil();
            let needs_adv = run_cells.iter().any(|c| c.ch as u32 > 127);
            let shaping = if needs_adv { Shaping::Advanced } else { Shaping::Basic };
            let buf = self.make_tui_run_buffer(&text, color, phys_font, w, shaping);
            result.push((buf, x, w, color));
        }
        result
    }
}

// -------------------------------------------------------------------------
// Text extraction helpers
// -------------------------------------------------------------------------

fn format_duration(ms: u64) -> String {
    if ms < 1_000 {
        format!("{}ms", ms)
    } else if ms < 60_000 {
        format!("{:.1}s", ms as f32 / 1000.0)
    } else {
        format!("{}m{}s", ms / 60_000, (ms % 60_000) / 1_000)
    }
}

fn block_header_label(block: &Block) -> String {
    match &block.content {
        BlockContent::ShellCommand { input, .. } => input.clone(),
        BlockContent::AgentMessage { role, .. } => {
            let role_str = match role {
                beyonder_core::MessageRole::Assistant => "agent",
                beyonder_core::MessageRole::User => "user",
                beyonder_core::MessageRole::System => "system",
            };
            let agent = block
                .agent_id
                .as_ref()
                .map(|a| {
                    // Extract just the name part (before the ULID hyphen)
                    a.0.split('-').next().unwrap_or(&a.0).to_string()
                })
                .unwrap_or_else(|| role_str.to_string());
            format!("◆ {}", agent)
        }
        BlockContent::ApprovalRequest { action, .. } => {
            format!("⚠ Approval Required: {}", action_summary(action))
        }
        BlockContent::ToolCall { tool_name, input, .. } => {
            if tool_name == "shell.exec" {
                input.get("cmd").and_then(|v| v.as_str()).unwrap_or("shell").to_string()
            } else {
                let detail = input.get("path")
                    .or_else(|| input.get("url"))
                    .or_else(|| input.get("query"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if detail.is_empty() { format!("⚙ {}", tool_name) } else { format!("⚙ {} {}", tool_name, detail) }
            }
        }
        BlockContent::PlanNode { description, .. } => {
            format!("◎ Plan: {}", description)
        }
        BlockContent::FileEdit { path, .. } => {
            format!("~ Edit: {}", path.display())
        }
        BlockContent::Text { text } => text.chars().take(60).collect(),
    }
}

fn block_content_text(block: &Block) -> String {
    match &block.content {
        BlockContent::ShellCommand { output, .. } => {
            output
                .rows
                .iter()
                .map(|row| row.cells.iter().map(|c| c.character).collect::<String>())
                .collect::<Vec<_>>()
                .join("\n")
        }
        BlockContent::AgentMessage { content_blocks, .. } => {
            content_blocks
                .iter()
                .map(|cb| match cb {
                    beyonder_core::ContentBlock::Text { text } => text.clone(),
                    beyonder_core::ContentBlock::Code { code, language } => {
                        let lang = language.as_deref().unwrap_or("");
                        format!("```{}\n{}\n```", lang, code)
                    }
                    beyonder_core::ContentBlock::Thinking { thinking } => {
                        format!("<thinking>{}</thinking>", thinking)
                    }
                })
                .collect::<Vec<_>>()
                .join("\n")
        }
        BlockContent::ApprovalRequest { action, reasoning, .. } => {
            let mut text = action_detail(action);
            if let Some(r) = reasoning {
                text.push('\n');
                text.push_str(r);
            }
            text
        }
        BlockContent::ToolCall { output, error, .. } => {
            if let Some(out) = output {
                out.clone()
            } else if let Some(e) = error {
                e.clone()
            } else {
                String::new()
            }
        }
        _ => String::new(),
    }
}

fn action_summary(action: &beyonder_core::AgentAction) -> String {
    match action {
        beyonder_core::AgentAction::FileWrite { path, .. } => {
            format!("Write {}", path.display())
        }
        beyonder_core::AgentAction::FileRead { path } => {
            format!("Read {}", path.display())
        }
        beyonder_core::AgentAction::FileDelete { path } => {
            format!("Delete {}", path.display())
        }
        beyonder_core::AgentAction::ShellExecute { command } => {
            format!("Run `{}`", command)
        }
        beyonder_core::AgentAction::NetworkRequest { url, method } => {
            format!("{} {}", method, url)
        }
        beyonder_core::AgentAction::AgentSpawn { agent_name } => {
            format!("Spawn agent `{}`", agent_name)
        }
        beyonder_core::AgentAction::ToolUse { tool_name } => {
            format!("Use tool `{}`", tool_name)
        }
    }
}

fn action_detail(action: &beyonder_core::AgentAction) -> String {
    // More verbose version for the approval block body
    action_summary(action)
}

/// Parse inline markdown on a single line and append colored spans.
/// Handles: **bold**, `inline code`, and normal text.
fn parse_inline(
    prefix: &str,
    line: &str,
    base: GlyphColor,
    bold: GlyphColor,
    code: GlyphColor,
    out: &mut Vec<(String, GlyphColor, bool)>,
) {
    // Empty line — just emit the prefix (carries the newline).
    if line.is_empty() {
        if !prefix.is_empty() {
            out.push((prefix.to_string(), base, false));
        }
        return;
    }
    let mut pfx = prefix;
    let mut rest = line;
    while !rest.is_empty() {
        if let Some(after) = rest.strip_prefix("**") {
            if let Some(end) = after.find("**") {
                out.push((format!("{}{}", pfx, &after[..end]), bold, true));
                pfx = "";
                rest = &after[end + 2..];
                continue;
            }
        }
        if let Some(after) = rest.strip_prefix('`') {
            if let Some(end) = after.find('`') {
                out.push((format!("{}{}", pfx, &after[..end]), code, false));
                pfx = "";
                rest = &after[end + 1..];
                continue;
            }
        }
        // Consume up to the next marker or end of line.
        let next = rest.find("**").unwrap_or(rest.len())
            .min(rest.find('`').unwrap_or(rest.len()));
        if next == 0 {
            // rest starts with an unclosed marker — treat it as literal text
            // and advance past it to prevent an infinite loop (common during
            // streaming when the response is cut off mid-**bold** or mid-`code`).
            let skip = if rest.starts_with("**") { 2 } else { 1 };
            let skip = skip.min(rest.len());
            out.push((format!("{}{}", pfx, &rest[..skip]), base, false));
            pfx = "";
            rest = &rest[skip..];
        } else {
            out.push((format!("{}{}", pfx, &rest[..next]), base, false));
            pfx = "";
            rest = &rest[next..];
        }
    }
}
