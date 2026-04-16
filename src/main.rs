use anyhow::Result;
use beyonder_config::BeyonderConfig;
use beyonder_ui::App;
use std::sync::Arc;
use tracing::info;
use tracing_subscriber::EnvFilter;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::window::{Icon, Window, WindowAttributes, WindowId};

/// Embedded app icon (256x256 PNG).
static ICON_PNG: &[u8] = include_bytes!("../assets/beyond-macos.png");

fn main() -> Result<()> {
    // Use RUST_LOG / BEYONDER_LOG as-is when set; fall back to sensible defaults.
    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("beyonder=info,wgpu_core=warn,wgpu_hal=warn"));
    // Write to stderr — stderr is unbuffered, so logs flush immediately even
    // when output is redirected to a file. stdout is fully buffered when not a
    // tty, which causes logs to disappear on a hang.
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(filter)
        .init();

    info!("Beyond starting");
    let config = BeyonderConfig::load_or_default();

    let event_loop = EventLoop::<()>::with_user_event().build()?;
    let proxy = event_loop.create_proxy();

    let mut handler = BeyonderHandler::new(config, proxy);
    event_loop.run_app(&mut handler)?;
    Ok(())
}

struct BeyonderHandler {
    config: BeyonderConfig,
    app: Option<App>,
    window: Option<Arc<Window>>,
    rt: tokio::runtime::Runtime,
    proxy: EventLoopProxy<()>,
    /// True when any event source has produced work that requires a redraw.
    needs_redraw: bool,
}

impl BeyonderHandler {
    fn new(config: BeyonderConfig, proxy: EventLoopProxy<()>) -> Self {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("Failed to build tokio runtime");
        Self {
            config,
            app: None,
            window: None,
            rt,
            proxy,
            needs_redraw: true,
        }
    }
}

impl ApplicationHandler<()> for BeyonderHandler {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.app.is_some() {
            return;
        }

        let mut window_attrs = WindowAttributes::default()
            .with_title("Beyond")
            .with_inner_size(winit::dpi::LogicalSize::new(1280u32, 800u32))
            .with_resizable(true);

        if let Some(icon) = load_winit_icon() {
            window_attrs = window_attrs.with_window_icon(Some(icon));
        }

        let window = event_loop
            .create_window(window_attrs)
            .expect("Failed to create window");
        let window = Arc::new(window);

        let config = self.config.clone();
        let proxy = self.proxy.clone();
        let app = self
            .rt
            .block_on(App::new(Arc::clone(&window), config, proxy))
            .expect("Failed to init Beyond app");

        // Set macOS Dock icon after NSApplication is initialized by winit.
        #[cfg(target_os = "macos")]
        set_macos_dock_icon();

        info!("Beyond initialized — window open");
        self.window = Some(Arc::clone(&window));
        self.app = Some(app);
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, _event: ()) {
        // An async event source (PTY, supervisor, broker, remote) woke us up.
        self.needs_redraw = true;
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        // Any window event that isn't a pure redraw request implies state change.
        if !matches!(event, WindowEvent::RedrawRequested) {
            self.needs_redraw = true;
        }

        let Some(app) = self.app.as_mut() else { return };

        let should_close = self.rt.block_on(app.handle_window_event(&event));
        if should_close || app.should_quit {
            event_loop.exit();
            return;
        }

        if matches!(event, WindowEvent::RedrawRequested) {
            if let Err(e) = app.render() {
                tracing::error!("Render error: {e}");
            }
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // Tick app state (drain supervisor/broker events) here, not inside
        // window_event(RedrawRequested), so it runs even when the window is
        // occluded or minimised (macOS suppresses RedrawRequested for hidden
        // windows, which would freeze streaming agent output).
        if let Some(app) = self.app.as_mut() {
            let had_work = self.rt.block_on(app.tick());
            if had_work {
                self.needs_redraw = true;
            }
        }

        // When a spinner is animating (running agent/command block) we need
        // continuous redraws even if tick() had no work this iteration.
        let animating = self.app.as_ref().map_or(false, |app| app.needs_animation());
        if animating {
            self.needs_redraw = true;
        }

        if self.needs_redraw {
            self.needs_redraw = false;
            if let Some(window) = &self.window {
                window.request_redraw();
            }
            // wgpu PresentMode::Fifo provides VSync-locked frame pacing, so we
            // use Poll (re-enter immediately) and let render()'s
            // get_current_texture() block until the next display refresh. This
            // gives us exact VSync alignment with zero manual timer overhead.
            event_loop.set_control_flow(ControlFlow::Poll);
        } else {
            // No animation on the active tab. Use a short poll interval when
            // any agent/tool work is in-flight (active or stashed tabs) to
            // catch events promptly — macOS doesn't always deliver
            // EventLoopProxy wakes from background threads reliably. When
            // truly idle, use a longer interval.
            let has_work = self
                .app
                .as_ref()
                .map_or(false, |app| app.has_pending_async_work());
            let timeout = if has_work { 16 } else { 500 };
            event_loop.set_control_flow(ControlFlow::WaitUntil(
                std::time::Instant::now() + std::time::Duration::from_millis(timeout),
            ));
        }
    }
}

/// Load the embedded icon as a winit Icon (for Linux/Windows taskbar).
fn load_winit_icon() -> Option<Icon> {
    let img = image::load_from_memory_with_format(ICON_PNG, image::ImageFormat::Png).ok()?;
    let rgba = img.into_rgba8();
    let (w, h) = (rgba.width(), rgba.height());
    Icon::from_rgba(rgba.into_raw(), w, h).ok()
}

/// Set the macOS Dock icon programmatically via NSApplication.
/// `with_window_icon` only affects Linux/Windows; macOS ignores it for
/// non-bundled apps and shows the default Terminal icon instead.
#[cfg(target_os = "macos")]
fn set_macos_dock_icon() {
    use objc2::rc::Retained;
    use objc2::runtime::AnyObject;
    use objc2::{class, msg_send, msg_send_id};
    use std::ffi::c_void;

    unsafe {
        let ptr = ICON_PNG.as_ptr() as *const c_void;
        let len = ICON_PNG.len();
        let data: Retained<AnyObject> = msg_send_id![class!(NSData), dataWithBytes:ptr length:len];

        let image: Retained<AnyObject> =
            msg_send_id![msg_send_id![class!(NSImage), alloc], initWithData:&*data];

        let app: Retained<AnyObject> = msg_send_id![class!(NSApplication), sharedApplication];
        let () = msg_send![&*app, setApplicationIconImage:&*image];
    }
}
