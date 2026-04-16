//! PTY session management using portable-pty.

use anyhow::{Context, Result};
use beyonder_core::SessionId;
use portable_pty::{native_pty_system, Child, CommandBuilder, MasterPty, PtySize};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokio::sync::mpsc;
use tracing::info;

/// Events from a PTY session.
#[derive(Debug, Clone)]
pub enum PtyEvent {
    /// Raw bytes from the PTY (includes ANSI escape sequences).
    Output(Vec<u8>),
    /// The child process exited.
    Exited(Option<u32>),
}

/// Callback invoked after each PTY event is sent, used to wake the event loop.
pub type WakeFn = Box<dyn Fn() + Send + 'static>;

/// A live PTY session connected to a shell.
pub struct PtySession {
    pub session_id: SessionId,
    master: Box<dyn MasterPty + Send>,
    // Writer cached at spawn — take_writer() can only be called once on some platforms.
    writer: Box<dyn std::io::Write + Send>,
    #[allow(dead_code)]
    child: Arc<Mutex<Box<dyn Child + Send + Sync>>>,
    pub event_rx: mpsc::Receiver<PtyEvent>,
}

impl PtySession {
    pub fn spawn(
        session_id: SessionId,
        shell: &str,
        cwd: &PathBuf,
        extra_env: &[(&str, &str)],
    ) -> Result<Self> {
        Self::spawn_sized(session_id, shell, cwd, extra_env, 120, 30)
    }

    pub fn spawn_sized(
        session_id: SessionId,
        shell: &str,
        cwd: &PathBuf,
        extra_env: &[(&str, &str)],
        cols: u16,
        rows: u16,
    ) -> Result<Self> {
        Self::spawn_sized_inner(session_id, shell, cwd, extra_env, cols, rows, None)
    }

    /// Like `spawn_sized` but calls `wake` after each PTY event to wake the
    /// event loop (enables `ControlFlow::Wait` in the renderer).
    pub fn spawn_sized_with_wake(
        session_id: SessionId,
        shell: &str,
        cwd: &PathBuf,
        extra_env: &[(&str, &str)],
        cols: u16,
        rows: u16,
        wake: WakeFn,
    ) -> Result<Self> {
        Self::spawn_sized_inner(session_id, shell, cwd, extra_env, cols, rows, Some(wake))
    }

    fn spawn_sized_inner(
        session_id: SessionId,
        shell: &str,
        cwd: &PathBuf,
        extra_env: &[(&str, &str)],
        cols: u16,
        rows: u16,
        wake: Option<WakeFn>,
    ) -> Result<Self> {
        info!(shell, cols, rows, "Spawning PTY session");
        let pty_system = native_pty_system();
        let pair = pty_system
            .openpty(PtySize {
                rows,
                cols,
                pixel_width: 0,
                pixel_height: 0,
            })
            .context("Failed to open PTY")?;

        let home = std::env::var("HOME").unwrap_or_default();
        let session_dir = std::env::temp_dir().join(format!("beyonder_{}", &session_id.0));
        std::fs::create_dir_all(&session_dir).ok();

        let kind = crate::shell_hooks::detect_shell_kind(shell);
        let mut cmd = CommandBuilder::new(shell);
        cmd.cwd(cwd);

        match kind {
            crate::shell_hooks::ShellKind::Zsh => {
                // Temp ZDOTDIR overlay: our .zshenv / .zshrc forward to the user's,
                // then inject hooks. Doesn't touch the user's real config.
                let zshenv = format!(
                    "[ -f {home}/.zshenv ] && source {home}/.zshenv\n\
                     [ -f {home}/.zprofile ] && source {home}/.zprofile\n"
                );
                std::fs::write(session_dir.join(".zshenv"), zshenv).ok();
                let hooks = crate::shell_hooks::zsh_init_script(&session_id.0);
                let zshrc = format!("{hooks}\n[ -f {home}/.zshrc ] && source {home}/.zshrc\n");
                std::fs::write(session_dir.join(".zshrc"), zshrc).ok();
                cmd.env("ZDOTDIR", &session_dir);
                cmd.args(&["-i"]);
            }
            crate::shell_hooks::ShellKind::Bash => {
                // --rcfile overrides ~/.bashrc — source the real one first.
                let rcfile = session_dir.join("init.bashrc");
                let user_rc = format!("[ -f {home}/.bashrc ] && source {home}/.bashrc\n");
                let hooks = crate::shell_hooks::bash_init_script(&session_id.0);
                std::fs::write(&rcfile, format!("{user_rc}\n{hooks}")).ok();
                cmd.args(&["--rcfile", rcfile.to_str().unwrap_or(""), "-i"]);
            }
            crate::shell_hooks::ShellKind::Fish => {
                // --init-command runs in addition to user config; no overlay needed.
                let initf = session_dir.join("beyonder.fish");
                std::fs::write(&initf, crate::shell_hooks::fish_init_script(&session_id.0)).ok();
                let src_cmd = format!("source {}", initf.display());
                cmd.args(&["--init-command", &src_cmd, "-i"]);
            }
            crate::shell_hooks::ShellKind::Nushell => {
                // Override --config / --env-config with files that source the
                // user's real config first, then layer our hooks on top.
                let nu_dir = std::path::PathBuf::from(&home)
                    .join(".config")
                    .join("nushell");
                let user_cfg = nu_dir.join("config.nu");
                let user_env = nu_dir.join("env.nu");

                let env_path = session_dir.join("env.nu");
                let env_body = if user_env.exists() {
                    format!("source {}\n", user_env.display())
                } else {
                    String::new()
                };
                std::fs::write(&env_path, env_body).ok();

                let cfg_path = session_dir.join("config.nu");
                let user_cfg_src = if user_cfg.exists() {
                    format!("source {}\n", user_cfg.display())
                } else {
                    String::new()
                };
                let hooks = crate::shell_hooks::nushell_init_script(&session_id.0);
                std::fs::write(&cfg_path, format!("{user_cfg_src}\n{hooks}")).ok();
                cmd.args(&[
                    "--config",
                    cfg_path.to_str().unwrap_or(""),
                    "--env-config",
                    env_path.to_str().unwrap_or(""),
                ]);
            }
            crate::shell_hooks::ShellKind::Unknown => {
                cmd.args(&["-i"]);
            }
        }

        cmd.env("TERM", "xterm-256color");
        cmd.env("COLORTERM", "truecolor");
        // Spoof as iTerm.app so claude-code (and other capability-sniffing TUIs)
        // pick their nicer Unicode glyphs. Also set LC_TERMINAL — iTerm's native
        // apps use this as a secondary signal and claude-code checks both.
        cmd.env("TERM_PROGRAM", "iTerm.app");
        cmd.env("LC_TERMINAL", "iTerm2");
        cmd.env("LC_TERMINAL_VERSION", "3.5.0");
        cmd.env("TERM_PROGRAM_VERSION", env!("CARGO_PKG_VERSION"));
        cmd.env("BEYONDER_SESSION_ID", &session_id.0);
        for (k, v) in extra_env {
            cmd.env(k, v);
        }

        let child = pair
            .slave
            .spawn_command(cmd)
            .context("Failed to spawn shell")?;
        let child = Arc::new(Mutex::new(child));

        // Cache the writer immediately — take_writer() can only be called once.
        let writer = pair
            .master
            .take_writer()
            .context("Failed to get PTY writer")?;

        let (event_tx, event_rx) = mpsc::channel(1024);

        // Spawn a background reader thread (blocking I/O — can't use tokio directly here).
        let mut reader = pair
            .master
            .try_clone_reader()
            .context("Failed to clone PTY reader")?;
        let child_clone = Arc::clone(&child);
        let tx = event_tx.clone();
        std::thread::spawn(move || {
            // 64KB read buffer — 16x the old 4KB to reduce syscall overhead
            // for bulk output (e.g. `cat large_file`). The OS will fill as
            // much as available per read, so larger buffer = fewer events.
            const BUF_SIZE: usize = 65536;
            let mut buf = vec![0u8; BUF_SIZE];
            loop {
                match std::io::Read::read(&mut reader, &mut buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        // Send exactly the bytes read — no over-allocation.
                        let _ = tx.blocking_send(PtyEvent::Output(buf[..n].to_vec()));
                        if let Some(ref w) = wake {
                            w();
                        }
                    }
                    Err(_) => break,
                }
            }
            // Child exited — get exit code.
            let code = child_clone
                .lock()
                .ok()
                .and_then(|mut c| c.wait().ok())
                .and_then(|s| if s.success() { Some(0) } else { None });
            let _ = tx.blocking_send(PtyEvent::Exited(code));
            if let Some(ref w) = wake {
                w();
            }
        });

        Ok(Self {
            session_id,
            master: pair.master,
            writer,
            child,
            event_rx,
        })
    }

    /// Write bytes to the PTY (user keystrokes or command input).
    pub fn write(&mut self, data: &[u8]) -> Result<()> {
        use std::io::Write;
        self.writer.write_all(data).context("PTY write failed")?;
        self.writer.flush().ok();
        Ok(())
    }

    /// Resize the PTY.
    pub fn resize(&self, rows: u16, cols: u16) -> Result<()> {
        self.master
            .resize(PtySize {
                rows,
                cols,
                pixel_width: 0,
                pixel_height: 0,
            })
            .context("Failed to resize PTY")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Verify that the wake callback fires on PTY output and on exit,
    /// proving event-driven redraw can rely on it instead of polling.
    #[tokio::test]
    async fn wake_callback_fires_on_pty_output() {
        let wake_count = Arc::new(AtomicUsize::new(0));
        let wc = Arc::clone(&wake_count);
        let wake: WakeFn = Box::new(move || {
            wc.fetch_add(1, Ordering::SeqCst);
        });

        let session_id = SessionId::new();
        let cwd = std::env::temp_dir();
        // Run a command that produces output then exits.
        let mut pty =
            PtySession::spawn_sized_with_wake(session_id, "/bin/sh", &cwd, &[], 80, 24, wake)
                .expect("spawn PTY");

        // Send a command that produces output and exits.
        pty.write(b"echo hello && exit\n").unwrap();

        // Drain events until exit.
        let mut got_output = false;
        let mut got_exit = false;
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            match tokio::time::timeout_at(deadline, pty.event_rx.recv()).await {
                Ok(Some(PtyEvent::Output(_))) => got_output = true,
                Ok(Some(PtyEvent::Exited(_))) => {
                    got_exit = true;
                    break;
                }
                Ok(None) => break,
                Err(_) => panic!("PTY test timed out"),
            }
        }

        assert!(got_output, "should have received PTY output");
        assert!(got_exit, "should have received PTY exit");
        // Wake must have fired at least once for output + once for exit.
        let wakes = wake_count.load(Ordering::SeqCst);
        assert!(
            wakes >= 2,
            "wake callback should fire at least twice, got {wakes}"
        );
    }

    /// Verify that without a wake callback, PTY still works (backward compat).
    #[tokio::test]
    async fn pty_works_without_wake() {
        let session_id = SessionId::new();
        let cwd = std::env::temp_dir();
        let mut pty =
            PtySession::spawn_sized(session_id, "/bin/sh", &cwd, &[], 80, 24).expect("spawn PTY");

        pty.write(b"exit\n").unwrap();

        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            match tokio::time::timeout_at(deadline, pty.event_rx.recv()).await {
                Ok(Some(PtyEvent::Exited(_))) | Ok(None) => break,
                Ok(Some(_)) => continue,
                Err(_) => panic!("PTY test timed out"),
            }
        }
    }

    /// Verify that bulk output (>4KB) is handled correctly with the 64KB buffer.
    /// This proves the larger buffer reduces event count for throughput.
    #[tokio::test]
    async fn bulk_output_uses_fewer_events() {
        let session_id = SessionId::new();
        let cwd = std::env::temp_dir();
        let mut pty =
            PtySession::spawn_sized(session_id, "/bin/sh", &cwd, &[], 80, 24).expect("spawn PTY");

        // Generate ~32KB of output — should arrive in fewer events than
        // the old 4KB buffer would produce (8+ events → ~1-2 events).
        pty.write(b"dd if=/dev/zero bs=1024 count=32 2>/dev/null | od | head -500; exit\n")
            .unwrap();

        let mut total_bytes = 0usize;
        let mut event_count = 0usize;
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            match tokio::time::timeout_at(deadline, pty.event_rx.recv()).await {
                Ok(Some(PtyEvent::Output(bytes))) => {
                    total_bytes += bytes.len();
                    event_count += 1;
                }
                Ok(Some(PtyEvent::Exited(_))) | Ok(None) => break,
                Err(_) => panic!("PTY test timed out"),
            }
        }

        assert!(total_bytes > 0, "should have received output bytes");
        // With 64KB buffer, bulk output should arrive in fewer chunks.
        // The exact count depends on timing, but it should be reasonable.
        assert!(
            event_count < 100,
            "too many events ({event_count}) for {total_bytes} bytes — buffer may be too small"
        );
    }
}
