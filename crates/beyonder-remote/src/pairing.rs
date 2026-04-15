//! Pairing token + QR payload.
//!
//! First run generates a 32-byte random token, persists it at
//! `$XDG_CONFIG_HOME/beyond/remote.token` (0600), and emits a QR whose
//! payload is `beyonder://<host>:<port>?tok=<base64url>`. The iPhone app
//! scans once, stores the URL in Keychain, and reuses it forever.

use anyhow::{Context, Result};
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use rand::RngCore;
use std::path::PathBuf;

pub struct PairingSecret {
    pub token: String,
}

impl PairingSecret {
    pub fn load_or_create() -> Result<Self> {
        let path = token_path()?;
        if let Ok(bytes) = std::fs::read(&path) {
            let token = String::from_utf8(bytes)
                .context("remote token file not utf-8")?
                .trim()
                .to_string();
            if !token.is_empty() {
                return Ok(Self { token });
            }
        }
        let mut buf = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut buf);
        let token = URL_SAFE_NO_PAD.encode(buf);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        std::fs::write(&path, &token).context("writing remote token")?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
        }
        Ok(Self { token })
    }

    /// URL form suitable for QR encoding.
    /// `tls=1` switches the phone to `wss://` (used when ngrok fronts the socket).
    pub fn pairing_url(&self, host: &str, port: u16, tls: bool) -> String {
        let t = if tls { 1 } else { 0 };
        format!("beyonder://{host}:{port}?tok={}&tls={t}", self.token)
    }

    /// Render the pairing URL as an ASCII-art QR (dense, fits ~40x40 cells)
    /// suitable for display as a single text block.
    pub fn qr_ascii(&self, host: &str, port: u16, tls: bool) -> Result<String> {
        use qrcode::{EcLevel, QrCode};
        let url = self.pairing_url(host, port, tls);
        let code = QrCode::with_error_correction_level(url.as_bytes(), EcLevel::L)
            .context("qr encode failed")?;
        // Render half-height so each row = 2 modules, halving terminal line count.
        let string = code
            .render::<char>()
            .quiet_zone(true)
            .module_dimensions(1, 1)
            .dark_color('█')
            .light_color(' ')
            .build();
        Ok(string)
    }
}

fn token_path() -> Result<PathBuf> {
    let base = if let Ok(x) = std::env::var("XDG_CONFIG_HOME") {
        PathBuf::from(x)
    } else {
        let home = std::env::var("HOME").context("HOME not set")?;
        PathBuf::from(home).join(".config")
    };
    Ok(base.join("beyond").join("remote.token"))
}
