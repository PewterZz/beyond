//! Off-LAN transports for the /phone bridge.
//!
//! Default mode is direct LAN + mDNS (see `lib.rs`). These helpers let the
//! user surface the same WebSocket over a Tailscale tailnet or a public
//! ngrok tunnel without changing the protocol — only the pairing URL the
//! phone receives differs.

use anyhow::{anyhow, Context, Result};
use tokio::process::{Child, Command};

/// Returns the Tailscale MagicDNS hostname of this machine if `tailscale`
/// is installed and the daemon is logged in. Falls back to the first v4 IP.
pub fn detect_tailscale_host() -> Option<String> {
    let out = std::process::Command::new("tailscale")
        .args(["status", "--json"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).ok()?;
    if let Some(name) = v.get("Self").and_then(|s| s.get("DNSName")).and_then(|n| n.as_str()) {
        let trimmed = name.trim_end_matches('.');
        if !trimmed.is_empty() {
            return Some(trimmed.to_string());
        }
    }
    // Fallback: first IPv4 from `tailscale ip -4`.
    let out = std::process::Command::new("tailscale")
        .args(["ip", "-4"])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// A running `ngrok http` process. Killing the child tears the tunnel down.
pub struct NgrokTunnel {
    pub public_host: String,
    _child: Child,
}

impl NgrokTunnel {
    pub async fn start(port: u16) -> Result<Self> {
        // Require the binary up-front so we fail loudly.
        if std::process::Command::new("ngrok")
            .arg("version")
            .output()
            .is_err()
        {
            return Err(anyhow!(
                "ngrok not found — install from https://ngrok.com/download"
            ));
        }

        let mut cmd = Command::new("ngrok");
        cmd.args(["http", &port.to_string(), "--log=stdout", "--log-format=json"]);
        cmd.stdout(std::process::Stdio::null());
        cmd.stderr(std::process::Stdio::null());
        cmd.kill_on_drop(true);
        let child = cmd.spawn().context("spawn ngrok")?;

        // Poll ngrok's local API until the public URL is ready.
        let deadline = tokio::time::Instant::now() + tokio::time::Duration::from_secs(10);
        loop {
            if tokio::time::Instant::now() >= deadline {
                return Err(anyhow!("ngrok did not publish a tunnel URL within 10s"));
            }
            tokio::time::sleep(tokio::time::Duration::from_millis(250)).await;
            let Ok(resp) = reqwest::get("http://127.0.0.1:4040/api/tunnels").await else {
                continue;
            };
            let Ok(json) = resp.json::<serde_json::Value>().await else {
                continue;
            };
            if let Some(url) = json
                .get("tunnels")
                .and_then(|t| t.as_array())
                .and_then(|a| a.iter().find_map(|t| t.get("public_url")?.as_str()))
            {
                // `https://abc.ngrok.io` → `abc.ngrok.io`.
                let host = url
                    .trim_start_matches("https://")
                    .trim_start_matches("http://")
                    .to_string();
                return Ok(Self {
                    public_host: host,
                    _child: child,
                });
            }
        }
    }
}
