//! Beyonder remote bridge — exposes the block stream to a companion iOS app
//! over a persistent WebSocket, and accepts prompts/commands from the phone.
//!
//! Three transport modes, all sharing the same protocol:
//!   - **LAN** (default)       — mDNS on `_beyonder._tcp.local.`, WS over raw TCP.
//!   - **Tailscale**           — phone dials the host's MagicDNS name over WireGuard.
//!   - **ngrok**               — public `wss://…ngrok.io` tunnel; token auth only.
//!
//! All three reuse the same bound port; only the pairing URL the phone receives
//! (and the `tls` flag embedded in it) changes.

pub mod mdns;
pub mod pairing;
pub mod protocol;
pub mod server;
pub mod transport;

pub use pairing::PairingSecret;
pub use protocol::{ClientMsg, ContentPatch, Hello, ServerMsg};
pub use transport::{detect_tailscale_host, NgrokTunnel};

use anyhow::Result;
use std::net::SocketAddr;
use std::sync::atomic::Ordering;
use tokio::sync::broadcast;

pub struct RemoteHub {
    pub port: u16,
    outbound: broadcast::Sender<ServerMsg>,
    inbound_rx: std::sync::Arc<tokio::sync::Mutex<tokio::sync::mpsc::UnboundedReceiver<ClientMsg>>>,
    connected: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Current pairing URL reflecting the active endpoint (LAN / tailnet / ngrok).
    pub pairing_url: String,
    pub qr_ascii: String,
    pub endpoint_label: String,
    secret: PairingSecret,
    _mdns: Option<mdns::MdnsHandle>,
    ngrok: Option<NgrokTunnel>,
    _handle: server::ServerHandle,
}

impl RemoteHub {
    pub async fn start(
        session_id: String,
        active_model: String,
        active_provider: String,
    ) -> Result<Self> {
        let secret = PairingSecret::load_or_create()?;
        let bind: SocketAddr = "0.0.0.0:0".parse().unwrap();
        let handle = server::start(server::ServerConfig {
            bind,
            token: secret.token.clone(),
            session_id,
            active_model,
            active_provider,
        })
        .await?;

        let host = primary_host();
        let pairing_url = secret.pairing_url(&host, handle.port, false);
        let qr_ascii = secret
            .qr_ascii(&host, handle.port, false)
            .unwrap_or_default();

        let mdns = mdns::MdnsHandle::announce(handle.port, "beyonder").ok();
        if mdns.is_none() {
            tracing::warn!("remote: mDNS announcement failed; enter URL manually on phone");
        }

        Ok(Self {
            port: handle.port,
            outbound: handle.outbound.clone(),
            inbound_rx: handle.inbound_rx.clone(),
            connected: handle.connected.clone(),
            pairing_url,
            qr_ascii,
            endpoint_label: format!("lan ({host})"),
            secret,
            _mdns: mdns,
            ngrok: None,
            _handle: handle,
        })
    }

    pub fn is_connected(&self) -> bool {
        self.connected.load(Ordering::Relaxed)
    }

    /// Rewrite the pairing URL + QR for a new endpoint. The bound port and
    /// token stay the same; only what the phone dials changes.
    pub fn set_endpoint(&mut self, host: &str, port: u16, tls: bool, label: String) {
        self.pairing_url = self.secret.pairing_url(host, port, tls);
        self.qr_ascii = self
            .secret
            .qr_ascii(host, port, tls)
            .unwrap_or_default();
        self.endpoint_label = label;
    }

    /// Switch the advertised endpoint to the host's Tailscale MagicDNS name.
    /// Returns the host that was selected, or `None` if tailscale isn't set up.
    pub fn use_tailscale(&mut self) -> Option<String> {
        let host = transport::detect_tailscale_host()?;
        self.set_endpoint(&host, self.port, false, format!("tailscale ({host})"));
        Some(host)
    }

    /// Start an ngrok tunnel and repoint the pairing URL at its public host
    /// (`wss://<host>:443`). Dropping the hub kills the ngrok child.
    pub async fn use_ngrok(&mut self) -> Result<String> {
        let tunnel = NgrokTunnel::start(self.port).await?;
        let host = tunnel.public_host.clone();
        self.ngrok = Some(tunnel);
        self.set_endpoint(&host, 443, true, format!("ngrok ({host})"));
        Ok(host)
    }

    /// Broadcast an event to the paired phone (if any). Non-blocking.
    pub fn send(&self, msg: ServerMsg) -> bool {
        self.outbound.send(msg).is_ok()
    }

    /// Drain any queued inbound messages without blocking.
    pub fn poll_inbound(&self, out: &mut Vec<ClientMsg>) {
        if let Ok(mut rx) = self.inbound_rx.try_lock() {
            while let Ok(msg) = rx.try_recv() {
                out.push(msg);
            }
        }
    }
}

fn primary_host() -> String {
    if let Ok(out) = std::process::Command::new("hostname").output() {
        if out.status.success() {
            let h = String::from_utf8_lossy(&out.stdout).trim().to_string();
            if !h.is_empty() {
                if h.ends_with(".local") {
                    return h;
                }
                return format!("{h}.local");
            }
        }
    }
    "beyonder.local".into()
}
