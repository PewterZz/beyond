//! mDNS / Bonjour advertisement so the iPhone app discovers the host
//! without the user typing an IP. Service type: `_beyonder._tcp.local.`

use anyhow::{Context, Result};
use mdns_sd::{ServiceDaemon, ServiceInfo};

pub struct MdnsHandle {
    daemon: ServiceDaemon,
    fullname: String,
}

impl MdnsHandle {
    pub fn announce(port: u16, instance: &str) -> Result<Self> {
        let daemon = ServiceDaemon::new().context("mdns daemon")?;
        let host = hostname_fqdn();
        let ips: Vec<std::net::IpAddr> = local_ips();
        let service = ServiceInfo::new(
            "_beyonder._tcp.local.",
            instance,
            &host,
            &ips[..],
            port,
            None,
        )
        .context("mdns service info")?;
        let fullname = service.get_fullname().to_string();
        daemon.register(service).context("mdns register")?;
        Ok(Self { daemon, fullname })
    }
}

impl Drop for MdnsHandle {
    fn drop(&mut self) {
        let _ = self.daemon.unregister(&self.fullname);
        let _ = self.daemon.shutdown();
    }
}

fn hostname_fqdn() -> String {
    let base = std::process::Command::new("hostname")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "beyonder".into());
    if base.ends_with(".local") || base.ends_with(".local.") {
        if base.ends_with('.') {
            base
        } else {
            format!("{base}.")
        }
    } else {
        format!("{base}.local.")
    }
}

/// Non-loopback IPv4 addresses (mDNS works best with plain v4 on LAN).
fn local_ips() -> Vec<std::net::IpAddr> {
    // Parse `ifconfig` / `ip addr`; falls back to empty (daemon will still work
    // by answering with its own detected addrs).
    let mut out = vec![];
    if let Ok(output) = std::process::Command::new("hostname").arg("-I").output() {
        if output.status.success() {
            for tok in String::from_utf8_lossy(&output.stdout).split_whitespace() {
                if let Ok(ip) = tok.parse() {
                    out.push(ip);
                }
            }
        }
    }
    if out.is_empty() {
        // macOS: no `-I`; grab from `ifconfig`.
        if let Ok(output) = std::process::Command::new("ifconfig").output() {
            for line in String::from_utf8_lossy(&output.stdout).lines() {
                let line = line.trim();
                if let Some(rest) = line.strip_prefix("inet ") {
                    if let Some(ip_str) = rest.split_whitespace().next() {
                        if let Ok(ip) = ip_str.parse::<std::net::Ipv4Addr>() {
                            if !ip.is_loopback() && !ip.is_link_local() {
                                out.push(ip.into());
                            }
                        }
                    }
                }
            }
        }
    }
    out
}
