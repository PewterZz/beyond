use beyonder_core::ApprovalMode;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Which LLM backend to use and its connection parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ProviderConfig {
    /// Ollama — local (localhost:11434) or cloud (ollama.com Turbo/Pro).
    Ollama {
        #[serde(default = "default_ollama_base_url")]
        base_url: String,
        /// Env var holding the bearer token for cloud. None = local.
        #[serde(default)]
        api_key_env: Option<String>,
    },
    /// llama.cpp llama-server with `--jinja` and an OpenAI-compat `/v1` endpoint.
    LlamaCpp {
        #[serde(default = "default_local_v1_url")]
        base_url: String,
        /// Optional env var for auth if the server is fronted by a reverse proxy.
        #[serde(default)]
        api_key_env: Option<String>,
    },
    /// Apple MLX mlx_lm.server (mlx-lm >= 0.19 recommended).
    Mlx {
        #[serde(default = "default_local_v1_url")]
        base_url: String,
        #[serde(default)]
        api_key_env: Option<String>,
    },
}

fn default_ollama_base_url() -> String {
    "http://localhost:11434".to_string()
}
fn default_local_v1_url() -> String {
    "http://127.0.0.1:8080/v1".to_string()
}

impl Default for ProviderConfig {
    fn default() -> Self {
        // Respect OLLAMA_API_KEY at default-construction time so the cloud
        // backend activates automatically when the env var is present.
        if std::env::var("OLLAMA_API_KEY")
            .map(|v| !v.is_empty())
            .unwrap_or(false)
        {
            ProviderConfig::Ollama {
                base_url: "https://ollama.com".to_string(),
                api_key_env: Some("OLLAMA_API_KEY".to_string()),
            }
        } else {
            ProviderConfig::Ollama {
                base_url: default_ollama_base_url(),
                api_key_env: None,
            }
        }
    }
}

impl ProviderConfig {
    /// Short lowercase name used as the agent name and for display.
    pub fn name(&self) -> &'static str {
        match self {
            ProviderConfig::Ollama { .. } => "ollama",
            ProviderConfig::LlamaCpp { .. } => "llama_cpp",
            ProviderConfig::Mlx { .. } => "mlx",
        }
    }

    /// Construct a ProviderConfig from a short name string, using sensible
    /// defaults. Used by the `/provider` runtime command.
    pub fn from_name(name: &str) -> Self {
        match name {
            "llama_cpp" => ProviderConfig::LlamaCpp {
                base_url: default_local_v1_url(),
                api_key_env: None,
            },
            "mlx" => ProviderConfig::Mlx {
                base_url: default_local_v1_url(),
                api_key_env: None,
            },
            _ => ProviderConfig::default(),
        }
    }
}

/// Top-level Beyonder configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeyonderConfig {
    /// Name of the active theme — resolved against the built-in palette list.
    /// Unknown names fall back to `mocha`.
    #[serde(default = "default_theme_name")]
    pub theme: String,
    pub font: FontConfig,
    pub shell: ShellConfig,
    pub data_dir: PathBuf,
    #[serde(default = "default_model")]
    pub model: String,
    #[serde(default)]
    pub provider: ProviderConfig,
    /// Policy governing whether agent tool calls need user approval.
    #[serde(default)]
    pub approval_mode: ApprovalMode,
}

fn default_theme_name() -> String {
    "mocha".to_string()
}

fn default_model() -> String {
    "qwen2.5-coder:7b".to_string()
}

impl Default for BeyonderConfig {
    fn default() -> Self {
        Self {
            theme: default_theme_name(),
            font: FontConfig::default(),
            shell: ShellConfig::default(),
            data_dir: default_data_dir(),
            model: default_model(),
            provider: ProviderConfig::default(),
            approval_mode: ApprovalMode::default(),
        }
    }
}

impl BeyonderConfig {
    pub fn load_or_default() -> Self {
        let path = config_path();
        if path.exists() {
            match std::fs::read_to_string(&path) {
                Ok(s) => toml::from_str(&s).unwrap_or_default(),
                Err(_) => Self::default(),
            }
        } else {
            Self::default()
        }
    }

    /// Resolve the named theme to a concrete `Theme` palette.
    pub fn resolved_theme(&self) -> crate::theme::Theme {
        crate::theme::theme_by_name(&self.theme)
    }

    pub fn db_path(&self) -> PathBuf {
        self.data_dir.join("beyonder.db")
    }

    pub fn save(&self) -> std::io::Result<()> {
        let path = config_path();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let toml_str = toml::to_string_pretty(self).map_err(std::io::Error::other)?;
        std::fs::write(&path, toml_str)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FontConfig {
    pub family: String,
    pub size: f32,
}

impl Default for FontConfig {
    fn default() -> Self {
        Self {
            family: "monospace".to_string(),
            size: 14.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ShellConfig {
    pub program: Option<String>,
}

/// Base directory for all Beyonder state. Pinned to `$HOME/.config/beyond` on
/// every platform (rather than macOS's `~/Library/Application Support`) so users
/// have a single, predictable place for configs, the SQLite store, logs, etc.
/// Honours `$XDG_CONFIG_HOME` if set.
pub fn beyonder_dir() -> PathBuf {
    if let Ok(xdg) = std::env::var("XDG_CONFIG_HOME") {
        if !xdg.is_empty() {
            return PathBuf::from(xdg).join("beyond");
        }
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".config").join("beyond")
}

pub fn config_path() -> PathBuf {
    beyonder_dir().join("config.toml")
}

fn default_data_dir() -> PathBuf {
    beyonder_dir().join("data")
}
