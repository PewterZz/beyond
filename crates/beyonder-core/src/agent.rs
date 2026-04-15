use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use ulid::Ulid;

use crate::capability::CapabilitySet;

/// Unique agent identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentId(pub String);

impl AgentId {
    pub fn new() -> Self {
        Self(Ulid::new().to_string())
    }

    pub fn named(name: &str) -> Self {
        Self(format!("{}-{}", name, Ulid::new()))
    }
}

impl Default for AgentId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for AgentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Metadata and state for an agent process.
/// Agents are first-class citizens in Beyonder — they have lifecycle,
/// resource limits, and capability sets, analogous to OS processes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentInfo {
    pub id: AgentId,
    pub name: String,
    pub kind: AgentKind,
    pub state: AgentState,
    pub capabilities: CapabilitySet,
    pub resource_limits: ResourceLimits,
    pub metrics: AgentMetrics,
    pub spawned_at: DateTime<Utc>,
}

impl AgentInfo {
    pub fn new(name: impl Into<String>, kind: AgentKind) -> Self {
        Self {
            id: AgentId::new(),
            name: name.into(),
            kind,
            state: AgentState::Spawning,
            capabilities: CapabilitySet::default(),
            resource_limits: ResourceLimits::default(),
            metrics: AgentMetrics::default(),
            spawned_at: Utc::now(),
        }
    }
}

/// How the agent runs.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentKind {
    /// An ACP-compatible process (Claude Code, Gemini CLI, etc.)
    /// Communicates via JSON-RPC over stdio.
    AcpProcess { binary: String, args: Vec<String> },
    /// LLM backend driven by Ollama — same API for local (localhost:11434)
    /// and cloud (ollama.com Turbo/Pro). Only differs in base_url + auth.
    Ollama {
        base_url: String,
        model: String,
        /// Env var that holds the bearer token (e.g. "OLLAMA_API_KEY"). None for local.
        api_key_env: Option<String>,
    },
    /// llama.cpp llama-server with OpenAI-compatible /v1/chat/completions endpoint.
    /// Requires the server to be started with `--jinja` for reliable tool calling.
    LlamaCpp {
        base_url: String,
        model: String,
        /// Optional env var for auth (e.g. when fronted by a reverse proxy).
        api_key_env: Option<String>,
    },
    /// Apple MLX mlx_lm.server with OpenAI-compatible /v1/chat/completions endpoint.
    /// Requires mlx-lm >= 0.19 for tool calling support.
    Mlx {
        base_url: String,
        model: String,
        api_key_env: Option<String>,
    },
    /// A WASM component plugin (post-MVP).
    WasmPlugin { module_path: String },
    /// Built-in agent (system-level operations).
    BuiltIn,
}

/// Agent lifecycle state — mirrors OS process states.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentState {
    Spawning,
    Ready,
    Busy { current_action: String },
    Suspended,
    Dead { reason: DeathReason },
}

impl AgentState {
    pub fn is_alive(&self) -> bool {
        !matches!(self, AgentState::Dead { .. })
    }

    pub fn is_available(&self) -> bool {
        matches!(self, AgentState::Ready)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeathReason {
    Completed,
    Killed,
    Crashed { exit_code: Option<i32> },
    ResourceLimitExceeded,
    Timeout,
}

/// Resource limits enforced by the agent supervisor.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub max_memory_bytes: Option<u64>,
    pub max_cpu_time_secs: Option<u64>,
    pub max_file_writes: Option<u32>,
    pub max_network_calls: Option<u32>,
    pub max_tokens: Option<u64>,
    pub timeout_secs: Option<u64>,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self {
            max_memory_bytes: Some(512 * 1024 * 1024), // 512 MB
            max_cpu_time_secs: Some(300),               // 5 minutes
            max_file_writes: None,
            max_network_calls: None,
            max_tokens: Some(100_000),
            timeout_secs: Some(600), // 10 minutes
        }
    }
}

/// Runtime metrics tracked per agent.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AgentMetrics {
    pub tokens_used: u64,
    pub actions_taken: u32,
    pub file_writes: u32,
    pub network_calls: u32,
    pub approvals_requested: u32,
    pub approvals_granted: u32,
    pub approvals_denied: u32,
}
