//! Agent supervisor — spawns, monitors, and kills agent processes.
//! Agents are first-class processes with lifecycle management.

use anyhow::Result;
use beyonder_core::{AgentId, AgentInfo, AgentKind, AgentState, CapabilitySet, DeathReason, SessionId};
use beyonder_acp::AcpClient;
use beyonder_acp::client::{AgentEvent, StreamPause};
use std::collections::HashMap;
use std::path::PathBuf;
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use crate::provider::{
    AgentBackend,
    ollama::{OllamaBackend, OllamaConfig, ToolDescriptor},
    openai_compat::{OpenAICompatBackend, OpenAICompatConfig},
};
use crate::tools::registry::ToolRegistry;
use crate::tools::{ToolContext, ToolOutput};

/// Events from the supervisor.
#[derive(Debug)]
pub enum SupervisorEvent {
    AgentSpawned(AgentInfo),
    AgentReady(AgentId),
    AgentEvent { agent_id: AgentId, event: AgentEvent },
    AgentDied { agent_id: AgentId, reason: DeathReason },
}

/// Commands sent to an agent's background turn-driver task.
enum AgentCmd {
    Prompt(String),
    ResetConversation,
}

pub struct AgentHandle {
    pub info: AgentInfo,
    /// Send commands to the agent's background turn-driver task.
    cmd_tx: mpsc::UnboundedSender<AgentCmd>,
}

/// Manages the lifecycle of all agent processes.
pub struct AgentSupervisor {
    agents: HashMap<AgentId, AgentHandle>,
    event_tx: mpsc::UnboundedSender<SupervisorEvent>,
}

impl AgentSupervisor {
    pub fn new(event_tx: mpsc::UnboundedSender<SupervisorEvent>) -> Self {
        Self { agents: HashMap::new(), event_tx }
    }

    /// Clone the supervisor event sender — used by callers that need to inject events.
    pub fn event_tx(&self) -> mpsc::UnboundedSender<SupervisorEvent> {
        self.event_tx.clone()
    }

    /// Spawn a new agent process (ACP subprocess or Ollama backend).
    pub async fn spawn_agent(
        &mut self,
        name: impl Into<String>,
        kind: AgentKind,
        capabilities: CapabilitySet,
    ) -> Result<AgentId> {
        let name = name.into();
        let mut info = AgentInfo::new(&name, kind.clone());
        info.capabilities = capabilities.clone();
        let agent_id = info.id.clone();

        // Extract working directory from the first Directory-scoped capability.
        let cwd = capabilities.capabilities.iter()
            .find_map(|c| {
                if let beyonder_core::CapabilityScope::Directory(p) = &c.scope {
                    Some(p.clone())
                } else {
                    None
                }
            })
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());

        // Internal event channel: backend → forwarding task → supervisor channel.
        // Unbounded so the HTTP stream reader is never blocked by backpressure.
        let (event_tx, mut event_rx) = mpsc::unbounded_channel::<AgentEvent>();

        let registry = ToolRegistry::default();

        let backend: Box<dyn AgentBackend + Send> = match &kind {
            AgentKind::AcpProcess { binary, args } => {
                let args: Vec<&str> = args.iter().map(String::as_str).collect();
                Box::new(AcpClient::spawn(binary, &args, event_tx).await?)
            }
            AgentKind::Ollama { base_url, model, api_key_env } => {
                let api_key = api_key_env.as_deref()
                    .and_then(|env| std::env::var(env).ok());
                let config = OllamaConfig { base_url: base_url.clone(), model: model.clone(), api_key };
                let tool_descs: Vec<ToolDescriptor> = registry.all_tools()
                    .map(|t| ToolDescriptor {
                        name: t.name().to_string(),
                        description: t.description().to_string(),
                        schema: t.input_schema(),
                    })
                    .collect();
                Box::new(OllamaBackend::new(config, event_tx, tool_descs, cwd.clone()))
            }
            AgentKind::LlamaCpp { base_url, model, api_key_env } => {
                let api_key = api_key_env.as_deref()
                    .and_then(|env| std::env::var(env).ok());
                let config = OpenAICompatConfig { base_url: base_url.clone(), model: model.clone(), api_key };
                let tool_descs: Vec<ToolDescriptor> = registry.all_tools()
                    .map(|t| ToolDescriptor {
                        name: t.name().to_string(),
                        description: t.description().to_string(),
                        schema: t.input_schema(),
                    })
                    .collect();
                Box::new(OpenAICompatBackend::new(config, event_tx, tool_descs, cwd.clone()))
            }
            AgentKind::Mlx { base_url, model, api_key_env } => {
                let api_key = api_key_env.as_deref()
                    .and_then(|env| std::env::var(env).ok());
                let config = OpenAICompatConfig { base_url: base_url.clone(), model: model.clone(), api_key };
                let tool_descs: Vec<ToolDescriptor> = registry.all_tools()
                    .map(|t| ToolDescriptor {
                        name: t.name().to_string(),
                        description: t.description().to_string(),
                        schema: t.input_schema(),
                    })
                    .collect();
                Box::new(OpenAICompatBackend::new(config, event_tx, tool_descs, cwd.clone()))
            }
            _ => anyhow::bail!("Unsupported agent kind in MVP"),
        };

        info!(%agent_id, %name, "Agent spawned");
        let _ = self.event_tx.send(SupervisorEvent::AgentSpawned(info.clone()));

        // Forward streaming events (TextDelta, Error) from the backend's internal
        // channel to the supervisor channel. This runs independently of turn-driving.
        let sup_tx_fwd = self.event_tx.clone();
        let aid_fwd = agent_id.clone();
        tokio::spawn(async move {
            while let Some(event) = event_rx.recv().await {
                // Unbounded send — never blocks, never deadlocks.
                let _ = sup_tx_fwd
                    .send(SupervisorEvent::AgentEvent { agent_id: aid_fwd.clone(), event });
            }
        });

        // Per-agent turn-driver: receives commands and drives the full
        // turn loop (including tool-use) in a background task.
        let (cmd_tx, cmd_rx) = mpsc::unbounded_channel::<AgentCmd>();
        let sup_tx_driver = self.event_tx.clone();
        let aid_driver = agent_id.clone();
        tokio::spawn(async move {
            agent_turn_task(aid_driver, backend, registry, cwd, cmd_rx, sup_tx_driver).await;
        });

        self.agents.insert(agent_id.clone(), AgentHandle { info, cmd_tx });
        Ok(agent_id)
    }

    /// Send a prompt to an agent — returns immediately, turn runs in background.
    pub fn prompt_agent(&self, agent_id: &AgentId, text: &str) -> Result<()> {
        let handle = self.agents.get(agent_id)
            .ok_or_else(|| anyhow::anyhow!("Agent {agent_id} not found"))?;
        handle.cmd_tx.send(AgentCmd::Prompt(text.to_string()))
            .map_err(|_| anyhow::anyhow!("Agent {agent_id} turn-driver task is gone"))?;
        Ok(())
    }

    /// Reset conversation history for all active agents — called on /clear.
    pub fn reset_all_conversations(&self) {
        for handle in self.agents.values() {
            let _ = handle.cmd_tx.send(AgentCmd::ResetConversation);
        }
    }

    /// Kill an agent. Dropping `cmd_tx` signals the turn-driver task to stop.
    pub async fn kill_agent(&mut self, agent_id: &AgentId) -> Result<()> {
        if let Some(_handle) = self.agents.remove(agent_id) {
            warn!(%agent_id, "Agent killed");
            let _ = self.event_tx.send(SupervisorEvent::AgentDied {
                agent_id: agent_id.clone(),
                reason: DeathReason::Killed,
            });
        }
        Ok(())
    }

    pub fn list_agents(&self) -> Vec<&AgentInfo> {
        self.agents.values().map(|h| &h.info).collect()
    }

    pub fn get_agent(&self, agent_id: &AgentId) -> Option<&AgentInfo> {
        self.agents.get(agent_id).map(|h| &h.info)
    }
}

/// Background task that owns one agent backend and drives its turn loop.
/// Receives prompts from the channel; each prompt triggers a full turn
/// (including tool-use loops) without blocking the UI event loop.
async fn agent_turn_task(
    agent_id: AgentId,
    mut backend: Box<dyn AgentBackend + Send>,
    registry: ToolRegistry,
    cwd: PathBuf,
    mut cmd_rx: mpsc::UnboundedReceiver<AgentCmd>,
    sup_tx: mpsc::UnboundedSender<SupervisorEvent>,
) {
    while let Some(cmd) = cmd_rx.recv().await {
        let prompt = match cmd {
            AgentCmd::ResetConversation => {
                backend.reset_conversation().await;
                info!(%agent_id, "supervisor: conversation reset");
                continue;
            }
            AgentCmd::Prompt(p) => p,
        };
        info!(%agent_id, prompt_len = prompt.len(), "supervisor: starting turn");
        if let Err(e) = backend.start_turn(&prompt).await {
            warn!(%agent_id, "supervisor: start_turn error: {e}");
            let _ = sup_tx.send(SupervisorEvent::AgentEvent {
                agent_id: agent_id.clone(),
                event: AgentEvent::Error(e.to_string()),
            });
            let _ = sup_tx.send(SupervisorEvent::AgentEvent {
                agent_id: agent_id.clone(),
                event: AgentEvent::TurnComplete { stop_reason: "error".into() },
            });
            continue;
        }

        loop {
            info!(%agent_id, "supervisor: awaiting stream_until_pause");
            match backend.stream_until_pause().await {
                Err(e) => {
                    warn!(%agent_id, "supervisor: stream_until_pause error: {e}");
                    let _ = sup_tx.send(SupervisorEvent::AgentEvent {
                        agent_id: agent_id.clone(),
                        event: AgentEvent::Error(e.to_string()),
                    });
                    let _ = sup_tx.send(SupervisorEvent::AgentEvent {
                        agent_id: agent_id.clone(),
                        event: AgentEvent::TurnComplete {
                            stop_reason: "error".into(),
                        },
                    });
                    break;
                }
                Ok(StreamPause::Done { stop_reason }) => {
                    info!(%agent_id, %stop_reason, "supervisor: turn complete");
                    let _ = sup_tx.send(SupervisorEvent::AgentEvent {
                        agent_id: agent_id.clone(),
                        event: AgentEvent::TurnComplete { stop_reason },
                    });
                    break;
                }
                Ok(StreamPause::ToolUse(tools)) => {
                    // Notify the UI of each tool call.
                    for tool in &tools {
                        let _ = sup_tx.send(SupervisorEvent::AgentEvent {
                            agent_id: agent_id.clone(),
                            event: AgentEvent::ToolCallRequested {
                                id: tool.id.clone(),
                                name: tool.name.clone(),
                                input: tool.input.clone(),
                            },
                        });
                    }

                    // Execute tools, emit results to UI, and collect for the backend.
                    let mut results: Vec<(String, serde_json::Value)> = vec![];
                    for tool in tools {
                        let output = run_tool(&registry, &tool.name, tool.input.clone(), cwd.clone()).await;
                        let _ = sup_tx.send(SupervisorEvent::AgentEvent {
                            agent_id: agent_id.clone(),
                            event: AgentEvent::ToolResult {
                                id: tool.id.clone(),
                                name: tool.name.clone(),
                                output: output.content.clone(),
                                is_error: output.is_error,
                            },
                        });
                        results.push((tool.id, output.to_json()));
                    }

                    info!(%agent_id, count = results.len(), "supervisor: submitting tool results");
                    if let Err(e) = backend.submit_tool_results(&results).await {
                        warn!(%agent_id, "supervisor: submit_tool_results error: {e}");
                        break;
                    }
                }
            }
        }
    }
}

/// Execute a single tool call (auto-approved for agent backends).
async fn run_tool(
    registry: &ToolRegistry,
    name: &str,
    input: serde_json::Value,
    cwd: PathBuf,
) -> ToolOutput {
    match registry.get(name) {
        Some(tool) => {
            let ctx = ToolContext {
                agent_id: AgentId::new(),
                session_id: SessionId::new(),
                cwd,
            };
            tool.execute(input, ctx, CancellationToken::new())
                .await
                .unwrap_or_else(|e| ToolOutput::error(e.to_string()))
        }
        None => ToolOutput::error(format!("Unknown tool: {name}")),
    }
}
