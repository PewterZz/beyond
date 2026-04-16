//! ACP client — connects Beyonder to an ACP-compatible agent process.

use anyhow::{bail, Context, Result};
use serde_json::json;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::process::{Child, Command};
use tokio::sync::{mpsc::UnboundedSender, oneshot};
use tracing::info;

use crate::messages::{
    methods, AcpContentBlock, AcpMessage, AcpStreamEvent, ClientCapabilities, ClientInfo,
    ContentDelta, InitializeParams, InitializeResult, JsonRpcRequest, PromptTurnParams,
};
use crate::transport::{StdioReceiver, StdioSender};

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

fn next_id() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::SeqCst)
}

/// Events emitted by the ACP client during a prompt turn.
#[derive(Debug, Clone)]
pub enum AgentEvent {
    /// Text delta from the agent.
    TextDelta(String),
    /// The agent wants to call a tool — requires capability check.
    ToolCallRequested {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    /// A tool has finished executing.
    ToolResult {
        id: String,
        name: String,
        output: String,
        is_error: bool,
    },
    /// The turn is complete.
    TurnComplete { stop_reason: String },
    /// An error occurred.
    Error(String),
}

/// A tool use request extracted from the stream.
#[derive(Debug, Clone)]
pub struct ToolUseReq {
    pub id: String,
    pub name: String,
    pub input: serde_json::Value,
}

/// Pause point returned by `stream_until_pause`.
pub enum StreamPause {
    ToolUse(Vec<ToolUseReq>),
    Done { stop_reason: String },
}

/// Internal state for accumulating streaming tool input JSON.
struct PartialTool {
    id: String,
    name: String,
    input_json: String,
}

/// The ACP client manages the connection to one agent subprocess.
pub struct AcpClient {
    sender: StdioSender,
    receiver: StdioReceiver,
    _child: Child,
    #[allow(dead_code)]
    pending: HashMap<u64, oneshot::Sender<serde_json::Value>>,
    /// Channel to forward streaming events to the caller.
    event_tx: UnboundedSender<AgentEvent>,
    /// Partially-received tool calls, keyed by content_block index.
    pending_tools: HashMap<usize, PartialTool>,
}

impl AcpClient {
    /// Spawn an ACP agent subprocess and initialize the connection.
    pub async fn spawn(
        binary: &str,
        args: &[&str],
        event_tx: UnboundedSender<AgentEvent>,
    ) -> Result<Self> {
        info!(binary, "Spawning ACP agent");
        let mut child = Command::new(binary)
            .args(args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::inherit())
            .spawn()
            .with_context(|| format!("Failed to spawn {binary}"))?;

        let stdin = child.stdin.take().context("No stdin")?;
        let stdout = child.stdout.take().context("No stdout")?;

        let mut client = Self {
            sender: StdioSender::new(stdin),
            receiver: StdioReceiver::spawn(stdout),
            _child: child,
            pending: HashMap::new(),
            event_tx,
            pending_tools: HashMap::new(),
        };

        client.initialize().await?;
        Ok(client)
    }

    /// Send the ACP initialize handshake.
    async fn initialize(&mut self) -> Result<()> {
        let id = next_id();
        let params = InitializeParams {
            client_info: ClientInfo {
                name: "Beyond".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
            capabilities: ClientCapabilities::default(),
        };

        let request = JsonRpcRequest::new(id, methods::INITIALIZE, serde_json::to_value(&params)?);
        self.sender.send(&request).await?;

        let response = self.recv_response(id).await?;
        let _result: InitializeResult =
            serde_json::from_value(response).context("Failed to parse initialize result")?;

        let notif = JsonRpcRequest::notification(methods::INITIALIZED, json!({}));
        self.sender.send(&notif).await?;

        info!("ACP handshake complete");
        Ok(())
    }

    /// Send a user prompt message. Does not stream — call `stream_until_pause` after.
    pub async fn start_prompt(&mut self, text: &str) -> Result<()> {
        let id = next_id();
        let params = PromptTurnParams {
            session_id: None,
            messages: vec![AcpMessage::user(text)],
        };

        let request = JsonRpcRequest::new(id, methods::PROMPT_TURN, serde_json::to_value(&params)?);
        self.sender.send(&request).await?;
        Ok(())
    }

    /// Drive the stream until a pause point (tool use or end of turn).
    /// TextDelta and Error events are forwarded via the event_tx channel.
    /// TurnComplete is NOT sent here on Done — the caller does that.
    pub async fn stream_until_pause(&mut self) -> Result<StreamPause> {
        let mut completed_tools: Vec<ToolUseReq> = Vec::new();

        loop {
            let Some(value) = self.receiver.recv().await else {
                bail!("ACP agent disconnected");
            };

            if let Ok(event) = serde_json::from_value::<AcpStreamEvent>(value.clone()) {
                match event {
                    AcpStreamEvent::ContentBlockDelta { index, delta } => match delta {
                        ContentDelta::TextDelta { text } => {
                            let _ = self.event_tx.send(AgentEvent::TextDelta(text));
                        }
                        ContentDelta::InputJsonDelta { partial_json } => {
                            if let Some(pt) = self.pending_tools.get_mut(&index) {
                                pt.input_json.push_str(&partial_json);
                            }
                        }
                        ContentDelta::ThinkingDelta { .. } => {}
                    },
                    AcpStreamEvent::ContentBlockStart {
                        index,
                        content_block,
                    } => {
                        if let AcpContentBlock::ToolUse { id, name, input } = content_block {
                            if input != serde_json::Value::Null
                                && input != serde_json::Value::Object(Default::default())
                            {
                                // Input already present, store as completed right away
                                // on ContentBlockStop we'll finalize it
                                let input_json = serde_json::to_string(&input).unwrap_or_default();
                                self.pending_tools.insert(
                                    index,
                                    PartialTool {
                                        id,
                                        name,
                                        input_json,
                                    },
                                );
                            } else {
                                self.pending_tools.insert(
                                    index,
                                    PartialTool {
                                        id,
                                        name,
                                        input_json: String::new(),
                                    },
                                );
                            }
                        }
                    }
                    AcpStreamEvent::ContentBlockStop { index } => {
                        if let Some(pt) = self.pending_tools.remove(&index) {
                            let input = if pt.input_json.is_empty() {
                                serde_json::Value::Object(Default::default())
                            } else {
                                serde_json::from_str(&pt.input_json)
                                    .unwrap_or(serde_json::Value::Object(Default::default()))
                            };
                            completed_tools.push(ToolUseReq {
                                id: pt.id,
                                name: pt.name,
                                input,
                            });
                        }
                    }
                    AcpStreamEvent::MessageStop { stop_reason } => {
                        self.pending_tools.clear();
                        if stop_reason == "tool_use" {
                            return Ok(StreamPause::ToolUse(completed_tools));
                        } else {
                            return Ok(StreamPause::Done { stop_reason });
                        }
                    }
                    AcpStreamEvent::Error { error } => {
                        let _ = self.event_tx.send(AgentEvent::Error(error.clone()));
                        bail!("Agent error: {error}");
                    }
                }
            }
        }
    }

    /// Submit tool results back to the agent.
    pub async fn submit_tool_results(
        &mut self,
        results: &[(String, serde_json::Value)],
    ) -> Result<()> {
        for (tool_use_id, result) in results {
            let id = next_id();
            let request = JsonRpcRequest::new(
                id,
                methods::TOOLS_CALL,
                json!({
                    "tool_use_id": tool_use_id,
                    "result": result,
                }),
            );
            self.sender.send(&request).await?;
        }
        Ok(())
    }

    /// Send a user prompt and stream events back (backward-compat).
    /// On tool_use stop, emits ToolCallRequested events and TurnComplete, then returns.
    pub async fn prompt(&mut self, text: &str) -> Result<()> {
        self.start_prompt(text).await?;
        match self.stream_until_pause().await? {
            StreamPause::Done { stop_reason } => {
                let _ = self.event_tx.send(AgentEvent::TurnComplete { stop_reason });
            }
            StreamPause::ToolUse(tools) => {
                for tool in &tools {
                    let _ = self.event_tx.send(AgentEvent::ToolCallRequested {
                        id: tool.id.clone(),
                        name: tool.name.clone(),
                        input: tool.input.clone(),
                    });
                }
                let _ = self.event_tx.send(AgentEvent::TurnComplete {
                    stop_reason: "tool_use".to_string(),
                });
            }
        }
        Ok(())
    }

    /// Respond to a tool call with a result (legacy single-call API).
    pub async fn tool_result(
        &mut self,
        tool_use_id: &str,
        result: serde_json::Value,
    ) -> Result<()> {
        let id = next_id();
        let request = JsonRpcRequest::new(
            id,
            methods::TOOLS_CALL,
            json!({
                "tool_use_id": tool_use_id,
                "result": result,
            }),
        );
        self.sender.send(&request).await?;
        Ok(())
    }

    async fn recv_response(&mut self, id: u64) -> Result<serde_json::Value> {
        loop {
            let Some(value) = self.receiver.recv().await else {
                bail!("ACP agent disconnected while waiting for response {id}");
            };

            if let Some(msg_id) = value.get("id") {
                let msg_id = msg_id.as_u64().unwrap_or(0);
                if msg_id == id {
                    if let Some(result) = value.get("result") {
                        return Ok(result.clone());
                    } else if let Some(error) = value.get("error") {
                        bail!("RPC error: {error}");
                    }
                }
            }
        }
    }
}
