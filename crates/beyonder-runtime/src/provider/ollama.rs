//! Ollama provider — local (http://localhost:11434) and cloud (https://ollama.com
//! Turbo/Pro). Same `/api/chat` NDJSON streaming API; only base_url and
//! optional bearer auth differ.

use anyhow::Result;
use async_trait::async_trait;
use beyonder_acp::client::{AgentEvent, StreamPause, ToolUseReq};
use beyonder_core::ApprovalMode;
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::sync::mpsc::UnboundedSender;
use tracing::{debug, info, warn};

#[derive(Clone, Debug)]
pub struct OllamaConfig {
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
}

impl OllamaConfig {
    /// Auto-detect: if OLLAMA_API_KEY is set → cloud+glm-5.1, else → local+qwen2.5-coder:7b.
    pub fn auto() -> Self {
        match std::env::var("OLLAMA_API_KEY") {
            Ok(key) if !key.is_empty() => Self {
                base_url: "https://ollama.com".to_string(),
                api_key: Some(key),
                model: "glm-5.1".to_string(),
            },
            _ => Self {
                base_url: "http://localhost:11434".to_string(),
                api_key: None,
                model: "qwen2.5-coder:7b".to_string(),
            },
        }
    }
}

/// Tool schema fed to Ollama's `tools` array. Sourced from the ToolRegistry.
#[derive(Clone, Debug)]
pub struct ToolDescriptor {
    pub name: String,
    pub description: String,
    pub schema: serde_json::Value,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct OllamaMessage {
    role: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    content: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OllamaToolCall>>,
    /// Only set on role=tool — names the tool this message is a result for.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    name: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct OllamaToolCall {
    function: OllamaFunction,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
struct OllamaFunction {
    name: String,
    /// Ollama emits this as a JSON object (not a string).
    arguments: serde_json::Value,
}

#[derive(Serialize, Clone, Debug)]
struct OllamaTool {
    #[serde(rename = "type")]
    kind: String,
    function: OllamaToolDef,
}

#[derive(Serialize, Clone, Debug)]
struct OllamaToolDef {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

pub struct OllamaBackend {
    http: Client,
    config: OllamaConfig,
    tools: Vec<OllamaTool>,
    messages: Vec<OllamaMessage>,
    event_tx: UnboundedSender<AgentEvent>,
    /// Ollama doesn't emit tool_call_ids, so we generate and track them to
    /// map result submissions back to the right function name.
    pending_tool_calls: Vec<(String, OllamaToolCall)>,
}

impl OllamaBackend {
    pub fn new(
        config: OllamaConfig,
        event_tx: UnboundedSender<AgentEvent>,
        tools: Vec<ToolDescriptor>,
        cwd: std::path::PathBuf,
        approval_mode: ApprovalMode,
    ) -> Self {
        let ollama_tools: Vec<OllamaTool> = tools
            .iter()
            .map(|t| OllamaTool {
                kind: "function".to_string(),
                function: OllamaToolDef {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    parameters: t.schema.clone(),
                },
            })
            .collect();

        let system_msg = OllamaMessage {
            role: "system".to_string(),
            content: super::build_system_prompt(&cwd, &tools, approval_mode),
            tool_calls: None,
            name: None,
        };

        Self {
            http: Client::builder()
                .connect_timeout(Duration::from_secs(10))
                .tcp_keepalive(Duration::from_secs(30))
                .read_timeout(Duration::from_secs(120))
                .build()
                .expect("reqwest client"),
            config,
            tools: ollama_tools,
            messages: vec![system_msg],
            event_tx,
            pending_tool_calls: vec![],
        }
    }

    async fn post_and_stream(&mut self) -> Result<StreamPause> {
        let url = format!("{}/api/chat", self.config.base_url);
        let body = serde_json::json!({
            "model": self.config.model,
            "messages": self.messages,
            "tools": self.tools,
            "stream": true,
        });

        info!(model = %self.config.model, url = %url, "ollama: sending request");

        let mut req = self.http.post(&url).json(&body);
        if let Some(key) = &self.config.api_key {
            req = req.bearer_auth(key);
        }

        let resp = req
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("Ollama request failed: {e}"))?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            warn!(status = %status, body = %body, "ollama: request failed");
            anyhow::bail!("Ollama returned {status}: {body}");
        }

        info!(status = %status, "ollama: stream opened");

        let mut stream = resp.bytes_stream();
        let mut acc_content = String::new();
        let mut tool_calls: Vec<OllamaToolCall> = vec![];
        let mut done_reason = String::new();
        let mut buf: Vec<u8> = Vec::new();
        let mut chunk_count: usize = 0;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| {
                warn!(
                    chunks_received = chunk_count,
                    "ollama: stream error mid-response: {e}"
                );
                anyhow::anyhow!("Stream error: {e}")
            })?;
            chunk_count += 1;
            debug!(
                chunk = chunk_count,
                bytes = chunk.len(),
                "ollama: chunk received"
            );
            buf.extend_from_slice(&chunk);
            while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
                let line: Vec<u8> = buf.drain(..=pos).collect();
                let line_str = match std::str::from_utf8(&line) {
                    Ok(s) => s.trim(),
                    Err(_) => continue,
                };
                if line_str.is_empty() {
                    continue;
                }
                let val: serde_json::Value = match serde_json::from_str(line_str) {
                    Ok(v) => v,
                    Err(e) => {
                        warn!("ollama: parse error: {e} — raw line: {line_str}");
                        continue;
                    }
                };

                if let Some(msg) = val.get("message") {
                    if let Some(c) = msg.get("content").and_then(|v| v.as_str()) {
                        if !c.is_empty() {
                            debug!(bytes = c.len(), "ollama: text delta");
                            acc_content.push_str(c);
                            let _ = self.event_tx.send(AgentEvent::TextDelta(c.to_string()));
                        }
                    }
                    if let Some(tc) = msg.get("tool_calls") {
                        if let Ok(parsed) =
                            serde_json::from_value::<Vec<OllamaToolCall>>(tc.clone())
                        {
                            info!(count = parsed.len(), "ollama: tool calls received");
                            tool_calls.extend(parsed);
                        }
                    }
                }

                if val.get("done").and_then(|v| v.as_bool()).unwrap_or(false) {
                    done_reason = val
                        .get("done_reason")
                        .and_then(|v| v.as_str())
                        .unwrap_or("stop")
                        .to_string();
                    info!(done_reason = %done_reason, chunks = chunk_count, "ollama: done frame received");
                }
            }
        }

        info!(
            chunks = chunk_count,
            content_len = acc_content.len(),
            buf_remaining = buf.len(),
            "ollama: stream closed"
        );

        // Flush any remaining bytes in buf — Ollama occasionally sends the final
        // `{"done":true}` frame without a trailing newline, which the loop above
        // never processes. Treat whatever is left as one last line.
        let remaining = buf.trim_ascii();
        if !remaining.is_empty() {
            if let Ok(line_str) = std::str::from_utf8(remaining) {
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(line_str) {
                    if let Some(msg) = val.get("message") {
                        if let Some(c) = msg.get("content").and_then(|v| v.as_str()) {
                            if !c.is_empty() {
                                acc_content.push_str(c);
                                let _ = self.event_tx.send(AgentEvent::TextDelta(c.to_string()));
                            }
                        }
                        if let Some(tc) = msg.get("tool_calls") {
                            if let Ok(parsed) =
                                serde_json::from_value::<Vec<OllamaToolCall>>(tc.clone())
                            {
                                tool_calls.extend(parsed);
                            }
                        }
                    }
                    if val.get("done").and_then(|v| v.as_bool()).unwrap_or(false) {
                        done_reason = val
                            .get("done_reason")
                            .and_then(|v| v.as_str())
                            .unwrap_or("stop")
                            .to_string();
                    }
                }
            }
        }

        // Record the assistant turn in conversation history.
        self.messages.push(OllamaMessage {
            role: "assistant".to_string(),
            content: acc_content,
            tool_calls: if tool_calls.is_empty() {
                None
            } else {
                Some(tool_calls.clone())
            },
            name: None,
        });

        if !tool_calls.is_empty() {
            let mut reqs = Vec::with_capacity(tool_calls.len());
            self.pending_tool_calls.clear();
            for (i, call) in tool_calls.iter().enumerate() {
                let id = format!("tc_{}_{}", self.messages.len(), i);
                reqs.push(ToolUseReq {
                    id: id.clone(),
                    name: call.function.name.clone(),
                    input: call.function.arguments.clone(),
                });
                self.pending_tool_calls.push((id, call.clone()));
            }
            Ok(StreamPause::ToolUse(reqs))
        } else {
            let stop_reason = if done_reason.is_empty() {
                "stop".to_string()
            } else {
                done_reason
            };
            Ok(StreamPause::Done { stop_reason })
        }
    }
}

#[async_trait]
impl super::AgentBackend for OllamaBackend {
    async fn start_turn(&mut self, user_text: &str) -> Result<()> {
        self.messages.push(OllamaMessage {
            role: "user".to_string(),
            content: user_text.to_string(),
            tool_calls: None,
            name: None,
        });
        Ok(())
    }

    async fn stream_until_pause(&mut self) -> Result<StreamPause> {
        self.post_and_stream().await
    }

    async fn reset_conversation(&mut self) {
        // Keep only the system message (index 0); drop all user/assistant/tool turns.
        self.messages.truncate(1);
        self.pending_tool_calls.clear();
    }

    async fn submit_tool_results(&mut self, results: &[(String, serde_json::Value)]) -> Result<()> {
        for (id, result) in results {
            let tool_name = self
                .pending_tool_calls
                .iter()
                .find(|(pid, _)| pid == id)
                .map(|(_, tc)| tc.function.name.clone())
                .unwrap_or_default();
            // ToolOutput.to_json() shape: {"type": "text"|"error", "text": "..."}
            let content = result
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            self.messages.push(OllamaMessage {
                role: "tool".to_string(),
                content,
                tool_calls: None,
                name: Some(tool_name),
            });
        }
        self.pending_tool_calls.clear();
        Ok(())
    }
}
