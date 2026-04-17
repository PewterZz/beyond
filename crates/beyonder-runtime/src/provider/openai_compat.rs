//! OpenAI-compatible chat-completions backend.
//!
//! Works with any server that speaks `/v1/chat/completions` with SSE streaming
//! and OpenAI-style tool calling — specifically llama.cpp `llama-server` (start
//! with `--jinja`) and Apple MLX `mlx_lm.server` (mlx-lm >= 0.19).
//!
//! Key protocol differences vs Ollama NDJSON:
//!   • Framing: SSE (`data: {…}\n\n`, terminated by `data: [DONE]\n\n`).
//!   • Tool-call arguments arrive as a **string** that must be concatenated
//!     across multiple delta events before parsing as JSON.
//!   • Tool result messages use `tool_call_id` not `name`.
//!   • The server assigns real `call_xxx` ids; we use them verbatim.

use anyhow::Result;
use async_trait::async_trait;
use beyonder_acp::client::{AgentEvent, StreamPause, ToolUseReq};
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::sync::mpsc::UnboundedSender;
use tracing::{debug, info, warn};

use super::ToolDescriptor;

#[derive(Clone, Debug)]
pub struct OpenAICompatConfig {
    /// Base URL including the `/v1` path segment, e.g. `http://127.0.0.1:8080/v1`.
    pub base_url: String,
    pub model: String,
    pub api_key: Option<String>,
}

// ── Wire types (request side) ─────────────────────────────────────────────────

#[derive(Serialize, Clone, Debug)]
struct OAITool {
    #[serde(rename = "type")]
    kind: &'static str,
    function: OAIToolDef,
}

#[derive(Serialize, Clone, Debug)]
struct OAIToolDef {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

/// A message in the running conversation history.
#[derive(Serialize, Clone, Debug)]
struct OAIMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    /// Only present on assistant turns that produced tool calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OAIAssistantToolCall>>,
    /// Only present on tool-result turns (role = "tool").
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

/// The tool-call entry recorded in an assistant message for history.
/// `arguments` is kept as a string (as OpenAI requires) even though we parse
/// it internally to build `ToolUseReq::input`.
#[derive(Serialize, Clone, Debug)]
struct OAIAssistantToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    function: OAIAssistantFunction,
}

#[derive(Serialize, Clone, Debug)]
struct OAIAssistantFunction {
    name: String,
    arguments: String,
}

// ── Wire types (response / SSE side) ─────────────────────────────────────────

#[derive(Deserialize, Debug)]
struct ChatCompletionChunk {
    choices: Vec<ChunkChoice>,
}

#[derive(Deserialize, Debug)]
struct ChunkChoice {
    delta: Delta,
    finish_reason: Option<String>,
}

#[derive(Deserialize, Debug)]
struct Delta {
    content: Option<String>,
    tool_calls: Option<Vec<ToolCallDelta>>,
}

/// A single streaming tool-call delta. All fields are optional because they
/// arrive across multiple events: the first event has `id`+`name`, subsequent
/// events have only `index`+`arguments` fragment.
#[derive(Deserialize, Debug)]
struct ToolCallDelta {
    index: usize,
    id: Option<String>,
    function: Option<ToolCallFunctionDelta>,
}

#[derive(Deserialize, Debug)]
struct ToolCallFunctionDelta {
    name: Option<String>,
    arguments: Option<String>,
}

// ── Backend ───────────────────────────────────────────────────────────────────

/// Accumulator for a single tool call being assembled across SSE deltas.
struct PartialToolCall {
    id: String,
    name: String,
    arguments_buf: String,
}

pub struct OpenAICompatBackend {
    http: Client,
    config: OpenAICompatConfig,
    tools: Vec<OAITool>,
    messages: Vec<OAIMessage>,
    event_tx: UnboundedSender<AgentEvent>,
    /// Completed tool calls from the last response, keyed by id.
    /// Held until submit_tool_results clears them.
    pending_tool_calls: Vec<(String, String)>, // (id, name)
}

impl OpenAICompatBackend {
    pub fn new(
        config: OpenAICompatConfig,
        event_tx: UnboundedSender<AgentEvent>,
        tools: Vec<ToolDescriptor>,
        cwd: std::path::PathBuf,
        approval_mode: beyonder_core::ApprovalMode,
    ) -> Self {
        let oai_tools: Vec<OAITool> = tools
            .iter()
            .map(|t| OAITool {
                kind: "function",
                function: OAIToolDef {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    parameters: t.schema.clone(),
                },
            })
            .collect();

        let system_msg = OAIMessage {
            role: "system".to_string(),
            content: Some(super::build_system_prompt(&cwd, &tools, approval_mode)),
            tool_calls: None,
            tool_call_id: None,
        };

        Self {
            http: Client::builder()
                .connect_timeout(Duration::from_secs(10))
                .tcp_keepalive(Duration::from_secs(30))
                .read_timeout(Duration::from_secs(120))
                .build()
                .expect("reqwest client"),
            config,
            tools: oai_tools,
            messages: vec![system_msg],
            event_tx,
            pending_tool_calls: vec![],
        }
    }

    async fn post_and_stream(&mut self) -> Result<StreamPause> {
        let url = format!("{}/chat/completions", self.config.base_url);
        let body = serde_json::json!({
            "model": self.config.model,
            "messages": self.messages,
            "tools": self.tools,
            "stream": true,
            "stream_options": { "include_usage": false },
        });

        info!(model = %self.config.model, url = %url, "oai_compat: sending request");

        let mut req = self.http.post(&url).json(&body);
        if let Some(key) = &self.config.api_key {
            req = req.bearer_auth(key);
        }

        let resp = req
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("OpenAI-compat request failed: {e}"))?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            warn!(status = %status, body = %body, "oai_compat: request failed");
            anyhow::bail!("OpenAI-compat server returned {status}: {body}");
        }

        info!(status = %status, "oai_compat: stream opened");

        let mut stream = resp.bytes_stream();
        let mut acc_content = String::new();
        let mut partial_calls: Vec<PartialToolCall> = vec![];
        let mut finish_reason = String::new();
        let mut buf: Vec<u8> = Vec::new();
        let mut chunk_count: usize = 0;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| {
                warn!(
                    chunks_received = chunk_count,
                    "oai_compat: stream error mid-response: {e}"
                );
                anyhow::anyhow!("Stream error: {e}")
            })?;
            chunk_count += 1;
            debug!(
                chunk = chunk_count,
                bytes = chunk.len(),
                "oai_compat: chunk received"
            );
            buf.extend_from_slice(&chunk);

            // SSE events are delimited by double newlines.
            loop {
                // Find `\n\n` or `\r\n\r\n`.
                let boundary = find_sse_boundary(&buf);
                if let Some(end) = boundary {
                    let event_bytes: Vec<u8> = buf.drain(..end).collect();
                    // Drain the delimiter itself.
                    while buf.first() == Some(&b'\n') || buf.first() == Some(&b'\r') {
                        buf.remove(0);
                    }

                    let event_str = match std::str::from_utf8(&event_bytes) {
                        Ok(s) => s.trim(),
                        Err(_) => continue,
                    };

                    if event_str.is_empty() {
                        continue;
                    }

                    // Each SSE event can have multiple `data:` lines (rare but valid).
                    for line in event_str.lines() {
                        let data = if let Some(rest) = line.strip_prefix("data:") {
                            rest.trim()
                        } else {
                            continue;
                        };

                        if data == "[DONE]" {
                            info!(chunks = chunk_count, "oai_compat: [DONE] received");
                            continue;
                        }

                        let chunk_val: ChatCompletionChunk = match serde_json::from_str(data) {
                            Ok(v) => v,
                            Err(e) => {
                                warn!("oai_compat: parse error: {e} — raw: {data}");
                                continue;
                            }
                        };

                        for choice in chunk_val.choices {
                            if let Some(reason) = choice.finish_reason {
                                if !reason.is_empty() && reason != "null" {
                                    finish_reason = reason;
                                }
                            }

                            let delta = choice.delta;

                            if let Some(c) = delta.content {
                                if !c.is_empty() {
                                    debug!(bytes = c.len(), "oai_compat: text delta");
                                    acc_content.push_str(&c);
                                    let _ = self.event_tx.send(AgentEvent::TextDelta(c));
                                }
                            }

                            if let Some(tc_deltas) = delta.tool_calls {
                                for tc in tc_deltas {
                                    // Grow the partial-calls vec to accommodate this index.
                                    while partial_calls.len() <= tc.index {
                                        partial_calls.push(PartialToolCall {
                                            id: String::new(),
                                            name: String::new(),
                                            arguments_buf: String::new(),
                                        });
                                    }
                                    let partial = &mut partial_calls[tc.index];
                                    if let Some(id) = tc.id {
                                        partial.id = id;
                                    }
                                    if let Some(func) = tc.function {
                                        if let Some(name) = func.name {
                                            partial.name = name;
                                        }
                                        if let Some(args_frag) = func.arguments {
                                            partial.arguments_buf.push_str(&args_frag);
                                        }
                                    }
                                }
                            }
                        }
                    }
                } else {
                    break;
                }
            }
        }

        info!(
            chunks = chunk_count,
            content_len = acc_content.len(),
            tool_calls = partial_calls.len(),
            "oai_compat: stream closed"
        );

        // Build finalized tool calls from assembled fragments.
        let mut finalized: Vec<(String, String, serde_json::Value)> = vec![]; // (id, name, input)
        let mut assistant_tool_calls: Vec<OAIAssistantToolCall> = vec![];

        for partial in partial_calls {
            if partial.name.is_empty() {
                continue;
            }
            let input: serde_json::Value = match serde_json::from_str(&partial.arguments_buf) {
                Ok(v) => v,
                Err(e) => {
                    warn!(
                        name = %partial.name,
                        "oai_compat: could not parse tool arguments: {e} — raw: {}",
                        partial.arguments_buf
                    );
                    let _ = self.event_tx.send(AgentEvent::Error(format!(
                        "Tool '{}' returned unparseable arguments: {e}",
                        partial.name
                    )));
                    // Skip this call — don't surface a partial/corrupt tool call.
                    continue;
                }
            };
            assistant_tool_calls.push(OAIAssistantToolCall {
                id: partial.id.clone(),
                kind: "function",
                function: OAIAssistantFunction {
                    name: partial.name.clone(),
                    arguments: partial.arguments_buf.clone(),
                },
            });
            finalized.push((partial.id, partial.name, input));
        }

        // Record the assistant turn in conversation history.
        self.messages.push(OAIMessage {
            role: "assistant".to_string(),
            content: if acc_content.is_empty() {
                None
            } else {
                Some(acc_content)
            },
            tool_calls: if assistant_tool_calls.is_empty() {
                None
            } else {
                Some(assistant_tool_calls)
            },
            tool_call_id: None,
        });

        if !finalized.is_empty() {
            let mut reqs = Vec::with_capacity(finalized.len());
            self.pending_tool_calls.clear();
            for (id, name, input) in finalized {
                reqs.push(ToolUseReq {
                    id: id.clone(),
                    name: name.clone(),
                    input,
                });
                self.pending_tool_calls.push((id, name));
            }
            info!(count = reqs.len(), "oai_compat: tool calls ready");
            Ok(StreamPause::ToolUse(reqs))
        } else {
            let stop_reason = if finish_reason.is_empty() {
                "stop".to_string()
            } else {
                finish_reason
            };
            Ok(StreamPause::Done { stop_reason })
        }
    }
}

#[async_trait]
impl super::AgentBackend for OpenAICompatBackend {
    async fn start_turn(&mut self, user_text: &str) -> Result<()> {
        self.messages.push(OAIMessage {
            role: "user".to_string(),
            content: Some(user_text.to_string()),
            tool_calls: None,
            tool_call_id: None,
        });
        Ok(())
    }

    async fn stream_until_pause(&mut self) -> Result<StreamPause> {
        self.post_and_stream().await
    }

    async fn reset_conversation(&mut self) {
        self.messages.truncate(1);
        self.pending_tool_calls.clear();
    }

    async fn submit_tool_results(&mut self, results: &[(String, serde_json::Value)]) -> Result<()> {
        for (id, result) in results {
            let content = result
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            self.messages.push(OAIMessage {
                role: "tool".to_string(),
                content: Some(content),
                tool_calls: None,
                tool_call_id: Some(id.clone()),
            });
        }
        self.pending_tool_calls.clear();
        Ok(())
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Find the position just before the next SSE event boundary (`\n\n` or `\r\n\r\n`).
/// Returns the byte index of the start of the delimiter.
fn find_sse_boundary(buf: &[u8]) -> Option<usize> {
    // Search for \n\n
    for i in 0..buf.len().saturating_sub(1) {
        if buf[i] == b'\n' && buf[i + 1] == b'\n' {
            return Some(i);
        }
        // Also handle \r\n\r\n
        if i + 3 < buf.len()
            && buf[i] == b'\r'
            && buf[i + 1] == b'\n'
            && buf[i + 2] == b'\r'
            && buf[i + 3] == b'\n'
        {
            return Some(i);
        }
    }
    None
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sse(data: &str) -> Vec<u8> {
        format!("data: {data}\n\n").into_bytes()
    }

    /// Feed a synthetic SSE stream that has:
    ///   1. Two text-delta events
    ///   2. A tool-call spread across three events (first event has id+name,
    ///      next two have argument fragments)
    ///   3. A [DONE] terminator
    /// Then check that the reassembled tool-call input is correct JSON.
    #[tokio::test]
    async fn test_sse_tool_call_reassembly() {
        let stream_bytes: Vec<u8> = [
            // Text deltas
            sse(r#"{"choices":[{"delta":{"content":"Hello "},"finish_reason":null}]}"#),
            sse(r#"{"choices":[{"delta":{"content":"world"},"finish_reason":null}]}"#),
            // First tool-call delta: id + name (arguments empty)
            sse(r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_abc","function":{"name":"shell_exec","arguments":""}}]},"finish_reason":null}]}"#),
            // Second tool-call delta: first half of arguments string
            sse(r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"cmd\":"}}]},"finish_reason":null}]}"#),
            // Third tool-call delta: second half of arguments string
            sse(r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"ls -la\"}"}}]},"finish_reason":"tool_calls"}]}"#),
            // DONE
            sse("[DONE]"),
        ]
        .concat();

        // Drive the SSE parser directly — parse the raw bytes as if post_and_stream did it.
        let mut acc_content = String::new();
        let mut partial_calls: Vec<PartialToolCall> = vec![];
        let mut finish_reason = String::new();
        let mut buf = stream_bytes;

        loop {
            let boundary = find_sse_boundary(&buf);
            if let Some(end) = boundary {
                let event_bytes: Vec<u8> = buf.drain(..end).collect();
                while buf.first() == Some(&b'\n') || buf.first() == Some(&b'\r') {
                    buf.remove(0);
                }
                let event_str = std::str::from_utf8(&event_bytes).unwrap().trim();
                if event_str.is_empty() {
                    continue;
                }

                for line in event_str.lines() {
                    let data = if let Some(rest) = line.strip_prefix("data:") {
                        rest.trim()
                    } else {
                        continue;
                    };
                    if data == "[DONE]" {
                        continue;
                    }

                    let chunk: ChatCompletionChunk = serde_json::from_str(data).unwrap();
                    for choice in chunk.choices {
                        if let Some(r) = choice.finish_reason {
                            if !r.is_empty() && r != "null" {
                                finish_reason = r;
                            }
                        }
                        if let Some(c) = choice.delta.content {
                            acc_content.push_str(&c);
                        }
                        if let Some(tc_deltas) = choice.delta.tool_calls {
                            for tc in tc_deltas {
                                while partial_calls.len() <= tc.index {
                                    partial_calls.push(PartialToolCall {
                                        id: String::new(),
                                        name: String::new(),
                                        arguments_buf: String::new(),
                                    });
                                }
                                let partial = &mut partial_calls[tc.index];
                                if let Some(id) = tc.id {
                                    partial.id = id;
                                }
                                if let Some(func) = tc.function {
                                    if let Some(name) = func.name {
                                        partial.name = name;
                                    }
                                    if let Some(args) = func.arguments {
                                        partial.arguments_buf.push_str(&args);
                                    }
                                }
                            }
                        }
                    }
                }
            } else {
                break;
            }
        }

        assert_eq!(acc_content, "Hello world");
        assert_eq!(finish_reason, "tool_calls");
        assert_eq!(partial_calls.len(), 1);
        assert_eq!(partial_calls[0].id, "call_abc");
        assert_eq!(partial_calls[0].name, "shell_exec");

        let input: serde_json::Value = serde_json::from_str(&partial_calls[0].arguments_buf)
            .expect("arguments_buf must parse as JSON");
        assert_eq!(input["cmd"], "ls -la");
    }

    #[test]
    fn test_find_sse_boundary_lf() {
        let buf = b"data: foo\n\ndata: bar\n\n";
        assert_eq!(find_sse_boundary(buf), Some(9));
    }

    #[test]
    fn test_find_sse_boundary_crlf() {
        let buf = b"data: foo\r\n\r\ndata: bar\r\n\r\n";
        assert_eq!(find_sse_boundary(buf), Some(9));
    }

    #[test]
    fn test_find_sse_boundary_none() {
        let buf = b"data: foo";
        assert_eq!(find_sse_boundary(buf), None);
    }
}
