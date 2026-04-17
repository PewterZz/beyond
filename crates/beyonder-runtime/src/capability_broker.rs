//! The capability broker is Beyonder's security boundary.
//! It intercepts every agent action and enforces permissions,
//! creating ApprovalRequest blocks when human approval is needed.

use beyonder_core::{
    AgentAction, AgentId, ApprovalMode, Block, BlockContent, BlockKind, BlockStatus,
    CapabilityKind, CapabilitySet, SessionId,
};
use std::collections::HashMap;
use tokio::sync::{mpsc, oneshot};
use tracing::info;

/// A pending approval waiting for the human's decision.
pub struct PendingApproval {
    pub block: Block,
    pub responder: oneshot::Sender<ApprovalDecision>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ApprovalDecision {
    Granted,
    GrantedAlways,
    Denied,
}

/// Events emitted by the broker.
#[allow(clippy::large_enum_variant)]
#[derive(Debug)]
pub enum BrokerEvent {
    /// A new approval block that needs to be shown to the user.
    ApprovalRequired(Block),
    /// An action was auto-approved (no UI needed).
    AutoApproved { agent_id: AgentId, action: String },
    /// An action was denied.
    Denied { agent_id: AgentId, action: String },
}

/// Intercepts agent actions and enforces capabilities.
pub struct CapabilityBroker {
    /// Per-agent capability sets.
    capabilities: HashMap<AgentId, CapabilitySet>,
    /// Channel to emit broker events to the UI layer.
    event_tx: mpsc::Sender<BrokerEvent>,
    /// Pending approvals waiting for human response.
    pending: HashMap<String, oneshot::Sender<ApprovalDecision>>,
    /// Session-wide approval policy. Overrides per-capability grant modes —
    /// switching this re-applies on the next action check.
    approval_mode: ApprovalMode,
}

impl CapabilityBroker {
    pub fn new(event_tx: mpsc::Sender<BrokerEvent>) -> Self {
        Self::with_mode(event_tx, ApprovalMode::default())
    }

    pub fn with_mode(event_tx: mpsc::Sender<BrokerEvent>, mode: ApprovalMode) -> Self {
        Self {
            capabilities: HashMap::new(),
            event_tx,
            pending: HashMap::new(),
            approval_mode: mode,
        }
    }

    pub fn approval_mode(&self) -> ApprovalMode {
        self.approval_mode
    }

    pub fn set_approval_mode(&mut self, mode: ApprovalMode) {
        self.approval_mode = mode;
    }

    /// Register an agent with its initial capability set.
    pub fn register_agent(&mut self, agent_id: AgentId, caps: CapabilitySet) {
        self.capabilities.insert(agent_id, caps);
    }

    /// Check if an agent's action is permitted, and if so, how.
    /// Returns an approval channel if human input is required.
    pub async fn check_action(
        &mut self,
        agent_id: &AgentId,
        action: &AgentAction,
        session_id: &SessionId,
    ) -> ActionDecision {
        let kind = action_to_capability_kind(action);

        // Session-wide ApprovalMode is the top-level gate.
        match self.approval_mode {
            ApprovalMode::Bypass => {
                info!(agent = %agent_id, action = ?kind, "Auto-approved (bypass)");
                let _ = self
                    .event_tx
                    .send(BrokerEvent::AutoApproved {
                        agent_id: agent_id.clone(),
                        action: kind.display_name().to_string(),
                    })
                    .await;
                return ActionDecision::Approved;
            }
            ApprovalMode::Auto => {
                if is_safe_action(action) {
                    info!(agent = %agent_id, action = ?kind, "Auto-approved (auto/safe)");
                    let _ = self
                        .event_tx
                        .send(BrokerEvent::AutoApproved {
                            agent_id: agent_id.clone(),
                            action: kind.display_name().to_string(),
                        })
                        .await;
                    return ActionDecision::Approved;
                }
                // Fall through to approval request below.
            }
            ApprovalMode::Manual => {
                // Fall through to approval request below.
            }
        }

        // Ask the human.
        let (tx, rx) = oneshot::channel();
        let block = make_approval_block(agent_id, action, session_id);
        let block_id = block.id.clone();

        self.pending.insert(block_id.0.clone(), tx);

        let _ = self
            .event_tx
            .send(BrokerEvent::ApprovalRequired(block))
            .await;

        ActionDecision::NeedsApproval { approval_rx: rx }
    }

    /// Called when the human makes an approval decision for a given block.
    pub fn resolve_approval(&mut self, block_id: &str, decision: ApprovalDecision) {
        if let Some(tx) = self.pending.remove(block_id) {
            let _ = tx.send(decision);
        }
    }

    /// Grant an additional capability to an agent at runtime.
    pub fn grant_capability(&mut self, agent_id: &AgentId, cap: beyonder_core::Capability) {
        self.capabilities
            .entry(agent_id.clone())
            .or_default()
            .add(cap);
    }
}

pub enum ActionDecision {
    Approved,
    Denied(String),
    NeedsApproval {
        approval_rx: oneshot::Receiver<ApprovalDecision>,
    },
}

/// Whether an action is safe enough to skip approval in Auto mode.
/// Reads are safe; anything that mutates, executes, or reaches the network is not.
fn is_safe_action(action: &AgentAction) -> bool {
    matches!(action, AgentAction::FileRead { .. })
}

fn action_to_capability_kind(action: &AgentAction) -> CapabilityKind {
    match action {
        AgentAction::FileRead { .. } => CapabilityKind::FileRead { patterns: vec![] },
        AgentAction::FileWrite { .. } => CapabilityKind::FileWrite { patterns: vec![] },
        AgentAction::FileDelete { .. } => CapabilityKind::FileDelete { patterns: vec![] },
        AgentAction::ShellExecute { .. } => CapabilityKind::ShellExecute {
            allowed_commands: None,
        },
        AgentAction::NetworkRequest { .. } => CapabilityKind::NetworkAccess {
            allowed_hosts: vec![],
        },
        AgentAction::AgentSpawn { .. } => CapabilityKind::AgentSpawn,
        AgentAction::ToolUse { tool_name } => CapabilityKind::ToolUse {
            tool_names: vec![tool_name.clone()],
        },
    }
}

#[allow(dead_code)]
fn action_path(action: &AgentAction) -> Option<std::path::PathBuf> {
    match action {
        AgentAction::FileRead { path }
        | AgentAction::FileWrite { path, .. }
        | AgentAction::FileDelete { path } => Some(path.clone()),
        _ => None,
    }
}

fn make_approval_block(agent_id: &AgentId, action: &AgentAction, session_id: &SessionId) -> Block {
    let mut block = Block::new(
        BlockKind::Approval,
        session_id.clone(),
        BlockContent::ApprovalRequest {
            action: action.clone(),
            reasoning: None,
            granted: None,
            granter: None,
        },
    );
    block.agent_id = Some(agent_id.clone());
    block.status = BlockStatus::Pending;
    block
}
