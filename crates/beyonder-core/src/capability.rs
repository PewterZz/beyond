use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// The core security primitive in Beyonder.
/// Every action an agent wants to take requires a matching capability token.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capability {
    pub kind: CapabilityKind,
    pub scope: CapabilityScope,
    pub grant_mode: GrantMode,
}

impl Capability {
    pub fn new(kind: CapabilityKind, scope: CapabilityScope, grant_mode: GrantMode) -> Self {
        Self {
            kind,
            scope,
            grant_mode,
        }
    }

    /// Check if this capability covers the given action kind and path.
    pub fn covers_file_action(&self, action_kind: &CapabilityKind, path: &Path) -> bool {
        if std::mem::discriminant(&self.kind) != std::mem::discriminant(action_kind) {
            return false;
        }
        match &self.scope {
            CapabilityScope::Directory(dir) => path.starts_with(dir),
            CapabilityScope::Global => true,
            CapabilityScope::Session(_) => true,
        }
    }
}

/// What the capability grants permission to do.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CapabilityKind {
    FileRead {
        patterns: Vec<String>,
    },
    FileWrite {
        patterns: Vec<String>,
    },
    FileDelete {
        patterns: Vec<String>,
    },
    ShellExecute {
        allowed_commands: Option<Vec<String>>,
    },
    NetworkAccess {
        allowed_hosts: Vec<String>,
    },
    AgentSpawn,
    ToolUse {
        tool_names: Vec<String>,
    },
    HumanPrompt,
}

impl CapabilityKind {
    pub fn display_name(&self) -> &'static str {
        match self {
            Self::FileRead { .. } => "File Read",
            Self::FileWrite { .. } => "File Write",
            Self::FileDelete { .. } => "File Delete",
            Self::ShellExecute { .. } => "Shell Execute",
            Self::NetworkAccess { .. } => "Network Access",
            Self::AgentSpawn => "Spawn Agent",
            Self::ToolUse { .. } => "Tool Use",
            Self::HumanPrompt => "Ask Human",
        }
    }
}

/// The scope within which the capability applies.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CapabilityScope {
    Directory(PathBuf),
    Session(String),
    Global,
}

/// How the capability is granted — controls the approval flow.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GrantMode {
    /// Auto-approved, no human prompt.
    Always,
    /// Ask once; remember for this session.
    Once,
    /// Ask every time.
    PerUse,
    /// Never allowed.
    Never,
}

/// A set of capabilities held by an agent.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CapabilitySet {
    pub capabilities: Vec<Capability>,
}

impl CapabilitySet {
    pub fn add(&mut self, cap: Capability) {
        self.capabilities.push(cap);
    }

    /// Find the grant mode for a given action kind and target path.
    /// Returns None if no capability covers this action (requires user approval).
    pub fn grant_mode_for(
        &self,
        kind: &CapabilityKind,
        path: Option<&PathBuf>,
    ) -> Option<&GrantMode> {
        self.capabilities
            .iter()
            .find(|cap| {
                std::mem::discriminant(&cap.kind) == std::mem::discriminant(kind)
                    && path
                        .map(|p| cap.covers_file_action(kind, p))
                        .unwrap_or(true)
            })
            .map(|cap| &cap.grant_mode)
    }

    /// Build a default capability set for general coding agents (MVP defaults).
    pub fn default_coding_agent(workspace: PathBuf) -> Self {
        let mut set = Self::default();
        set.add(Capability::new(
            CapabilityKind::FileRead {
                patterns: vec!["**".to_string()],
            },
            CapabilityScope::Directory(workspace.clone()),
            GrantMode::Always,
        ));
        set.add(Capability::new(
            CapabilityKind::FileWrite {
                patterns: vec!["**".to_string()],
            },
            CapabilityScope::Directory(workspace.clone()),
            GrantMode::Once,
        ));
        set.add(Capability::new(
            CapabilityKind::ShellExecute {
                allowed_commands: None,
            },
            CapabilityScope::Global,
            GrantMode::PerUse,
        ));
        set.add(Capability::new(
            CapabilityKind::NetworkAccess {
                allowed_hosts: vec![],
            },
            CapabilityScope::Global,
            GrantMode::Never,
        ));
        set
    }
}
