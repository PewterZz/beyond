//! Slash command registry.
//! Every /command Beyonder understands lives here — name, usage hint, description.
//! The renderer filters this list live as the user types to show a command palette.

#[derive(Debug, Clone)]
pub struct SlashCommand {
    /// The bare command name without the slash, e.g. "clear".
    pub name: &'static str,
    /// Short usage string shown in the palette, e.g. "/clear".
    pub usage: &'static str,
    /// One-line description shown next to the command.
    pub description: &'static str,
}

/// All registered slash commands.
pub static COMMANDS: &[SlashCommand] = &[
    // ── Generic ──────────────────────────────────────────────────────────────
    SlashCommand {
        name: "clear",
        usage: "/clear",
        description: "Clear all blocks from the stream",
    },
    SlashCommand {
        name: "help",
        usage: "/help",
        description: "Show available commands",
    },
    SlashCommand {
        name: "quit",
        usage: "/quit",
        description: "Exit Beyond",
    },
    SlashCommand {
        name: "exit",
        usage: "/exit",
        description: "Exit Beyond",
    },
    SlashCommand {
        name: "scroll",
        usage: "/scroll top|bottom",
        description: "Jump to top or bottom of block stream",
    },
    SlashCommand {
        name: "font",
        usage: "/font <size>",
        description: "Set font size (e.g. /font 14)",
    },
    SlashCommand {
        name: "find",
        usage: "/find <pattern>",
        description: "Search blocks with a regex (Cmd+F also toggles)",
    },
    // ── Agent ─────────────────────────────────────────────────────────────────
    SlashCommand {
        name: "agent",
        usage: "/agent list|spawn <name>|kill <id>",
        description: "Manage running agents",
    },
    // ── Beyonder-specific ─────────────────────────────────────────────────────
    SlashCommand {
        name: "mode",
        usage: "/mode auto|cmd|agent",
        description: "Switch input mode: auto, shell-only (cmd), or agent-only",
    },
    SlashCommand {
        name: "model",
        usage: "/model <name>",
        description: "Set the active AI model (e.g. qwen2.5-coder:7b)",
    },
    SlashCommand {
        name: "provider",
        usage: "/provider <name>",
        description: "Set the active AI provider (ollama | anthropic)",
    },
    SlashCommand {
        name: "session",
        usage: "/session new|list",
        description: "Manage sessions",
    },
    SlashCommand {
        name: "phone",
        usage: "/phone on|off|pair|status|tailscale|ngrok",
        description: "Bridge this terminal to the companion iOS app over LAN",
    },
    SlashCommand {
        name: "theme",
        usage: "/theme <name>",
        description: "Switch color theme (catppuccin-mocha | …)",
    },
];

/// Return commands whose name starts with `prefix` (case-insensitive).
pub fn filter(prefix: &str) -> Vec<&'static SlashCommand> {
    let lower = prefix.to_lowercase();
    COMMANDS
        .iter()
        .filter(|c| c.name.starts_with(lower.as_str()))
        .collect()
}
