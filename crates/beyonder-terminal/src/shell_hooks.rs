//! Shell integration hooks that emit OSC sequences delineating prompt /
//! command-start / command-end so the block builder can carve the byte stream
//! into discrete `ShellCommand` blocks.
//!
//! Every shell emits **both** OSC 633 (VS Code's extension — carries the
//! command text and cwd) and OSC 133 (the FinalTerm standard — universal but
//! info-light). External tools see standard 133; we use the richer 633.

use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShellKind {
    Zsh,
    Bash,
    Fish,
    Nushell,
    Unknown,
}

/// Identify a shell by the basename of its path. Falls back to `Unknown`,
/// which means "no integration" — the PTY still runs, just without block
/// boundary markers.
pub fn detect_shell_kind(shell_path: &str) -> ShellKind {
    let name = Path::new(shell_path)
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();
    match name.as_str() {
        "zsh" => ShellKind::Zsh,
        "bash" => ShellKind::Bash,
        "fish" => ShellKind::Fish,
        "nu" | "nushell" => ShellKind::Nushell,
        _ => ShellKind::Unknown,
    }
}

/// Zsh: precmd + preexec hooks. Emits 633 (with cwd) and 133.
pub fn zsh_init_script(session_id: &str) -> String {
    format!(
        r#"
# Beyonder shell integration for zsh
export BEYONDER_SESSION_ID="{session_id}"
unsetopt PROMPT_SP
unsetopt PROMPT_CR

beyonder_preexec() {{
    local cmd="$1"
    printf '\033]633;A\007'
    printf '\033]633;E;%s\007' "$cmd"
    printf '\033]133;C\007'
}}

beyonder_precmd() {{
    local code=$?
    printf '\033]133;D;%d\007' "$code"
    printf '\033]633;B;%d\007' "$code"
    printf '\033]633;P;Cwd=%s\007' "$PWD"
    printf '\033]133;A\007'
}}

autoload -Uz add-zsh-hook
add-zsh-hook preexec beyonder_preexec
add-zsh-hook precmd beyonder_precmd
"#
    )
}

/// Bash: DEBUG trap (preexec) + PROMPT_COMMAND (precmd). Emits 633 + 133.
pub fn bash_init_script(session_id: &str) -> String {
    format!(
        r#"
# Beyonder shell integration for bash
export BEYONDER_SESSION_ID="{session_id}"

beyonder_preexec() {{
    printf '\033]633;A\007'
    printf '\033]633;E;%s\007' "$1"
    printf '\033]133;C\007'
}}

beyonder_precmd() {{
    local code=$?
    printf '\033]133;D;%d\007' "$code"
    printf '\033]633;B;%d\007' "$code"
    printf '\033]633;P;Cwd=%s\007' "$PWD"
    printf '\033]133;A\007'
}}

trap 'beyonder_preexec "$BASH_COMMAND"' DEBUG
PROMPT_COMMAND="beyonder_precmd${{PROMPT_COMMAND:+;$PROMPT_COMMAND}}"
"#
    )
}

/// Fish: event-based hooks (preexec / postexec / prompt). Emits 633 + 133.
pub fn fish_init_script(session_id: &str) -> String {
    format!(
        r#"
# Beyonder shell integration for fish
set -gx BEYONDER_SESSION_ID "{session_id}"

function __beyonder_preexec --on-event fish_preexec
    printf '\e]633;A\a'
    printf '\e]633;E;%s\a' $argv[1]
    printf '\e]133;C\a'
end

function __beyonder_postexec --on-event fish_postexec
    set -l code $status
    printf '\e]133;D;%d\a' $code
    printf '\e]633;B;%d\a' $code
    printf '\e]633;P;Cwd=%s\a' (pwd)
end

function __beyonder_prompt --on-event fish_prompt
    printf '\e]133;A\a'
end
"#
    )
}

/// Nushell: pre_execution + pre_prompt config hooks. Emits 633 + 133.
/// Exit code comes from `$env.LAST_EXIT_CODE` since nu has no post-execution hook.
pub fn nushell_init_script(session_id: &str) -> String {
    format!(
        r#"
# Beyonder shell integration for nushell
$env.BEYONDER_SESSION_ID = "{session_id}"

$env.config = ($env.config? | default {{}})
$env.config.hooks = ($env.config.hooks? | default {{}})

let __beyonder_pre_exec = {{|cmd|
    print -n $"(char esc)]633;A(char bel)"
    print -n $"(char esc)]633;E;($cmd)(char bel)"
    print -n $"(char esc)]133;C(char bel)"
}}

let __beyonder_pre_prompt = {{||
    let code = ($env.LAST_EXIT_CODE? | default 0)
    print -n $"(char esc)]133;D;($code)(char bel)"
    print -n $"(char esc)]633;B;($code)(char bel)"
    print -n $"(char esc)]633;P;Cwd=(pwd)(char bel)"
    print -n $"(char esc)]133;A(char bel)"
}}

$env.config.hooks.pre_execution = (($env.config.hooks.pre_execution? | default []) | append $__beyonder_pre_exec)
$env.config.hooks.pre_prompt = (($env.config.hooks.pre_prompt? | default []) | append $__beyonder_pre_prompt)
"#
    )
}

/// OSC sequence markers emitted by the shell hooks.
pub mod markers {
    pub const BEL: u8 = 0x07;
}
