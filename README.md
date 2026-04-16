# Beyond

An AI-native terminal written in Rust. Beyond replaces the traditional scroll buffer with a **block-oriented** model — every piece of content (shell output, agent messages, approvals, diffs, tool calls) is a persistent, addressable `Block` with provenance. Agents are first-class, long-lived processes with capability sets and resource limits, supervised like OS processes. Rendering is GPU-accelerated (`wgpu` + `glyphon`) inside a single `winit` window.

> Status: early. Phase 2 (glyphon 0.8 text, wgpu 24, continuous render loop) has landed. Expect breaking changes.

## Highlights

- **Blocks, not scrollback.** Shell commands, agent turns, and approvals are all `Block`s with ULID IDs, parent/child relationships, and a provenance chain.
- **Agents as processes.** The `AgentSupervisor` spawns, monitors, and kills agents. Each has a `CapabilitySet` and `ResourceLimits`; tool calls flow through a `CapabilityBroker`.
- **Multi-provider LLM support.** Ollama (local + Cloud/Turbo), llama.cpp, and Apple MLX — switch at runtime with `/provider`.
- **GPU rendering.** Single `winit` window, `wgpu` 24, `glyphon` 0.8 for text, custom per-block renderers.
- **OSC-133 shell integration.** `zsh`/`bash` hooks let Beyond see where one command ends and the next begins.

## Requirements

- A GPU that supports `wgpu` (Metal on macOS, Vulkan/DX12 elsewhere)
- At least one LLM provider running for agent features:
  - [Ollama](https://ollama.com) — `ollama serve` (local) or Ollama Cloud (Turbo)
  - [llama.cpp](https://github.com/ggerganov/llama.cpp) — `llama-server -m model.gguf --jinja -c 8192`
  - [Apple MLX](https://github.com/ml-explore/mlx-lm) — `mlx_lm.server --model <id>` (macOS Apple Silicon)

## Installation

### Quick install (macOS / Linux)

```bash
curl -fsSL https://raw.githubusercontent.com/PewterZz/Beyond/main/install.sh | bash
```

Or specify a version:

```bash
VERSION=0.1.0 curl -fsSL https://raw.githubusercontent.com/PewterZz/Beyond/main/install.sh | bash
```

### Cargo (from source)

```bash
cargo install beyonder
```

### Homebrew

```bash
brew tap PewterZz/tap
brew install beyond
```

### Build from source

```bash
git clone git@github.com:PewterZz/Beyond.git
cd Beyond
cargo build --release
# Binary at target/release/beyonder
```

## Quick start

```bash
beyonder
```

The window opens at 1280x800. Type shell commands normally, or `/help` for slash commands. `/agent spawn <name>` to start an agent.

## Common commands

```bash
cargo build                      # debug build
cargo build --release            # release (LTO, codegen-units=1)
cargo run                        # launch the app
cargo check -p <crate>           # fast single-crate check
cargo test                       # all tests
cargo test -p beyonder-core      # single crate
cargo clippy --workspace --all-targets
cargo fmt --all
```

Logging uses `tracing`. Tune with `RUST_LOG` (default `beyonder=info,wgpu_core=warn,wgpu_hal=warn`). Logs go to **stderr**; redirect with `cargo run 2> run.log`.

## Architecture

The workspace binary is a thin `winit::ApplicationHandler` that owns the tokio runtime and delegates to `beyonder-ui::App`. Crates form a layered graph:

| Crate | Role |
|---|---|
| `beyonder-core` | Pure data model: `Block`, `AgentInfo`, `SessionId`, `CapabilitySet`, `ProvenanceChain`. No I/O. |
| `beyonder-store` | SQLite persistence (bundled `rusqlite`): `BlockStore`, `SessionStore`, migrations. |
| `beyonder-terminal` | PTY (`portable-pty`) + terminal emulation (`alacritty_terminal`), OSC-133 shell hooks, `BlockBuilder`. |
| `beyonder-acp` | Agent Client Protocol: messages, transport, `AcpClient`. |
| `beyonder-runtime` | `AgentSupervisor`, `CapabilityBroker`, tool registry, `AgentBackend` trait + Ollama/llama.cpp/MLX providers. |
| `beyonder-gpu` | `wgpu` renderer, `glyphon` text atlas, per-block renderers, viewport/scrolling. |
| `beyonder-ui` | `App` wiring, input editor, slash commands, history, mode detection. |
| `beyonder-config` | TOML config loader (`BeyonderConfig`). |

State-advancement (`App::tick()`) runs in winit's `about_to_wait`, **not** inside `RedrawRequested` — macOS suppresses redraws for hidden windows, and we don't want streaming agent output to stall when the window is occluded.

See [`CLAUDE.md`](./CLAUDE.md) for a deeper tour written for AI coding assistants working in this repo.

## Configuration

Config lives at `$XDG_CONFIG_HOME/beyond/config.toml` (or the platform equivalent). Missing or malformed config falls back to defaults — see `BeyonderConfig::load_or_default`.

### LLM Providers

```toml
# Ollama (default)
[provider]
kind = "ollama"
base_url = "http://localhost:11434"       # optional; cloud: "https://ollama.com"
api_key_env = "OLLAMA_API_KEY"            # optional; omit for local

# llama.cpp
[provider]
kind = "llama_cpp"
base_url = "http://127.0.0.1:8080/v1"

# Apple MLX (macOS Apple Silicon)
[provider]
kind = "mlx"
base_url = "http://127.0.0.1:8080/v1"
```

Switch at runtime with `/provider ollama|llama_cpp|mlx` and `/model <name>`. Changes take effect on the next agent spawn.

## Contributing

See [`CONTRIBUTING.md`](./CONTRIBUTING.md) and [`CODE_OF_CONDUCT.md`](./CODE_OF_CONDUCT.md). In short: keep PRs focused, run `cargo fmt` and `cargo clippy --workspace --all-targets` before pushing, and add tests for behavior changes.

## License

GPL-3.0-or-later. See [`LICENSE`](./LICENSE).
