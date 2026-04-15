# Changelog

All notable changes to Beyond will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) once it reaches 1.0.

## [Unreleased]

### Changed

- Renamed the user-facing product from "Beyonder" to "Beyond" (window title, ACP client identity, `TERM_PROGRAM`, agent system prompt, slash-command text). Internal crate and type names remain `beyonder_*`.
- Added `LICENSE` (GPL-3.0-or-later), `README.md`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, and this changelog.

## [0.1.0] — Phase 2

### Added

- `glyphon` 0.8 text rendering, `wgpu` 24, continuous render loop.
- Block-oriented content model (`Block`, `BlockId`, `ProvenanceChain`).
- `AgentSupervisor` with per-agent tokio turn-drivers.
- `CapabilityBroker` gating tool execution.
- Ollama backend (local + Turbo) as the sole LLM provider.
- OSC-133 shell integration for `zsh` and `bash`.
- SQLite persistence via bundled `rusqlite`.
