# Contributing to Beyond

Thanks for your interest in Beyond! This document covers the practical mechanics of contributing. For the architectural overview, see [`README.md`](./README.md) and [`CLAUDE.md`](./CLAUDE.md).

## Ground rules

- Be respectful. See [`CODE_OF_CONDUCT.md`](./CODE_OF_CONDUCT.md).
- By contributing, you agree your work is licensed under the project's GPL-3.0-or-later license.
- Security issues: please **do not** open a public issue — see [`SECURITY.md`](./SECURITY.md).

## Getting set up

```bash
git clone git@github.com:PewterZz/Beyond.git
cd Beyond
cargo build
cargo run
```

You'll need Rust stable (2021 edition) and a working `ollama serve` for anything agent-related.

## Workflow

1. **Open an issue first** for non-trivial work. Small bug fixes can go straight to a PR; features and refactors should be discussed first so you don't waste effort.
2. Branch from `main`: `git checkout -b feat/my-thing` (or `fix/…`, `refactor/…`, `docs/…`).
3. Make the change. Keep the diff scoped — one concern per PR.
4. Before pushing, run the checks below.
5. Open a PR targeting `main`. Fill in the PR template.

## Commit style

- Use conventional-ish prefixes: `feat:`, `fix:`, `refactor:`, `docs:`, `chore:`, `test:`.
- Write commits that describe **why**, not just **what**. A good body is worth more than a clever title.
- Keep commits logically atomic. Rebase-and-squash fixup commits before review.

## Pre-PR checklist

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo build --release   # optional, but catches LTO-only issues
```

CI runs the same. PRs that fail fmt/clippy/tests won't be merged.

## Code conventions

- **No unnecessary comments.** Well-named identifiers are the documentation. Only add a comment when the *why* is non-obvious (a hidden constraint, a workaround, a surprising invariant).
- **Don't introduce abstractions for hypothetical future needs.** Three similar lines beats a premature trait.
- **IDs are ULID-backed.** Use `BlockId` / `AgentId` / `SessionId` from `beyonder-core`; don't invent new ID types.
- **Workspace deps go in the root `Cargo.toml`** and are referenced in crate manifests as `foo = { workspace = true }`.
- **Tool calls go through `CapabilityBroker`.** Never bypass it.
- **Don't move state-advancement into `RedrawRequested`.** Keep it in `about_to_wait` — macOS suppresses redraws for hidden windows.

## Tests

- Prefer integration-style tests that exercise real behavior over mock-heavy unit tests.
- For bug fixes: write a failing test first, then fix it.
- Crate-scoped tests live in each crate's `tests/` or inline `#[cfg(test)]` modules.

## Reporting bugs

Open a GitHub issue with:

- What you did (exact steps, commands, config).
- What you expected.
- What actually happened, including any `RUST_LOG=beyonder=debug` output.
- OS, GPU, and `rustc --version`.

## Proposing features

Open an issue describing the use case and sketch of the design. For larger changes, a short design note in the issue is better than a surprise PR.

## Project layout

See the crate table in [`README.md`](./README.md). In brief: `core` → `store`/`terminal`/`acp` → `runtime` → `gpu`/`ui` → binary.

Thanks for helping make Beyond better.
