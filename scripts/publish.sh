#!/usr/bin/env bash
set -euo pipefail

# Publish order: leaf crates first, then dependents, then root.
# Each crate must be live on crates.io before its dependents can publish.

CRATES=(
    beyonder-core
    beyonder-store
    beyonder-acp
    beyonder-config
    beyonder-terminal
    beyonder-remote
    beyonder-gpu
    beyonder-runtime
    beyonder-ui
    beyonder
)

DRY_RUN="${DRY_RUN:-}"

for crate in "${CRATES[@]}"; do
    printf '\033[1;34m==>\033[0m Publishing %s...\n' "$crate"
    if [ -n "$DRY_RUN" ]; then
        cargo publish -p "$crate" --dry-run --allow-dirty
    else
        cargo publish -p "$crate"
        printf '    Waiting for crates.io index to update...\n'
        sleep 30
    fi
done

echo ""
echo "All crates published."
