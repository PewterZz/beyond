#!/usr/bin/env bash
set -euo pipefail

REPO="PewterZz/Beyond"
BINARIES="beyondtty beyonder"
INSTALL_DIR="${INSTALL_DIR:-/usr/local/bin}"

info()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
err()   { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

detect_platform() {
    local os arch
    os="$(uname -s)"
    arch="$(uname -m)"

    case "$os" in
        Darwin) os="apple-darwin" ;;
        Linux)  os="unknown-linux-gnu" ;;
        *)      err "Unsupported OS: $os" ;;
    esac

    case "$arch" in
        x86_64|amd64)  arch="x86_64" ;;
        arm64|aarch64) arch="aarch64" ;;
        *)             err "Unsupported architecture: $arch" ;;
    esac

    echo "${arch}-${os}"
}

get_latest_version() {
    if command -v curl &>/dev/null; then
        curl -fsSL "https://api.github.com/repos/${REPO}/releases/latest" \
            | grep '"tag_name"' | head -1 | sed -E 's/.*"v?([^"]+)".*/\1/'
    elif command -v wget &>/dev/null; then
        wget -qO- "https://api.github.com/repos/${REPO}/releases/latest" \
            | grep '"tag_name"' | head -1 | sed -E 's/.*"v?([^"]+)".*/\1/'
    else
        err "curl or wget is required"
    fi
}

download() {
    local url="$1" dest="$2"
    if command -v curl &>/dev/null; then
        curl -fsSL -o "$dest" "$url"
    else
        wget -qO "$dest" "$url"
    fi
}

main() {
    local version="${VERSION:-}"
    local platform
    platform="$(detect_platform)"

    if [ -z "$version" ]; then
        info "Fetching latest release..."
        version="$(get_latest_version)"
        [ -n "$version" ] || err "Could not determine latest version. Set VERSION=x.y.z manually."
    fi

    local tarball="beyondtty-v${version}-${platform}.tar.gz"
    local url="https://github.com/${REPO}/releases/download/v${version}/${tarball}"

    info "Downloading Beyond v${version} for ${platform}..."
    local tmpdir
    tmpdir="$(mktemp -d)"
    trap 'rm -rf "$tmpdir"' EXIT

    download "$url" "${tmpdir}/${tarball}"

    info "Extracting..."
    tar -xzf "${tmpdir}/${tarball}" -C "$tmpdir"

    for bin in $BINARIES; do
        if [ ! -f "${tmpdir}/${bin}" ]; then
            err "Binary '${bin}' not found in archive"
        fi
        info "Installing ${bin} to ${INSTALL_DIR}/${bin}..."
        if [ -w "$INSTALL_DIR" ]; then
            mv "${tmpdir}/${bin}" "${INSTALL_DIR}/${bin}"
        else
            sudo mv "${tmpdir}/${bin}" "${INSTALL_DIR}/${bin}"
        fi
        chmod +x "${INSTALL_DIR}/${bin}"
    done

    info "Beyond v${version} installed to ${INSTALL_DIR}"
    echo ""
    echo "  Run 'beyondtty' or 'beyonder' to launch."
    echo ""
    echo "  Make sure you have a GPU that supports wgpu (Metal/Vulkan/DX12)"
    echo "  and at least one LLM provider running for agent features."
    echo ""
    echo "  Recommended:"
    echo "    - nvim (or vim / nano) — opened when you click a file link"
    echo "      in the block stream. Respects \$VISUAL / \$EDITOR."
}

main "$@"
