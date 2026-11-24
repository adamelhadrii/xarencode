#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${XARENCODE_REPO:-https://github.com/adamelhadrii/xarencode.git}"
VENV_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/xarencode/venv"
BIN_DIR="$HOME/.local/bin"

echo "==> Installing XarenCode..."

if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 is required (3.11+)." >&2
    exit 1
fi

PY_OK=$(python3 -c 'import sys; print(int(sys.version_info >= (3, 11)))')
if [ "$PY_OK" != "1" ]; then
    echo "Error: Python 3.11+ is required (found $(python3 --version))." >&2
    exit 1
fi

mkdir -p "$(dirname "$VENV_DIR")" "$BIN_DIR"

if command -v pipx >/dev/null 2>&1; then
    echo "==> Installing via pipx..."
    pipx install --force "git+${REPO_URL}"
elif command -v uv >/dev/null 2>&1; then
    echo "==> Installing via uv..."
    uv tool install --force "git+${REPO_URL}"
else
    echo "==> Creating virtual environment at $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --quiet --upgrade pip
    "$VENV_DIR/bin/pip" install --upgrade "git+${REPO_URL}"
    ln -sf "$VENV_DIR/bin/xarencode" "$BIN_DIR/xarencode"
fi

if ! command -v llama-server >/dev/null 2>&1; then
    echo ""
    echo "Note: llama-server was not found on PATH."
    if command -v pacman >/dev/null 2>&1; then
        echo "  Install it with: sudo pacman -S --needed llama-cpp ripgrep"
    else
        echo "  Install llama.cpp (llama-server) for your OS before running xarencode."
    fi
fi

case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *)
        echo ""
        echo "Note: $BIN_DIR is not in your PATH. Add this to your shell rc:"
        echo "  export PATH=\"$BIN_DIR:\$PATH\""
        ;;
esac

echo ""
echo "==> Running xarencode doctor..."
"$BIN_DIR/xarencode" doctor || true

echo ""
echo "Done! Next steps:"
echo "  1. Download the default model (~2.4 GiB):  xarencode pull"
echo "  2. Start chatting in any project folder:   xarencode"
