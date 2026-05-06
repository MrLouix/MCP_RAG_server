#!/bin/bash
set -euo pipefail

# ─── Config bootstrap ───────────────────────────────────────
# If user mounted ./config:/app/config with their own config.yaml,
# this block does nothing. If no mount is present, the Dockerfile-
# bundled config.example.yaml becomes the active config.
if [ ! -f /app/config/config.yaml ]; then
    mkdir -p /app/config
    cp /app/config.example.yaml /app/config/config.yaml
    echo "[entrypoint] Created default /app/config/config.yaml"
fi

# ─── Ensure working directories exist & are writable ─────────
# These may have been auto-created by Docker with root ownership.
# We fix permissions so the host user can read/write them.
for d in /app/documents /app/models /app/rag_index; do
    if [ -d "$d" ] || mkdir -p "$d"; then
        chmod 777 "$d"
    fi
done

# Defaults
TRANSPORT="${RAG_TRANSPORT:-${TRANSPORT:-stdio}}"
CONFIG="${RAG_CONFIG:-/app/config/config.yaml}"

echo "=== MCP RAG Documents Server ==="
echo "Transport: $TRANSPORT"
echo "Config:    $CONFIG"
echo ""

if [ "$TRANSPORT" = "sse" ]; then
    exec python3 -m mcp_rag.server --config "$CONFIG" --transport sse \
        --port "${PORT:-3000}" --host "${HOST:-0.0.0.0}"
else
    exec python3 -m mcp_rag.server --config "$CONFIG" --transport stdio
fi
