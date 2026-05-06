#!/bin/bash
set -euo pipefail

# Default config
TRANSPORT="${RAG_TRANSPORT:-${TRANSPORT:-stdio}}"
CONFIG="${RAG_CONFIG:-/app/config.yaml}"

echo "=== MCP RAG Documents Server ==="
echo "Transport: $TRANSPORT"
echo "Config:    $CONFIG"
echo ""

if [ "$TRANSPORT" = "sse" ]; then
    exec python3 -m mcp_rag.server --config "$CONFIG" --transport sse --port "${PORT:-3000}" --host "${HOST:-0.0.0.0}"
else
    exec python3 -m mcp_rag.server --config "$CONFIG" --transport stdio
fi
