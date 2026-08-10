#!/usr/bin/env bash
# Build, smoke-test, and export the Chipzen upload tarball.
# Run from anywhere: paths resolve relative to this script.
set -euo pipefail

TAG="${1:-playground-bot:v1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUT="$SCRIPT_DIR/playground-bot.tar.gz"

command -v docker >/dev/null 2>&1 || {
    echo "error: docker not found on PATH. Install Docker Desktop (or docker CE in WSL) first." >&2
    exit 1
}

WEIGHTS="$REPO_ROOT/artifacts/tiny-policy-pure.json"
if [[ ! -f "$WEIGHTS" ]]; then
    echo "error: $WEIGHTS missing." >&2
    echo "Export it first (needs a Python with torch, e.g. the poker-nn-training venv):" >&2
    echo "  python tools/export_pure_weights.py" >&2
    exit 1
fi

echo "==> docker build ($TAG)"
docker build --platform linux/amd64 -f "$SCRIPT_DIR/Dockerfile" -t "$TAG" "$REPO_ROOT"

echo "==> smoke test: construct the bot inside the image"
docker run --rm --entrypoint python "$TAG" -u -c \
    "from bot import PlaygroundChipzenBot; PlaygroundChipzenBot(); print('bot constructs and weights load OK')"

IMAGE_BYTES="$(docker image inspect "$TAG" --format '{{.Size}}')"
IMAGE_MB=$((IMAGE_BYTES / 1024 / 1024))
echo "==> image size: ${IMAGE_MB} MB (platform cap: 200 MB)"
if (( IMAGE_MB > 200 )); then
    echo "error: image exceeds the 200 MB platform cap" >&2
    exit 1
fi

echo "==> docker save | gzip -> $OUT"
docker save "$TAG" | gzip > "$OUT"
ARCHIVE_MB=$(( $(stat -c %s "$OUT" 2>/dev/null || stat -f %z "$OUT") / 1024 / 1024 ))
echo "==> upload artifact: $OUT (${ARCHIVE_MB} MB compressed, cap: 250 MB)"
if (( ARCHIVE_MB > 250 )); then
    echo "error: archive exceeds the 250 MB upload cap" >&2
    exit 1
fi

echo "Done. Upload $OUT through the Chipzen developer UI."
