#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <checkpoint_path>"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

CHECKPOINT_PATH="$1"

python -m vdt_dev.runner \
  --job eval \
  --checkpoint-path "$CHECKPOINT_PATH"

