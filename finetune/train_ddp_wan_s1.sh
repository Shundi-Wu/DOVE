#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath -m "$SCRIPT_DIR/..")"
PYTHON="${PYTHON:-$PROJECT_ROOT/.venv/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
    echo "Python interpreter not found: $PYTHON" >&2
    exit 1
fi

exec "$PYTHON" "$SCRIPT_DIR/launch.py" \
    --config "$SCRIPT_DIR/configs/experiments/wan_s1.yaml" \
    "$@"
