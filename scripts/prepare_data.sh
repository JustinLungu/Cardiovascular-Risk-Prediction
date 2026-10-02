#!/usr/bin/env bash
set -euo pipefail

# Resolve the repository from this script so it works from any directory.
repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv is required. Follow the setup instructions in README.md." >&2
    exit 1
fi

uv run --locked --no-dev python src/download_data.py
uv run --locked --no-dev python src/extract_data.py
