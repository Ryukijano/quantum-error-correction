#!/usr/bin/env bash
# Launch the Streamlit QEC visualizer inside the syndrome-net (or jax) conda env.
# Usage:
#   ./run_app.sh              # uses 'syndrome-net' env
#   ENV=jax ./run_app.sh      # uses 'jax' env

set -euo pipefail

ENV_NAME="${ENV:-jax}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate via conda run so this script works without sourcing conda init
exec conda run -n "${ENV_NAME}" --no-capture-output \
    streamlit run "${REPO_DIR}/app/streamlit_app.py" \
    --server.headless true \
    --server.port "${PORT:-8501}" \
    "$@"
