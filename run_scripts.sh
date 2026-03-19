#!/usr/bin/env bash
# Run any syndrome-net script inside the conda environment.
# Usage:
#   ./run_scripts.sh scripts/plot_threshold.py --quick
#   ENV=jax ./run_scripts.sh scripts/benchmark_decoders.py --quick

set -euo pipefail

ENV_NAME="${ENV:-jax}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ $# -eq 0 ]; then
    echo "Usage: ENV=jax ./run_scripts.sh <script> [args...]"
    exit 1
fi

exec conda run -n "${ENV_NAME}" --no-capture-output \
    python "${REPO_DIR}/$@"
