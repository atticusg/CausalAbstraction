#!/bin/bash
# Run the full causalab pipeline: baseline → subspace → activation_manifold → evaluate → pullback
#
# Usage:
#   ./run_pipeline.sh +runner=weekdays_8b
#   ./run_pipeline.sh +runner=grid_5x5_8b analysis.batch_size=16
#
# Automatically finds a free GPU if available.
# Cleans up child processes on exit, Ctrl+C, or Ctrl+Z.

set -e

cleanup() {
    trap - EXIT INT TERM TSTP
    echo ""
    echo "Cleaning up child processes..."
    kill -- -$$ 2>/dev/null
}
trap cleanup EXIT INT TERM TSTP

# Find a GPU with no processes running on it
FREE_GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' '$2 < 100 {print $1; exit}')

if [ -n "$FREE_GPU" ]; then
    echo "Using free GPU: $FREE_GPU"
    export CUDA_VISIBLE_DEVICES=$FREE_GPU
else
    echo "No free GPU found, using default"
fi

for ANALYSIS in baseline subspace activation_manifold evaluate pullback; do
    echo "=== $ANALYSIS ==="
    uv run python -m causalab.runner.run_exp analysis=$ANALYSIS "$@"
done
