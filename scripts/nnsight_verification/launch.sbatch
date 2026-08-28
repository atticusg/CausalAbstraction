#!/bin/bash
#SBATCH --job-name=nnsight08-verify
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=/mnt/home/%u/slurm_logs/%x_%j.out
# N0 of the nnsight-engine plan: run NDIF's clq.py (verbatim) on our hardware.
# Policy: no --partition, no custom cpu/mem on a GPU job, --time set.
set -euo pipefail

source /mnt/data/scripts/config.sh
export HF_HOME=/mnt/data/artifacts/hf_cache

VENV="$HOME/.venvs/nnsight08-verify"
[ -x "$VENV/bin/python" ] || uv venv --python 3.12 "$VENV"
# Pins mirror causalab's uv.lock (torch 2.9.0, transformers 5.16.1) plus the
# plan's nnsight rev (causalab-nnsight-engine-plan §0 decision 1).
uv pip install --python "$VENV/bin/python" \
    torch==2.9.0 transformers==5.16.1 accelerate \
    "nnsight @ git+https://github.com/ndif-team/nnsight.git@8c480727"

cd "$SLURM_SUBMIT_DIR"
"$VENV/bin/python" scripts/nnsight_verification/clq.py
