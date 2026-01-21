#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --job-name=addition-full-pipeline
#SBATCH --output=~/slurm_logs/addition_full_pipeline-%A.log

# Run complete 3-step pipeline for general addition experiments:
#   Step 1: Generate and filter dataset
#   Step 2: Run interventions (GPU-intensive, saves raw results)
#   Step 3: Compute scores and generate visualizations (CPU-only, fast)
#
# Usage:
#   bash run_all.sh [--model MODEL] [--digits D] [--test]
#   bash run_all.sh --model meta-llama/Meta-Llama-3.1-8B-Instruct --digits 2
#   bash run_all.sh --test  # Quick test mode (layer 0 only, 8 examples)
#
# Or submit as SLURM job:
#   sbatch run_all.sh --model MODEL --digits D

set -e  # Exit on error

# Default values
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
DIGITS=2
TEST_FLAG=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --digits)
            DIGITS="$2"
            shift 2
            ;;
        --test)
            TEST_FLAG="--test"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--model MODEL] [--digits D] [--test]"
            exit 1
            ;;
    esac
done

cd /mnt/polished-lake/home/atticus/CausalAbstraction

MODEL_SHORT=$(echo "$MODEL" | awk -F'/' '{print $NF}' | tr '[:upper:]' '[:lower:]' | tr '-' '_')

echo "=========================================="
echo "Addition Task - Full Pipeline"
echo "=========================================="
echo "  Model: $MODEL"
echo "  Digits: $DIGITS"
echo "  Test mode: ${TEST_FLAG:-no}"
echo "=========================================="
echo ""

# Step 1: Generate and filter dataset
echo "[Step 1/3] Generating and filtering dataset..."
uv run python tasks/general_addition/experiments/01_generate_and_filter_dataset.py \
    --model "$MODEL" \
    --digits "$DIGITS" \
    $TEST_FLAG

echo "✓ Step 1 complete"
echo ""

# Step 2: Run interventions
echo "[Step 2/3] Running interventions..."
uv run python tasks/general_addition/experiments/02_run_interventions.py \
    --model "$MODEL" \
    --digits "$DIGITS" \
    $TEST_FLAG

echo "✓ Step 2 complete"
echo ""

# Step 3: Compute scores and visualize
echo "[Step 3/3] Computing scores and generating visualizations..."

if [ -n "$TEST_FLAG" ]; then
    RESULTS_FILE="tasks/general_addition/results/${MODEL_SHORT}_${DIGITS}d/test_results/raw_results.pkl"
else
    RESULTS_FILE="tasks/general_addition/results/${MODEL_SHORT}_${DIGITS}d/full_results/raw_results.pkl"
fi

uv run python tasks/general_addition/experiments/03_compute_scores_and_visualize.py \
    --results "$RESULTS_FILE"

echo "✓ Step 3 complete"
echo ""

echo "=========================================="
echo "✓ Full pipeline complete!"
echo "=========================================="
echo ""
echo "Results saved to: tasks/general_addition/results/${MODEL_SHORT}_${DIGITS}d/"
