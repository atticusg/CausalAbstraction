#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=entity-binding-full-pipeline
#SBATCH --output=~/slurm_logs/entity_binding_full_pipeline-%A.log

# Run complete 3-step pipeline for entity binding experiments:
#   Step 1: Generate and filter dataset
#   Step 2: Run interventions (GPU-intensive, saves raw results)
#   Step 3: Compute scores and generate visualizations (CPU-only, fast)
#
# Usage:
#   bash run_all.sh [--model MODEL] [--config CONFIG] [--test]
#   bash run_all.sh --model meta-llama/Llama-3.1-8B-Instruct --config love
#   bash run_all.sh --config action --test  # Quick test mode (layer 0 only, 8 examples)
#
# Or submit as SLURM job:
#   sbatch run_all.sh --model MODEL --config CONFIG

set -e  # Exit on error

# Default values
MODEL="meta-llama/Llama-3.1-8B-Instruct"
CONFIG="love"
TEST_FLAG=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --test)
            TEST_FLAG="--test"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--model MODEL] [--config CONFIG] [--test]"
            echo "  Configs: love, action, positional_entity"
            exit 1
            ;;
    esac
done

cd /mnt/polished-lake/home/atticus/CausalAbstraction

echo "=========================================="
echo "Entity Binding Task - Full Pipeline"
echo "=========================================="
echo "  Model: $MODEL"
echo "  Config: $CONFIG"
echo "  Test mode: ${TEST_FLAG:-no}"
echo "=========================================="
echo ""

# Step 1: Generate and filter dataset
echo "[Step 1/3] Generating and filtering dataset..."
uv run python tasks/entity_binding/experiments/generate_and_filter_dataset.py \
    --config "$CONFIG" \
    --model "$MODEL" \
    $TEST_FLAG

echo "✓ Step 1 complete"
echo ""

# Determine dataset path
if [ -n "$TEST_FLAG" ]; then
    DATASET_PATH="tasks/entity_binding/datasets/${CONFIG}_swap_query_group_test/filtered_dataset"
    RESULTS_PATH="tasks/entity_binding/results/${CONFIG}_test/raw_results.pkl"
else
    DATASET_PATH="tasks/entity_binding/datasets/${CONFIG}_swap_query_group/filtered_dataset"
    RESULTS_PATH="tasks/entity_binding/results/${CONFIG}/raw_results.pkl"
fi

# Step 2: Run interventions
echo "[Step 2/3] Running interventions..."
uv run python tasks/entity_binding/experiments/run_interventions.py \
    --config "$CONFIG" \
    --dataset "$DATASET_PATH" \
    --model "$MODEL" \
    $TEST_FLAG

echo "✓ Step 2 complete"
echo ""

# Step 3: Compute scores and visualize
echo "[Step 3/3] Computing scores and generating visualizations..."
uv run python tasks/entity_binding/experiments/visualize_results.py \
    --results "$RESULTS_PATH"

echo "✓ Step 3 complete"
echo ""

echo "=========================================="
echo "✓ Full pipeline complete!"
echo "=========================================="
echo ""

if [ -n "$TEST_FLAG" ]; then
    echo "Results saved to: tasks/entity_binding/results/${CONFIG}_test/"
else
    echo "Results saved to: tasks/entity_binding/results/${CONFIG}/"
fi
