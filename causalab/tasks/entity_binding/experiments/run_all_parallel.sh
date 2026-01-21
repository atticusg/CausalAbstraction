#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --job-name=entity-binding-all-parallel
#SBATCH --output=~/slurm_logs/entity_binding_all_parallel-%A.log

# Launch all entity binding configurations in parallel as separate SLURM jobs:
#   - love (2 entity groups)
#   - action (3 entity groups)
#   - positional_entity (arrow syntax experiments)
#
# PREREQUISITES:
#   - Datasets will be generated automatically by each job
#
# Expected runtime: ~30-90 min per config in parallel
#
# Usage:
#   sbatch run_all_parallel.sh [--model MODEL] [--test]
#   sbatch run_all_parallel.sh --model meta-llama/Llama-3.1-8B-Instruct
#   sbatch run_all_parallel.sh --test  # Test mode: layer 0 only

cd /mnt/polished-lake/home/atticus/CausalAbstraction

# Default values
MODEL="meta-llama/Llama-3.1-8B-Instruct"
TEST_FLAG=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --test)
            TEST_FLAG="--test"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--model MODEL] [--test]"
            exit 1
            ;;
    esac
done

if [ -n "$TEST_FLAG" ]; then
    echo "*** RUNNING IN TEST MODE ***"
    echo ""
fi

echo "=========================================="
echo "Launching 3 Parallel Entity Binding Jobs"
echo "=========================================="
echo "  Model: $MODEL"
echo "  Test mode: ${TEST_FLAG:-no}"
echo "=========================================="
echo ""

# Array of configurations
declare -a CONFIGS=("love" "action" "positional_entity")

# Launch each as a separate SLURM job
for i in "${!CONFIGS[@]}"; do
    CONFIG="${CONFIGS[$i]}"
    JOB_NUM=$((i+1))

    JOB_ID=$(sbatch --parsable \
        --nodes=1 \
        --gres=gpu:1 \
        --time=4:00:00 \
        --job-name="eb-${CONFIG}" \
        --output="${HOME}/slurm_logs/eb_${CONFIG}-%j.log" \
        --wrap="bash /mnt/polished-lake/home/atticus/CausalAbstraction/tasks/entity_binding/experiments/run_all.sh --model ${MODEL} --config ${CONFIG} ${TEST_FLAG}")

    echo "[${JOB_NUM}/3] Submitted ${CONFIG}: Job ID ${JOB_ID}"
done

echo ""
echo "=========================================="
echo "✓ All 3 jobs submitted!"
echo "=========================================="
echo ""
echo "Monitor with: squeue -u \$USER"
echo "View logs in: ~/slurm_logs/eb_*-*.log"
echo ""
echo "After all jobs complete, run step 3 for each result:"
echo "  Example:"
if [ -n "$TEST_FLAG" ]; then
    echo "    python visualize_results.py --results tasks/entity_binding/results/love_test/raw_results.pkl"
else
    echo "    python visualize_results.py --results tasks/entity_binding/results/love/raw_results.pkl"
fi
echo ""
echo "Or use a loop to process all results:"
if [ -n "$TEST_FLAG" ]; then
    echo "  for config in love action positional_entity; do"
    echo "    python visualize_results.py --results tasks/entity_binding/results/\${config}_test/raw_results.pkl"
    echo "  done"
else
    echo "  for config in love action positional_entity; do"
    echo "    python visualize_results.py --results tasks/entity_binding/results/\${config}/raw_results.pkl"
    echo "  done"
fi
