#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --job-name=addition-interventions-all
#SBATCH --output=~/slurm_logs/addition_interventions_all-%A.log

# Launch all 9 intervention experiments (step 2 only) in parallel
#
# PREREQUISITES:
#   - Run step 1 first to generate all 9 datasets
#   - Or use generate_all_datasets.sh to create them
#
# EXPERIMENTS: 3 models × 2 digit configs = 6 jobs
#   - Llama-3.1-8B: 2-digit, 3-digit
#   - Gemma-2-9B: 2-digit, 3-digit
#   - OLMo-2-13B: 2-digit, 3-digit
#
# Expected runtime: ~30-60 min per experiment in parallel
# Note: 4-digit experiments excluded (too slow/unreliable)
#
# Usage:
#   sbatch run_all_interventions_parallel.sh [--test]
#   sbatch run_all_interventions_parallel.sh --test  # Test mode: layer 0 only

cd /mnt/polished-lake/home/atticus/CausalAbstraction

# Check for --test flag
TEST_FLAG=""
for arg in "$@"; do
    if [ "$arg" == "--test" ]; then
        TEST_FLAG="--test"
        echo "*** RUNNING IN TEST MODE ***"
        echo ""
    fi
done

echo "=========================================="
echo "Launching 6 Parallel Intervention Jobs"
echo "=========================================="
echo ""

# Create array of model/digit combinations (2 and 3 digits only)
declare -a EXPERIMENTS=(
    "meta-llama/Meta-Llama-3.1-8B-Instruct:2:llama_2d"
    "meta-llama/Meta-Llama-3.1-8B-Instruct:3:llama_3d"
    "google/gemma-2-9b:2:gemma_2d"
    "google/gemma-2-9b:3:gemma_3d"
    "allenai/OLMo-2-1124-13B:2:olmo_2d"
    "allenai/OLMo-2-1124-13B:3:olmo_3d"
)

# Launch each as a separate job
for i in "${!EXPERIMENTS[@]}"; do
    IFS=':' read -r MODEL DIGITS NAME <<< "${EXPERIMENTS[$i]}"
    JOB_NUM=$((i+1))

    JOB_ID=$(sbatch --parsable \
        --nodes=1 \
        --gres=gpu:1 \
        --time=2:00:00 \
        --job-name="add-${NAME}" \
        --output="${HOME}/slurm_logs/add_${NAME}-%j.log" \
        --wrap="cd /mnt/polished-lake/home/atticus/CausalAbstraction && uv run python tasks/general_addition/experiments/02_run_interventions.py --model ${MODEL} --digits ${DIGITS} ${TEST_FLAG}")

    echo "[${JOB_NUM}/6] Submitted ${NAME}: Job ID ${JOB_ID}"
done

echo ""
echo "=========================================="
echo "✓ All 6 jobs submitted!"
echo "=========================================="
echo ""
echo "Monitor with: squeue -u \$USER"
echo "View logs in: ~/slurm_logs/add_*-*.log"
echo ""
echo "After all jobs complete, run step 3 for each result:"
echo "  Example:"
echo "    python 03_compute_scores_and_visualize.py \\"
echo "      --results tasks/general_addition/results/meta_llama_3.1_8b_instruct_2d/full_results/raw_results.pkl"
echo ""
echo "Or use a loop to process all results:"
echo "  for result in tasks/general_addition/results/*/full_results/raw_results.pkl; do"
echo "    python 03_compute_scores_and_visualize.py --results \$result"
echo "  done"
