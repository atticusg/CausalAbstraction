# General Addition Experiments - Refactored Pipeline

This directory contains scripts for running interchange intervention experiments on the general addition task, using the **new refactored API** that separates intervention execution from causal model scoring.

## Overview

The pipeline is organized into **3 separate steps**:

1. **Dataset Generation & Filtering** - Generate counterfactual pairs and filter for model capability
2. **Intervention Execution** - Run expensive neural network interventions (GPU-intensive)
3. **Score Computation & Visualization** - Compute causal scores and generate plots (CPU-light)

This separation allows you to:
- ✅ Run interventions once, analyze with different target variables many times
- ✅ Experiment with different causal model configurations post-hoc
- ✅ Skip dataset generation if you already have filtered data
- ✅ Run interventions in parallel across multiple models/configurations

---

## Quick Start

### Run Complete Pipeline (All 3 Steps)

```bash
# Run for a specific model and digit configuration
bash run_all.sh --model meta-llama/Meta-Llama-3.1-8B-Instruct --digits 2

# Test mode (faster, small dataset)
bash run_all.sh --model meta-llama/Meta-Llama-3.1-8B-Instruct --digits 2 --test

# Submit as SLURM job
sbatch run_all.sh --model meta-llama/Meta-Llama-3.1-8B-Instruct --digits 3
```

### Run Steps Individually

```bash
# Step 1: Generate and filter dataset (run once per model/digit combo)
python 01_generate_and_filter_datasets.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --digits 2

# Step 2: Run interventions (expensive GPU step)
python 02_run_interventions.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --digits 2

# Step 3: Compute scores and visualize (can run multiple times with different target variables!)
python 03_compute_scores_and_visualize.py --results tasks/general_addition/results/meta_llama_3.1_8b_instruct_2d/full_results/raw_results.pkl
```

### Run All 9 Model/Digit Combinations in Parallel

```bash
# Launch 9 parallel SLURM jobs for interventions (assumes datasets already generated)
sbatch run_all_interventions_parallel.sh

# Test mode
sbatch run_all_interventions_parallel.sh --test
```

---

## Script Details

### Step 1: `01_generate_and_filter_dataset.py`

**Purpose:** Generate counterfactual dataset and filter for examples where the model performs correctly.

**Key Arguments:**
- `--model`: Model name (e.g., `meta-llama/Meta-Llama-3.1-8B-Instruct`)
- `--digits`: Number of digits (2, 3, or 4)
- `--size`: Number of counterfactual pairs (default: 256)
- `--test`: Test mode (8 examples)

**Outputs:**
- `tasks/general_addition/datasets/random_cf_{model}_{digits}d/filtered_dataset/`
- `tasks/general_addition/datasets/random_cf_{model}_{digits}d/dataset_metadata.json`

**Example:**
```bash
python 01_generate_and_filter_dataset.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --digits 2 \
    --size 256
```

---

### Step 2: `02_run_interventions.py`

**Purpose:** Run interchange interventions across all layers and token positions. Saves **raw results** without computing causal scores yet.

**Note:** This script ALWAYS saves raw-only results. Scoring is done in step 3. This separation allows expensive GPU interventions to run separately from cheap CPU scoring.

**Key Arguments:**
- `--model`: Model name
- `--digits`: Number of digits (2, 3, or 4)
- `--dataset`: Path to filtered dataset (auto-detected if not provided)
- `--batch-size`: Batch size for interventions (default: 512)
- `--test`: Test mode (layer 0 only, 1 batch)

**Outputs:**
- `tasks/general_addition/results/{model}_{digits}d/full_results/raw_results.pkl`
- `tasks/general_addition/results/{model}_{digits}d/full_results/experiment_metadata.json`

**Example:**
```bash
python 02_run_interventions.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --digits 2 \
    --batch-size 512
```

**Note:** This is the expensive GPU step. Run once, then use step 3 multiple times to analyze different target variables.

**Why separate steps?** Interventions are GPU-intensive and slow. Scoring is CPU-only and fast. This separation lets you run interventions once and experiment with different target variables many times!

---

### Step 3: `03_compute_scores_and_visualize.py`

**Purpose:** Load raw intervention results, compute interchange scores for target variables, and generate visualizations.

**Key Arguments:**
- `--results`: Path to `raw_results.pkl` from step 2 (required)
- `--target-vars`: Specific variables to analyze (default: all digit variables + raw_output)
- `--include-pairs`: Also analyze all pairs of digit variables (slow)

**Outputs:**
- `heatmap_{variable}.png` - Heatmap visualizations
- `analysis_{variable}.txt` - Text analysis of results

**Example:**
```bash
# Analyze all default variables
python 03_compute_scores_and_visualize.py \
    --results tasks/general_addition/results/meta_llama_3.1_8b_instruct_2d/full_results/raw_results.pkl

# Analyze specific variables only
python 03_compute_scores_and_visualize.py \
    --results tasks/general_addition/results/meta_llama_3.1_8b_instruct_2d/full_results/raw_results.pkl \
    --target-vars digit_0_0 digit_0_1 raw_output

# Include all pairs of variables (can be slow for many digits)
python 03_compute_scores_and_visualize.py \
    --results tasks/general_addition/results/meta_llama_3.1_8b_instruct_2d/full_results/raw_results.pkl \
    --include-pairs
```

**Note:** You can run this step multiple times with different `--target-vars` without re-running the expensive interventions!

---

## Key API Changes (Migration from Old Code)

The scripts follow the **new refactored API** from the migration guide:

### Old Way (Single Step)
```python
# Old: Everything in one call
experiment = PatchResidualStream(pipeline, causal_model, layers, positions, checker, config)
results = experiment.perform_interventions(datasets, target_variables_list=[["answer"]])
experiment.plot_heatmaps(results, target_variables=["answer"])
```

### New Way (Two Steps)
```python
from causalab.causal.causal_utils import compute_interchange_scores

# Step 1: Run interventions (no causal model, no target variables)
config["id"] = "my_experiment"
experiment = PatchResidualStream(pipeline, layers, positions, config=config)
raw_results = experiment.perform_interventions(datasets)

# Step 2: Compute scores (can repeat with different target variables!)
results = compute_interchange_scores(
    raw_results, causal_model, datasets,
    target_variables_list=[["answer"]], checker=checker
)

# Step 3: Visualize
experiment.plot_heatmaps(results, target_variables=["answer"])
```

**Benefits:**
- Run expensive interventions once
- Analyze with different target variables many times (cheap!)
- Cleaner separation of concerns

---

## Directory Structure

```
tasks/general_addition/
├── experiments/
│   ├── 01_generate_and_filter_dataset.py   # Step 1: Dataset generation
│   ├── 02_run_interventions.py             # Step 2: Run interventions
│   ├── 03_compute_scores_and_visualize.py  # Step 3: Scoring & viz
│   ├── run_all.sh                          # Run complete pipeline
│   ├── run_all_interventions_parallel.sh   # Parallel intervention jobs
│   ├── tokenization_config.py              # Model-specific tokenization
│   └── README.md                            # This file
├── datasets/                                # Generated datasets
│   └── random_cf_{model}_{digits}d/
│       ├── filtered_dataset/
│       └── dataset_metadata.json
├── results/                                 # Experiment results
│   └── {model}_{digits}d/
│       └── full_results/
│           ├── raw_results.pkl
│           ├── experiment_metadata.json
│           ├── heatmap_*.png
│           └── analysis_*.txt
├── notebooks/                               # Jupyter notebooks
│   ├── 01_addition_task_and_causal_models.ipynb
│   ├── 02_token_positions_for_digits.ipynb
│   └── 03_counterfactuals_and_carry_variables.ipynb
└── [other files...]
```

---

## Supported Models

- **Llama 3.1 8B**: `meta-llama/Meta-Llama-3.1-8B-Instruct`
- **Gemma 2 9B**: `google/gemma-2-9b`
- **OLMo 2 13B**: `allenai/OLMo-2-1124-13B`

## Supported Digit Configurations

- **2 digits**: 10-99 (e.g., 23 + 45)
- **3 digits**: 100-999 (e.g., 234 + 567)
- **4 digits**: 1000-9999 (e.g., 2345 + 6789)

---

## Typical Workflow

### Single Model/Digit Configuration

```bash
# 1. Generate dataset (run once)
python 01_generate_and_filter_datasets.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --digits 2

# 2. Run interventions (expensive, run once)
python 02_run_interventions.py --model meta-llama/Meta-Llama-3.1-8B-Instruct --digits 2

# 3. Analyze with different target variables (cheap, run many times!)
python 03_compute_scores_and_visualize.py --results tasks/general_addition/results/meta_llama_3.1_8b_instruct_2d/full_results/raw_results.pkl --target-vars digit_0_0 digit_0_1

python 03_compute_scores_and_visualize.py --results tasks/general_addition/results/meta_llama_3.1_8b_instruct_2d/full_results/raw_results.pkl --target-vars raw_output

python 03_compute_scores_and_visualize.py --results tasks/general_addition/results/meta_llama_3.1_8b_instruct_2d/full_results/raw_results.pkl --include-pairs
```

### All 9 Configurations

```bash
# Option 1: Sequential (slow but simple)
for MODEL in "meta-llama/Meta-Llama-3.1-8B-Instruct" "google/gemma-2-9b" "allenai/OLMo-2-1124-13B"; do
    for DIGITS in 2 3 4; do
        bash run_all.sh --model $MODEL --digits $DIGITS
    done
done

# Option 2: Parallel (fast but requires SLURM)
sbatch run_all_interventions_parallel.sh
```

---

## Common Issues

### "Dataset not found"

Make sure you've run step 1 first:
```bash
python 01_generate_and_filter_dataset.py --model YOUR_MODEL --digits YOUR_DIGITS
```

### "Raw results not found"

Make sure you've run step 2 first:
```bash
python 02_run_interventions.py --model YOUR_MODEL --digits YOUR_DIGITS
```

### Out of memory

Reduce batch size:
```bash
python 02_run_interventions.py --model YOUR_MODEL --digits YOUR_DIGITS --batch-size 256
```

### Want to test quickly

Use test mode:
```bash
bash run_all.sh --model YOUR_MODEL --digits YOUR_DIGITS --test
```

---

## Migration Notes

If you have old code using `full_vector_residual_patching.py`, see the main `MIGRATION_GUIDE.md` in the repository root for detailed migration instructions.

**Key changes:**
- Experiment constructors no longer take `causal_model` or `checker`
- Config must include `"id"` field
- `perform_interventions()` no longer takes `target_variables_list`
- Use `compute_interchange_scores()` to add scoring post-hoc
- Scoring is done with `checker` in `compute_interchange_scores()`, not constructor

---

## See Also

- Main repository `MIGRATION_GUIDE.md` - Complete API migration guide
- `demos/onboarding_tutorial/` - Tutorial notebooks with new API examples
- `causal/causal_utils.py::compute_interchange_scores()` - Scoring function documentation
