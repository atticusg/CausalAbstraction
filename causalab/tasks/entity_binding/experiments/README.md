# Entity Binding Experiments - Consolidated Scripts

This directory contains three scripts for running entity binding experiments following the new API patterns. The workflow is similar to general_addition: clear separation between intervention (GPU-intensive) and post-processing (CPU-only).

## Quick Start

```bash
# Complete 3-step pipeline (test mode)
bash run_all.sh --config love --test

# Or run steps individually:
python generate_and_filter_dataset.py --config love --test
python run_interventions.py --config love --dataset datasets/love_swap_query_group_test/filtered_dataset --test
python visualize_results.py --results results/love_test/raw_results.pkl
```

## Scripts Overview

### Step 1: generate_and_filter_dataset.py

**Purpose**: Generate counterfactual datasets and filter based on model performance.

**What it does**:
- Generates counterfactual pairs
- Loads language model
- Filters to keep only correct predictions
- Saves BOTH original and filtered datasets

**Test Mode**: `--test` → 8 examples, batch_size=8

**Example**:
```bash
python generate_and_filter_dataset.py \
    --config love \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --size 128

# Test mode
python generate_and_filter_dataset.py --config love --test
```

**Outputs**:
```
datasets/love_swap_query_group/
├── original_dataset/          # Full unfiltered dataset
├── filtered_dataset/          # Filtered dataset
└── dataset_metadata.json      # Statistics
```

### Step 2: run_interventions.py

**Purpose**: Run interventions and save raw results (NO scoring/visualization).

**What it does**:
- Loads filtered dataset
- Loads language model
- Creates token positions
- Runs interventions across layers
- Saves raw results ONLY

**Test Mode**: `--test` → layer 0 only, 8 examples, batch_size=8

**Example**:
```bash
python run_interventions.py \
    --config love \
    --dataset datasets/love_swap_query_group/filtered_dataset \
    --model meta-llama/Llama-3.1-8B-Instruct

# Test mode (layer 0 only, fast)
python run_interventions.py \
    --config love \
    --dataset datasets/love_test/filtered_dataset \
    --test

# Custom layer range
python run_interventions.py \
    --config love \
    --dataset datasets/love/filtered_dataset \
    --layers 0:16  # Layers 0-15
```

**Outputs**:
```
results/love/
├── raw_results.pkl            # Raw intervention results (reusable!)
└── experiment_metadata.json   # Includes dataset_path for step 3
```

### Step 3: visualize_results.py

**Purpose**: Compute scores and generate visualizations from raw results.

**What it does**:
- Loads raw_results.pkl
- Loads causal model and dataset (from metadata)
- Computes interchange scores
- Generates heatmaps
- Generates text analysis

**Can be run multiple times with different target variables!**

**Test Mode**: `--test` → (for compatibility, no special behavior)

**Example**:
```bash
# Use default target variables
python visualize_results.py \
    --results results/love/raw_results.pkl

# Custom target variables
python visualize_results.py \
    --results results/love/raw_results.pkl \
    --target-vars query_entity positional_query_group

# Just one variable
python visualize_results.py \
    --results results/love/raw_results.pkl \
    --target-vars query_entity

# Custom output directory
python visualize_results.py \
    --results results/love/raw_results.pkl \
    --target-vars query_entity \
    --output results/love_query_entity_only
```

**Outputs**:
```
results/love/                  # Or custom output directory
├── results_with_scores.pkl   # Scored results
├── heatmaps/                 # Generated heatmaps
│   ├── query_entity/
│   ├── positional_query_group/
│   └── raw_output/
└── analysis/                 # Text analysis
    ├── query_entity.txt
    ├── positional_query_group.txt
    └── raw_output.txt
```

## Complete Workflows

### Workflow 1: Quick Test Run (2-5 minutes)

```bash
# Step 1: Generate test dataset (8 examples)
python generate_and_filter_dataset.py --config love --test

# Step 2: Run interventions (layer 0 only)
python run_interventions.py \
    --config love \
    --dataset datasets/love_swap_query_group_test/filtered_dataset \
    --test

# Step 3: Visualize
python visualize_results.py --results results/love_test/raw_results.pkl
```

### Workflow 2: Production Run (Full)

```bash
# Step 1: Generate dataset (128 examples)
python generate_and_filter_dataset.py \
    --config love \
    --size 128

# Step 2: Run interventions (all layers)
python run_interventions.py \
    --config love \
    --dataset datasets/love_swap_query_group/filtered_dataset

# Step 3: Visualize
python visualize_results.py --results results/love/raw_results.pkl
```

### Workflow 3: Exploratory Analysis (Run Once, Analyze Many Ways)

```bash
# Run steps 1 & 2 once
python generate_and_filter_dataset.py --config love --size 128
python run_interventions.py --config love --dataset datasets/love/filtered_dataset

# Analyze with different target variables (fast! no re-running interventions)
python visualize_results.py \
    --results results/love/raw_results.pkl \
    --target-vars query_entity \
    --output results/love_query_only

python visualize_results.py \
    --results results/love/raw_results.pkl \
    --target-vars positional_query_group \
    --output results/love_positional_only

python visualize_results.py \
    --results results/love/raw_results.pkl \
    --target-vars raw_output \
    --output results/love_raw_only
```

### Workflow 4: Using Bash Script (Easiest)

```bash
# Test mode (complete pipeline in 2-5 minutes)
bash run_all.sh --config love --test

# Production mode
bash run_all.sh --config love

# All configs in parallel (SLURM)
sbatch run_all_parallel.sh
```

## Configuration Options

### Task Configurations (`--config`)

| Config | Groups | Template | Default Target Variables |
|--------|--------|----------|-------------------------|
| `love` | 2 | "X loves Y" | `positional_query_group`, `query_entity`, `raw_output` |
| `action` | 3 | "X put Y in Z" | `positional_query_group`, `query_entity`, `raw_output` |
| `positional_entity` | 2 (fixed query) | "X loves Y" | Arrow syntax swaps (e.g., `positional_entity_g0_e1<-positional_entity_g1_e1`) |

### Token Position Types (`--token-positions`)

- **`last_token`** (default): Last token of the sequence
- **`both_e1`**: Both e1 entity tokens (for positional_entity experiments)

### Experiment Types (`--experiment-type`)

- **`residual_stream`** (default): PatchResidualStream
- **`attention_heads`**: Not yet implemented

## Bash Scripts

### run_all.sh - Complete 3-Step Pipeline

Runs all three steps automatically:

```bash
bash run_all.sh --config love              # Full run
bash run_all.sh --config action --test     # Test mode
sbatch run_all.sh --config love            # SLURM submission
```

### run_all_parallel.sh - All Configs in Parallel

Launches separate SLURM jobs for love, action, and positional_entity:

```bash
sbatch run_all_parallel.sh                 # Launch all
sbatch run_all_parallel.sh --test          # Test mode
squeue -u $USER                             # Monitor
```

**Note**: After parallel jobs complete, run step 3 manually for each result (see script output for commands).

## Test Mode Details

All scripts support `--test` flag:

| Script | Test Mode Behavior |
|--------|-------------------|
| **Step 1** | 8 examples, batch_size=8 |
| **Step 2** | Layer 0 only, 8 examples, batch_size=8 |
| **Step 3** | Process all available results (no filtering) |

**Use test mode for**:
- Quick sanity checks (~2-5 minutes total)
- Debugging before long runs
- Verifying new configurations

## Architecture Benefits

### Clean Separation of Concerns

```
Step 1: Dataset Generation        (One-time)
   ↓
Step 2: Run Interventions         (GPU-intensive, expensive)
   ↓
Step 3: Compute Scores & Visualize (CPU-only, cheap, can repeat!)
```

### Why This Design?

1. **Run interventions once, analyze many ways**
   - Step 2 is expensive (GPU, hours)
   - Step 3 is cheap (CPU, minutes)
   - Run step 3 multiple times with different target variables

2. **No duplication**
   - All scoring/viz logic in ONE place (step 3)
   - Easier to maintain and extend

3. **Flexibility**
   - Run steps separately or together
   - Rerun visualization with new variables
   - Output to different directories

4. **Matches general_addition**
   - Same 3-step workflow
   - Consistent across tasks

## Directory Structure

```
tasks/entity_binding/
├── experiment_config.py           # Task plugin interface
├── experiments/
│   ├── README.md                  # This file
│   ├── generate_and_filter_dataset.py   # Step 1
│   ├── run_interventions.py             # Step 2
│   ├── visualize_results.py             # Step 3
│   ├── run_all.sh                       # Bash automation
│   └── run_all_parallel.sh              # Parallel bash automation
├── datasets/                      # Generated by step 1
│   └── {config}_swap_query_group/
│       ├── original_dataset/
│       ├── filtered_dataset/
│       └── dataset_metadata.json
└── results/                       # Generated by steps 2 & 3
    └── {config}/
        ├── raw_results.pkl        # Step 2 output (reusable!)
        ├── experiment_metadata.json
        ├── results_with_scores.pkl     # Step 3 output
        ├── heatmaps/                   # Step 3 output
        │   ├── query_entity/
        │   └── ...
        └── analysis/                   # Step 3 output
            ├── query_entity.txt
            └── ...
```

## Task Plugin Interface

The [experiment_config.py](../experiment_config.py) module provides:

- `get_task_config(config_name)` - Get task configuration
- `get_causal_model(config, model_type)` - Get causal model
- `get_counterfactual_generator(cf_name, config)` - Get counterfactual generator
- `get_token_positions(pipeline, config, position_type)` - Get token positions
- `get_checker()` - Get output checker function
- `get_default_target_variables(config_name)` - Get default target variables
- `get_pipeline_config(config_name)` - Get pipeline parameters

Add new configurations without modifying core scripts!

## Advanced Usage

### Different Models

```bash
# Llama
bash run_all.sh --model meta-llama/Llama-3.1-8B-Instruct --config love

# Gemma
bash run_all.sh --model google/gemma-2-9b --config love

# Any HuggingFace model
bash run_all.sh --model organization/model-name --config love
```

### Positional Entity Experiments (Arrow Syntax)

```bash
# Generate dataset
python generate_and_filter_dataset.py --config positional_entity --size 128

# Run interventions (uses special token positions)
python run_interventions.py \
    --config positional_entity \
    --dataset datasets/positional_entity_swap_query_group/filtered_dataset \
    --token-positions both_e1

# Visualize (uses arrow syntax: g0_e1 <- g1_e1)
python visualize_results.py --results results/positional_entity/raw_results.pkl
```

### Custom Layer Ranges

```bash
# Only layers 0-7
python run_interventions.py --config love --dataset ... --layers 0:8

# Only layers 10-20
python run_interventions.py --config love --dataset ... --layers 10:21

# Input + first 5 layers
python run_interventions.py --config love --dataset ... --layers -1:5
```

## Troubleshooting

### No examples passed filtering

```
⚠ WARNING: No examples passed filtering! Model may not be capable of this task.
```

**Solutions**:
- Try a different model
- Adjust the checker function in experiment_config.py
- Check if prompt formatting is correct

### Out of memory

**Solutions**:
- Use `--test` mode
- Reduce `--batch-size`
- Use smaller `--size` for dataset
- Limit `--layers 0:8`

### Missing dataset_path error in step 3

```
Error: Dataset path not found in original experiment metadata
```

**Solution**: This should not happen with the new scripts. If it does, the experiment metadata is corrupted. Rerun step 2.

### Custom target variables not recognized

Check available variables in the causal model:
```bash
python -c "
from causalab.tasks.entity_binding.experiment_config import get_task_config, get_causal_model
config = get_task_config('love')
model = get_causal_model(config)
print(model.variables.keys())
"
```

## Comparison with Old Scripts

| Old Scripts (7 files) | New Scripts (3 files) |
|----------------------|----------------------|
| 01_generate_dataset.py | generate_and_filter_dataset.py --config love |
| 03_generate_dataset_action.py | generate_and_filter_dataset.py --config action |
| 02_run_interventions.py | run_interventions.py --config love |
| 04_run_interventions_action.py | run_interventions.py --config action |
| 05_run_interventions_swapped_positions.py | run_interventions.py --config positional_entity --token-positions both_e1 |
| visualize_results.py (old) | visualize_results.py (new) |
| generate_heatmaps.py | Merged into visualize_results.py |

## Benefits of New Architecture

1. **DRY Principle**: 3 scripts instead of 7+
2. **Clear Separation**: GPU work (step 2) vs CPU work (step 3)
3. **Reusable Results**: Run interventions once, visualize many ways
4. **Test Mode**: Fast iteration with `--test` flag
5. **Complete Data**: Original datasets saved
6. **Consistent**: Matches general_addition workflow
7. **No Duplication**: All scoring/viz logic in one place

## See Also

- [MIGRATION_GUIDE.md](../../../MIGRATION_GUIDE.md) - API migration guide for the entire codebase
- [experiment_config.py](../experiment_config.py) - Task plugin interface
- [tasks/general_addition/experiments/](../../general_addition/experiments/) - Same 3-step pattern
