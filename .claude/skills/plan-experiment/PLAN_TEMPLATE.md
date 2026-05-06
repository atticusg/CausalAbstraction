# Plan Template

Filled by `/plan-experiment`. Lands at `${SESSION_DIR}/plan/PLAN.md`. Read by `/run-experiment` (executes the plan) and `/interpret-experiment` (uses it as the interpretation lens).

This template covers **§B–F — implementation detail**. The objective, motivation, success criteria, and hypotheses live in the sibling `RESEARCH_OBJECTIVE.md`.

> Granularity rule: every section below is required, but each entry's depth adapts to the plan. If method choice *is* the research question, document at the method level. Otherwise stay at the analysis level. Optional fields are marked.

---

## §B. Causal model & dataset

This section is structured so it can be lifted verbatim into a `set_up_task.md` spec if `/setup-task` runs later (see `/plan-experiment instructions/characterize_task.md`).

**Task:** `{task_name}` — package at `causalab/tasks/{task_name}/`
- **Status:** `exists` / `to be created (autonomous /setup-task input below)`
- **Spec path** (only when status is `to be created`): `${SESSION_DIR}/plan/setup_task_{task_name}.md`. `/plan-experiment` lifts §B verbatim into this companion file at plan-time; `/run-experiment` Step 2 passes the path to `/setup-task`.

### Causal variables

| Name | Type (categorical / ordinal / numeric) | Cardinality | Sketch of value space |
|---|---|---|---|
| `var_1` | … | … | e.g. weekday name strings, 7 values: `Mon, Tue, …` |
| `var_2` | … | … | … |

### Mechanism summary

One line per derived variable. Function from parents to value.

- `output = f(var_1, var_2)` — *e.g. add `var_2` days to `var_1` modulo 7*

### Expected behavior

3–5 golden-path input → output examples the model **must** get right for the plan to be meaningful.

```
Input:  "If today is Monday, three days from now is"
Output: "Thursday"
```

### Edge / stress cases

Boundary inputs we expect to be tricky, with the predicted failure mode and what each tells us.

- `Sunday + 1` → `Monday` (week-wrap) — *if model fails, it lacks modular arithmetic*
- max-arity prompt → … — *…*

### Counterfactual generator

| Variable | `task.resample_variable` | Why |
|---|---|---|
| `var_1` | `var_1` (single-variable CFs) | required for `locate.mode: pairwise` on this variable |
| (or) all | `"all"` | only `centroid` mode is needed |

See `ARCHITECTURE.md` §5 for the `resample_variable` × `locate.mode` rule.

### Dataset sizing

```yaml
task:
  n_train: 200
  n_test: 100
  enumerate_all: false
  balanced: false
```

Rationale: …

---

## §C. Neural surface

### Model(s)

| Model | Config | Why |
|---|---|---|
| `meta-llama/Meta-Llama-3.1-8B` | `model: llama31_8b` | full sweep on a model that fits MBP MPS |
| (additional models) | … | … |

### Tokenization-check predictions

What we expect each variable to tokenize to in this model, and which checks we'll run during `/setup-task` Step 4 / `/run-experiment` Step 2.

- Each weekday should be a single token under `meta-llama/Meta-Llama-3.1-8B`. Check via test 3 (token alignment) before running interventions.
- *…*

### Compute budget

| Phase | Where | Expected wall time | GPUs |
|---|---|---|---|
| baseline | inline / slurm | … | … |
| locate | … | … | … |
| (per chained analysis) | … | … | … |

### Hardware constraints

State the binding constraints surfaced by the user's environment (e.g. MBP unified memory ceiling, slurm queue, GPU count). Plans that exceed these must say so explicitly so `/run-experiment` can prompt for a smaller alternative.

- **Local:** MBP 48 GB unified memory; MPS backend required for GPU; > 8B models do not fit.
- **Cluster:** slurm; default `--gres=gpu:1`, `--time=04:00:00`; opportunistic QoS available.

---

## §D. Analysis-chain DAG

The heart of the plan. Walk the dependency DAG from `ANALYSIS_GUIDE.md` and instantiate one node per chosen analysis.

### DAG diagram

```
baseline ──► locate ──► subspace ──► activation_manifold ──► path_steering
                                                       │
                                                       └──► pullback ◄── output_manifold ◄── baseline
```

(Trim the diagram to the nodes actually in this plan.)

### Per-node detail

For each node, fill the card below:

#### Node N: `{analysis_name}` (or `custom — out of scope, planned but not implemented`)

- **Research question (scoped):** *What does this node tell us, in one sentence?*
- **Method:** `{interchange / pca / das / spline / …}` — only document method-level choice when method choice **is** the research question.
- **Upstream artifacts consumed:** files this node reads (file paths under `${experiment_root}/...`). Use `null` for auto-discovered inputs and note which auto-discovery rule applies (per `ANALYSIS_GUIDE.md` "Auto-Discovery").
- **Downstream artifacts produced:** files this node writes (file paths under `${experiment_root}/...`).
- **Non-default knobs:** only the deltas vs `causalab/configs/analysis/{analysis_name}.yaml`. Each with a one-line rationale.
- **Pre-flight check:** the minimal finding required in this node's outputs before any dependent node should run. Acts as a gate.
  - Example: `locate` — must yield at least one cell with `KL_drop ≥ 0.3`. If not, **stop the chain** and revisit the task setup or layer range.
- **Spec path** (custom nodes only): path under `${SESSION_DIR}/plan/` to the markdown spec consumed by `/setup-methods` or `/setup-analyses`. `/plan-experiment` writes this companion spec at plan-time; `/run-experiment` Step 4 collects all such paths into per-skill batches and invokes each setup skill once.
  - For method nodes: `${SESSION_DIR}/plan/setup_method_{name}.md`
  - For analysis nodes: `${SESSION_DIR}/plan/setup_analysis_{name}.md`
- **Estimated runtime + GPU footprint:** `~X min on Y GPU`.

> Repeat the card for each node.

### Cross-analysis post-steps

Anything in the runner config's `post:` block (per `ARCHITECTURE.md` §2 "Execution model"):

- `variable_localization_heatmap` over `locate` outputs — *if more than one target variable is localized*

---

## §E. Risk register & contingency

### Pitfalls active for this plan

Cross-reference `.claude/skills/run-experiment/COMMON_PROBLEMS.md` and `ARCHITECTURE.md` §5. List only the ones that apply.

- `task.resample_variable` × `locate.mode` mismatch — *applies because we use `pairwise` on `day`*
- `task.balanced: true` overrides `resample_variable` — *not active in this plan*
- *…*

### Per-step contingency

For each node's pre-flight check, the next move if it fails.

| Node | If pre-flight fails, then |
|---|---|
| `baseline` | accuracy < 20% → reconsider task setup, single-token filter, model choice. Stop the chain. |
| `locate` | no cell with KL_drop ≥ 0.3 → re-examine token positions or expand layer range. Do not proceed to `subspace`. |
| (per node) | … |

---

## §F. Outputs of the plan itself

### Runner config(s)

Names only — YAML body is materialized by `/run-experiment` Step 3.

- `${SESSION_DIR}/code/configs/runners/{group}/{name}.yaml`
- *(if sweep)* `${SESSION_DIR}/code/configs/runners/{group}/{name}_k08.yaml`, `{name}_k16.yaml`, …

### Sweep & cache strategy

(Skip this subsection if the plan emits a single runner config.)

- **Sweep ID:** `{slug}` — appended to `experiment_root` so all sweep members share an artifact tree.
- **Resolved root:** `${SESSION_DIR}/artifacts/{task}/{model}/{sweep_id}/`
- **Sweep axis:** the parameter being varied (e.g. `subspace.k_features ∈ {8, 16, 32}`).
- **Cache reuse plan:**
  - `baseline`: runs **once** (shared across all members).
  - `locate`: runs **once** (shared, since `target_variables` is identical across members).
  - `subspace`: runs **N times**, each into a distinct `_subdir` (`pca_k8/`, `pca_k16/`, …).
  - `activation_manifold`: runs **N times**, each reading its own `subspace`.
- **Overwrite hazards verified absent:**
  - No two members share the same `_subdir` for any analysis. (Verified against `causalab/configs/analysis/{locate,subspace,activation_manifold}.yaml` `_subdir` patterns.)
  - No member changes `target_variables` (which would clobber `locate/{method}/results.json`). If you do need to vary `target_variables`, split into separate plans or separate `sweep_id`s.

See `instructions/plan_sweep.md` for the safety-rule details.

### Expected artifact tree

ASCII view rooted at `${SESSION_DIR}/artifacts/{task}/{model}/[{sweep_id}/]`. Pre-drawing it makes `/run-experiment` Step 6 (verify artifacts) trivial — just diff against this tree.

```
${SESSION_DIR}/artifacts/{task}/{model}/[{sweep_id}/]
├── baseline/
│   ├── accuracy.json
│   ├── per_class_output_dists.safetensors
│   └── counterfactual_sanity.json
├── locate/
│   └── interchange/
│       └── {variable}/
│           ├── results.json
│           └── heatmap.pdf
└── subspace/
    └── pca_k{N}/
        ├── rotation.pt
        ├── ref_dists.pt
        └── visualization/features_3d.html
```

### Hand-off

- Run `/run-experiment` to materialize runner config(s) and execute. It reads this plan and `RESEARCH_OBJECTIVE.md` from `${SESSION_DIR}/plan/`.
- Then `/interpret-experiment` reads the artifacts plus this plan as the interpretation lens.

---

## Review checkpoint (filled by `/plan-experiment`)

Before exiting `/plan-experiment`, the skill must surface to the user:

1. **Hypotheses + success criteria** (§A items 5–6 from `RESEARCH_OBJECTIVE.md`) — does the user agree these are the right tests?
2. **Sweep strategy** (§F) — does the user accept the cache-sharing plan, or do they want isolated `experiment_root`s?
3. **Compute estimate** (§C) — does the wall-time fit the user's window?

Record the user's response (or "approved silently in autonomous mode") in `${SESSION_DIR}/plan/approval.log`.
