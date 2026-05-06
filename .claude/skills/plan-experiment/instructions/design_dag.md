# Design the Analysis DAG → `PLAN.md` §D

Goal of this step: walk the dependency DAG from `ANALYSIS_GUIDE.md`, choose the nodes that answer the research questions implied by the objective, and fill one per-node card per chosen node.

## Source of truth

`ANALYSIS_GUIDE.md` is canonical for:

- the dependency DAG
- the "Research Questions → Analyses" table (which analysis answers which question)
- per-analysis "Key Parameter Decisions" (which knobs require a decision; everything else has a default)
- auto-discovery rules (what `null` resolves to per analysis)

Read it before drafting §D. Do not duplicate its content into the plan — reference it.

## Node selection

Walk from the objective's hypotheses to the analyses that test them:

| Hypothesis pattern | Analysis to chain |
|---|---|
| "Does the model solve the task?" | `baseline` |
| "Where does variable X live?" | `baseline` → `locate` |
| "What subspace captures X?" | `… → subspace` |
| "What's the geometry of X's representation?" | `… → activation_manifold` |
| "Is the subspace causally faithful?" | `… → path_steering` |
| "What output-distribution geometry corresponds?" | `baseline → output_manifold` |
| "What activation trajectories realize a belief path?" | `… → activation_manifold + output_manifold → pullback` |
| "Which heads attend where?" | `attention_pattern` (independent) |

**Always include `baseline`.** It is the gate for every downstream node and reuses output distributions for `locate`.

## Per-node card

Each node in §D has a card with these fields. Adapt depth — if method choice is the research question, expand the method line; otherwise leave it terse.

### Required fields

- **Research question (scoped):** *one sentence, narrower than the overall objective*. What does this node tell us, in isolation?
- **Method:** name from `ANALYSIS_GUIDE.md` "Key Parameter Decisions". Document method-level choice only when method choice **is** the research question (e.g. comparing DAS vs DBM).
- **Upstream artifacts consumed:** explicit file paths under `${experiment_root}/...`. Use `null` for auto-discovered inputs and cite the auto-discovery rule from `ANALYSIS_GUIDE.md` "Auto-Discovery". Example:
  - `subspace.layers: null` → resolves to `best_cell` from `${experiment_root}/locate/{method}/{variable}/results.json` (alphabetically-first match per `causalab/io/pipelines.py:140`).
- **Downstream artifacts produced:** explicit file paths the node writes. Pull from `ANALYSIS_GUIDE.md` "Research Questions → Analyses" Key outputs column.
- **Non-default knobs:** only the deltas vs `causalab/configs/analysis/{name}.yaml`. Each delta gets a one-line rationale.
- **Pre-flight check:** **the minimal finding required in this node's outputs before any dependent node runs.** This is a gate, not a prediction. Phrase it as a measurable condition on a specific output file.
  - Good: `locate` — `results.json` must contain at least one cell with `KL_drop ≥ 0.3`. If not, stop the chain.
  - Bad: `locate` — heatmap should show a hot region around layer 14. (This is a prediction; do not write predictions in the plan.)
- **Estimated runtime + GPU footprint:** `~X min on Y GPU(s)` or `~X min CPU` — derived from past runs of similar size.

### Pre-flight checks: how to phrase them

Pre-flight checks act as analysis-sequence gates. They must be:

- **Measurable** — a number, a presence check, a file existence, never a vibe.
- **Output-grounded** — references a file or field that the node writes.
- **Restrictive but not predictive** — sets the floor for "the upstream signal is real enough to run downstream", not "this is what we expect to find".

Examples by analysis (edit per-plan as appropriate):

| Node | Pre-flight check |
|---|---|
| `baseline` | `accuracy.json` reports `accuracy ≥ 0.20`. |
| `locate` | `{variable}/results.json` lists ≥ 1 cell with `KL_drop ≥ 0.3` (centroid mode) **or** `string_match_rate ≥ 0.3` (pairwise mode). |
| `subspace` | `metadata.json` reports `reconstruction_kl ≤ 1.0` (PCA) **or** `causal_alignment ≥ 0.5` (DAS). |
| `activation_manifold` | `manifold_spline/metrics.jsonl` final `reconstruction_kl ≤ 0.5`. |
| `output_manifold` | `manifold.pt` exists and `metadata.json` reports `reconstruction_hellinger ≤ 0.2`. |
| `path_steering` | per-metric `results.json` exists and the relevant score (isometry / coherence / conformal) is finite. |
| `pullback` | `belief_paths/` is non-empty and `results.json` reports finite path losses. |
| `attention_pattern` | `results.json` exists. |

These thresholds are starting points — adjust per task and discuss with the user during the review checkpoint.

## DAG diagram

Draw an ASCII diagram showing only the nodes in this plan, with the dependency edges from `ANALYSIS_GUIDE.md`. Trim unused nodes. Example:

```
baseline ──► locate ──► subspace ──► activation_manifold ──► path_steering
```

## Cross-analysis post-steps

Anything in the runner config's `post:` block (per `ARCHITECTURE.md` §2 "Execution model"):

- `variable_localization_heatmap` — aggregates `locate` outputs across multiple `target_variables`. Only useful when ≥ 2 variables are localized.

## Granularity rule

If the entire research question is method-comparison (e.g. "which subspace method best captures `day` — PCA, DAS, or DBM?"), expand the relevant node into one card per method *and* convert it into a sweep (see `plan_sweep.md`). Otherwise, one card per analysis.

For new analyses or methods that need to be implemented (no `analyses/<name>/` package yet, or a missing primitive in `methods/`), mark the node `custom` and record the missing pieces in the per-node card under a **Needed implementations:** line — list each missing analysis or method symbol with a one-line purpose. After plan approval and before `/run-experiment`, run `/setup-methods` and/or `/setup-analyses` (each accepts a batch of specs in a single invocation); those skills scaffold the missing pieces under `${SESSION_DIR}/code/` and the runner picks them up via session-code path injection. Implementation happens between plan approval and the first run, not inside this skill.
