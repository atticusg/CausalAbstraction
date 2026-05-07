# Sweep & Cache Strategy → `PLAN.md` §F

Goal of this step: when the plan emits multiple runner configs (a sweep), make the cache-sharing strategy explicit and verify no two members will silently overwrite each other.

## When a sweep is needed

A sweep is justified when the research question itself is parameter-comparison:

- "Which `k_features` (8, 16, 32) best captures `day`?"
- "Which subspace method (PCA, DAS, DBM) preserves causal structure best?"
- "Which `smoothness` (0.0, 1.0, 10.0) gives the cleanest manifold?"

If the question is single-axis ("does `day` localize?"), skip the sweep section and emit one runner config.

## Sweep ID

If sweeping, declare a `sweep_id` slug (kebab-case): `pca_k_scan`, `das_vs_dbm`, `smoothness_scan`. All sweep members share:

```
experiment_root = ${SESSION_DIR}/artifacts/{task}/{model}/{sweep_id}
```

This is a single-segment append, implemented today by `apply_experiment_root_variant` in `causalab/io/configs.py:40-55` (which appends `task.variant` if set). The skill piggybacks on this mechanism by either (a) setting `task.variant: {sweep_id}` in each member's runner config, or (b) requesting the user to add a `cfg.sweep_id` field — a one-liner in `apply_experiment_root_variant` deferred to implementation. The plan should declare which approach it uses, default to (a) for now.

## Cache reuse plan

For each analysis node in the DAG, document how many times it runs across the sweep, and what is cached vs recomputed:

| Node | Runs | Cached? | Notes |
|---|---|---|---|
| `baseline` | 1× | yes | shared across all members |
| `locate` | 1× | yes | shared (since `target_variables` is identical) |
| `subspace` | N× | distinct `_subdir` per member | one dir per member; no overlap |
| `activation_manifold` | N× | distinct `_subdir` | reads its corresponding subspace |

A member's first run computes the shared upstream stages; subsequent members hit those on disk via auto-discovery. **Note:** every analysis except `path_steering` overwrites unconditionally on re-run (verified in `causalab/analyses/{baseline,locate,subspace,activation_manifold}/main.py`). The cache-hit pattern works because the second member's *upstream* nodes have nothing to write — they read what the first member already produced.

## Overwrite hazard checklist

Before approving the sweep, verify each item:

### Hazard 1: two members map to the same `_subdir`

Each analysis's `_subdir` pattern is in `causalab/configs/analysis/<name>.yaml`. Two members must not collide:

| Analysis | `_subdir` pattern | Safe sweep axes | Unsafe sweep axes |
|---|---|---|---|
| `locate` | `${.method}` | `method` (interchange/dbm_binary) | `target_variables`, `mode`, `layers` |
| `subspace` | `${.method}_k${.k_features}` | `method`, `k_features` | other knobs |
| `activation_manifold` | `${.method}_s${.smoothness}` | `method`, `smoothness` | other knobs |

Sweeping along an axis encoded in `_subdir` is safe. Sweeping along any other axis collides — split into separate plans (or separate `sweep_id`s) instead.

### Hazard 2: `target_variables` varying across `locate` members

`locate` writes one `results.json` per `target_variable`, but they all live under one `locate/{method}/` subtree. Two `locate` runs with different `target_variables` lists each write *all* their variables under that subtree — not destructive on a per-variable basis (because the path is `locate/{method}/{variable}/results.json`), but **the auto-discovery rule** in `causalab/io/pipelines.py:140` returns the *alphabetically-first* `locate/` subdir's `best_cell`. If two members differ in which variable downstream `subspace` should target, auto-discovery will silently pick the wrong one.

**Rule:** if the sweep changes which variable downstream nodes target, do **not** share `experiment_root`. Use separate plans or separate `sweep_id`s.

### Hazard 3: alphabetical tie-breaking

Auto-discovery uses `sorted()` on directory names, not mtime (verified in `causalab/io/pipelines.py:88-159`). Naming matters:

- `subspace/pca_k08/` sorts before `subspace/pca_k16/`. With zero-padded names, the order is predictable.
- `subspace/das/` sorts before `subspace/pca_k16/`. Mixing methods + parameter scans makes auto-discovery fragile.

If the sweep mixes methods, force downstream nodes to declare an explicit `subspace: "pca_k08"` (or whichever) rather than relying on `null`. Document the explicit selection in the per-node card.

## Required §F outputs

After verifying the hazards, the sweep section of `PLAN.md` §F must contain:

1. **Sweep ID** declaration.
2. **Resolved root** path: `${SESSION_DIR}/artifacts/{task}/{model}/{sweep_id}/`.
3. **Sweep axis** — the parameter being varied.
4. **Cache-reuse table** (one row per analysis node, as above).
5. **Hazard checklist** with each item marked verified or, if unavoidable, a note on how it is mitigated (e.g. "`subspace` member directories are zero-padded so alphabetical order matches numeric order").
6. **Member runner-config names** — one per member, listed under `${SESSION_DIR}/code/configs/runners/{group}/{name}_<axis>.yaml`.

## Single-config plans

Skip the sweep section entirely. The plan emits one runner config; `experiment_root` is `${SESSION_DIR}/artifacts/{task}/{model}` (with optional `task.variant` appended by the existing mechanism).
