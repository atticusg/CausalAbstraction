---
name: <analysis_name_in_snake_case>
---

# Analysis spec: `<analysis_name>`

5-section spec consumed by `/setup-analyses`. Lands at `${SESSION_DIR}/code/analyses/<name>/set_up_analysis.md` after the skill writes it. The skill scaffolds `main.py`, `analysis.yaml`, and `README.md` from this spec; downstream `/run-experiment` references the analysis from a runner config.

---

## §1. Identity

**Research question** (one italicized sentence — drives the README opening):

> *Does intervening on attention head (L, H) change the model's output by more than baseline noise?*

**One paragraph — what the analysis does mechanically.** Forward passes only? Trained subspace? Iterative optimizer? Where does it sit in the pipeline (what does it read, what depends on it)?

> Example: *Reads `baseline/per_class_output_dists.safetensors` and `locate/interchange/<variable>/results.json`, then runs an interchange ablation per attention head on the train dataset and records the mean logit-difference shift per (layer, head) cell. Writes one `head_logit_diff/results.json` plus a `(layers × heads)` heatmap PDF.*

---

## §2. Position in the DAG

### Upstream artifacts read

List every file path under `${experiment_root}/...` this analysis reads, with the upstream analysis that produces it. Use `null` for auto-discovered inputs and cite the auto-discovery rule (see `ANALYSIS_GUIDE.md` "Auto-Discovery").

| Artifact | Produced by | Path |
|---|---|---|
| reference distributions | `baseline` | `${experiment_root}/baseline/per_class_output_dists.safetensors` |
| best-cell metadata | `locate` | auto-discovered (alphabetically-first `locate/<method>/<variable>/results.json`) |

### Downstream consumers

Which other analyses (shipped or session-local) can read this analysis's outputs? `null` if terminal.

> Example: *None — this is a terminal analysis whose outputs are interpreted in `/interpret-experiment`.*

---

## §3. Methods used

Every callable invoked in `main.py`. Each comes from one of:

- `causalab/methods/<…>` — shipped primitive.
- `${SESSION_DIR}/code/methods/<…>` — session-local primitive (must already be scaffolded by `/setup-methods`).

| Symbol | Source | Notes |
|---|---|---|
| `compute_reference_distributions` | `causalab.methods.metric` | reused as in baseline |
| `head_logit_diff` | `methods.head_logit_diff` (session) | scaffolded via `/setup-methods` |

If a session-local method listed here does not yet exist, the skill instructs the user to run `/setup-methods` for it first (passing the missing method's spec path).

---

## §4. Config schema

Every knob that needs a default in `analysis.yaml`. `_name_`, `_subdir`, `_output_dir` are auto-emitted. Dataset-construction knobs (`n_train`, `n_test`, `enumerate_all`, `balanced`) and `seed` are NOT included — those live in `cfg.task.*` and `cfg.seed` respectively (invariant 12).

| Knob | Type | Default | Description |
|---|---|---|---|
| `method` | `str` | `"interchange"` | underlying method ("interchange" / "ablation") |
| `layers` | `list[int] \| null` | `null` | layers to scan; `null` → all |
| `heads` | `list[int] \| null` | `null` | heads to scan; `null` → all |
| `batch_size` | `int` | `32` | inference batch size |

Choose a `_subdir` pattern that encodes the parameter axes most likely to be swept:

```yaml
_subdir: ${.method}                    # if only method varies
_subdir: ${.method}_l${.layers}        # if layers varies (zero-pad if numeric)
_subdir: default                       # if no axis varies in expected sweeps
```

`_subdir` decisions matter because two runs that share the same `_subdir` overwrite each other (see `plan_sweep.md` Hazard 1 in `plan-experiment`).

If the analysis emits any matplotlib figure, include:

```yaml
visualization:
  figure_format: pdf      # png or pdf — invariant 6
```

---

## §5. Outputs

Every file the analysis writes. Drives both the README "Saved artifacts" table and `/run-experiment`'s expected-artifact tree.

| File | Format | What it shows | Used by |
|---|---|---|---|
| `results.json` | `{"per_head": {"<L>_<H>": float}, "summary": {...}}` | mean logit-difference shift per cell | `/interpret-experiment`; downstream session-local analyses |
| `heatmap.pdf` | matplotlib heatmap | (layers × heads) cell scores | human reference |
| `metadata.json` | run config snapshot | provenance | `/run-experiment` Step 6 verification |

For each `.json` output, document the schema concretely so downstream consumers can write against it without re-reading the analysis source.

---

## Notes (optional)

Free-form section for the implementation agent — references to similar shipped analyses to model from (e.g. *"shape closely matches `causalab/analyses/locate/run_interchange.py`"*), known compute footprint, edge cases.
