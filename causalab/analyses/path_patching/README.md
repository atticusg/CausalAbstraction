# Path Patching

Path patching answers: *which attention head has a direct causal effect on the
model's answer — the logit difference between the correct and distractor tokens?*
For each `(layer, head)` sender it patches the head's output from a counterfactual
source while freezing every other path (the component families named by `restore`)
to the clean base — this is path patching, not a plain activation patch — and
measures the edge's **direct effect** on the logit difference (`base − patched`).
The result is a (layer × head) direct-effect heatmap: this is the IOI Fig. 3
head sweep, and name-mover heads score large-positive.

By default the receiver is the output logits (one intervened forward per cell).
An internal receiver — a head's query/value input, an MLP input, or a residual
position (or a *set* of head receivers patched simultaneously) — can be targeted
instead, reproducing the IOI Fig. 4/5 composition edges via a two-pass
collect/inject runner. This is a terminal circuit-discovery diagnostic: it reads
only the task and the model (plus the task's `generate_abc_dataset` for the `abc`
corruption), and no downstream analysis consumes its outputs. Run `baseline` first
to confirm the model solves the task.

---

## Configuration

**Root config** (`causalab/configs/config.yaml`) — `experiment_root`, `seed`.
Dataset size is read from the task config (`cfg.task.n_test`, `cfg.task.n_train`,
`cfg.task.enumerate_all`).

**Module config** (`causalab/configs/analysis/path_patching.yaml`):

```yaml
# @package path_patching
_name_: path_patching
_subdir: ${.corruption}_${.relative_to_base}
_output_dir: ${experiment_root}/path_patching/${._subdir}

# Sender grid: attention heads (the (layer x head) heatmap axes). null = all.
layers: null                     # list[int] or null = all layers
heads: null                      # list[int] or null = all heads

# Sender + internal-receiver read position. null/last_token = the final token
# (IOI Fig. 3/4 readout). A task position name (e.g. name_C / "S2") moves BOTH
# the sender and the receiver there (Fig. 5 duplicate -> S-Inhibition edge).
token_position: null             # null | last_token | <task position name>

# Receiver of the patched edge. Default = the output logits (single forward).
# Internal receivers use the two-pass runner and land in a receiver-tagged subdir.
receiver:
  kind: output                   # output | head_value_input | head_query_input | mlp_input | residual
  layer: null                    # required for internal receivers
  head: null                     # required for head_value_input / head_query_input
  residual_point: block_output   # used only when kind == residual (block_output | block_input)
  heads: null                    # null = single receiver; list[[layer, head]] = a receiver SET (patched together)

# Restorer composition = the estimand (component families frozen above the sender):
#   [attention, mlp] -> strict residual-stream-to-output direct effect
#   [attention]      -> MLPs/LayerNorm recompute (Wang et al. 2022 §3.1 direct effect)
restore: [attention, mlp]

# Counterfactual source ("corruption"):
#   abc         -> all three names replaced by distinct randoms (paper §3.1)
#   answer_flip -> resample `resample_variable` only (flips the answer)
corruption: abc
resample_variable: name_C        # used only when corruption == answer_flip

# Logit-difference metric: direct effect on logit[correct] - logit[distractor].
correct_variable: IO             # input variable holding the correct answer
distractor_variable: name_C      # input variable holding the distractor answer
relative_to_base: true           # report base - patched (the direct effect)

batch_size: 8
top_k: 20

visualization:
  figure_format: ${figure_format}   # png or pdf — invariant 6
```

**`restore`** is the key modeling choice — it selects the estimand. `[attention,
mlp]` freezes both families above the sender for a strict residual-to-output
direct effect; `[attention]` lets MLPs/LayerNorm recompute, matching Wang et al.
(2022) §3.1. **`corruption`** selects the counterfactual source; `abc` requires the
task to define `generate_abc_dataset` in its counterfactuals module.
**`receiver`** picks the estimand's downstream endpoint; internal and set
receivers each get their own output subdir so distinct receivers never clobber
each other's `results.json`. The variable names (`correct_variable`,
`distractor_variable`) are read directly from this config — the sweep manages its
own variable selection and runs once, not once per task target variable.

---

## Outputs

Files are written under
`{experiment_root}/path_patching/{corruption}_{relative_to_base}/` (e.g.
`abc_True/`). Internal and set receivers add a receiver-tagged subdirectory (e.g.
`head_query_input_L9H6/` or `head_query_input_set_L9H6_L9H9_L10H0/`).

### Interpretation

- **`results.json`** — The direct answer: the per-cell `effect_grid` (direct effect
  on the logit difference, keyed `"L|H"`), the `top_k_cells` ranked by absolute
  effect, plus the `receiver` and `restore` used. A large **positive** effect marks
  a name-mover head — patching it from the counterfactual moves the logit
  difference toward the base answer along the direct path; a large **negative**
  effect marks a negative name-mover (a head that suppresses the correct token);
  ~0 means the head has no direct effect on this edge. When the receiver is
  internal, senders at or downstream of its read point are structurally zero and
  dropped from the grid.

- **`heatmap.{png,pdf}`** — The `effect_grid` as a (layer × head) heatmap, colorbar
  labelled **"Direct effect on logit diff"**. It uses a symmetric diverging scale
  (`RdBu_r`) centered at 0 so the sign is legible — negative name-movers are not
  clamped to the floor color. Look for the small set of hot cells that localize the
  circuit's heads. Extension is set by `analysis.visualization.figure_format`.

### Saved artifacts

| File | Shape / Format | Used by |
|---|---|---|
| `results.json` | `{receiver, restore, effect_grid {"L|H": float}, top_k_cells}` | human reference |
| `heatmap.pdf` / `.png` | (layer × head) direct-effect heatmap | human reference |
| `metadata.json` | run config snapshot (sender/receiver/restore, corruption, correct/distractor variables, layers/heads, task/model, seed) | provenance |

The sender is always an attention head. When an internal receiver is used the
heatmap's layer axis is recomputed from the surviving (reachable) senders, so the
grid stays rectangular even after downstream layers are dropped.
