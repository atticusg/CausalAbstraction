# Path patching

`causalab.methods.path_patching` computes **edge-level** patched logits in the
freeze-recipe semantics of Hanna et al. (2023, "How does GPT-2 compute
greater-than?"): replace a sender component's contribution along a named set
of paths, with every off-path component frozen at its clean value. Instead of
one intervened forward pass per patch, the engine reduces the recipe to
*delta arithmetic on cached activations* plus direct re-evaluation of the
receiver MLP branches and the model's own final norm + LM head — exact, and
hundreds of times cheaper for sweeps.

The reduction is exact only for **pre-LN additive-trunk decoders**, and
nothing is assumed: every engine verifies its own architecture facts
empirically at construction (see *Guards*).

## Quick start

```python
from causalab.methods.path_patching import (
    PatchEngine, PathSpec, build_patch_cache, resolve_descriptor
)
from causalab.neural.pipeline import LMPipeline

pipeline = LMPipeline("gpt2", max_new_tokens=1, dtype=torch.float32,
                      position_ids=True)
desc = resolve_descriptor(pipeline.model)

clean = build_patch_cache(pipeline, desc, clean_texts, {"end": -1})
cf = build_patch_cache(pipeline, desc, cf_texts, {"end": -1})
engine = PatchEngine(desc, clean, cf)   # runs the construction guards

# a9.h1's direct edge to the logits:
logits = engine.patched_logits(("head", 9, 1), PathSpec.cascade())

# a7.h10's paths through MLPs 8-11 (Fig 3C), direct edge off-path:
logits = engine.patched_logits(
    ("head", 7, 10), PathSpec.cascade([8, 9, 10, 11], direct_to_logits=False)
)
```

`positions` are given in unpadded per-example coordinates (`-1` = last real
token) and converted with the attention mask, so variable-length prompts
under left padding are handled correctly. Load pipelines with
`position_ids=True` for absolute-position models (GPT-2): HF does not adjust
position ids for left padding on its own, so without it a shorter example in
a padded batch runs at shifted positions.

## Senders

```
("head", layer, head)         attention head (z @ W_O slice)
("mlp", layer)                MLP block
("neuron", layer, index)      one MLP output-projection input channel
("neuron_group", layer, [i])  several channels of one MLP
("embed",)                    the embedding contribution
("group", [senders...])       union (e.g. a circuit complement)
```

On models with per-branch post-norms (Gemma-2), a branch's contribution to
the trunk passes through its post-norm, which is nonlinear; sender deltas are
computed as `post_norm(branch_clean + raw_delta) - post_norm(branch_clean)`
— exactly what a live patched forward produces — and raw deltas of senders
inside one branch are summed before the post-norm.

## Path specification

A patch is an explicit edge set over {sender, receiver MLPs, logits}
(`PathSpec`): `sender->receiver_k`, `sender->logits`,
`receiver_j->receiver_k`, `receiver_k->logits`. Each receiver's input delta
sums only its included incoming edges; its output delta propagates only along
its included outgoing edges. Receivers re-evaluate one MLP branch each —
still exact.

The common case is the closed cascade — **`PathSpec.cascade(receivers,
direct_to_logits=..., multipath="first"|"all")`** — the documented default
entry point. `"first"` sends the sender's delta only into the
furthest-upstream receiver (the iterative path patch of Hanna et al. Fig 3C);
`"all"` sends it into every receiver (the §3.3 indirect split). The explicit
edge set is the power-user form; `spec.without_edge(j, k)` excludes an
individual receiver-to-receiver edge, which the closed-cascade API cannot
express.

**Granularity ceiling** (deliberate): edges are atomic. One edge cannot carry
different values per downstream branch — the "treeified" patching expressible
in rust-circuit is out of scope for this cache-arithmetic engine.

`engine.circuit_eval(heads, mlps, direction, head_receiver_mlps)` evaluates a
whole circuit (sufficiency: run corrupted, restore circuit paths to clean;
necessity: run clean, corrupt circuit paths), with per-head edge routing into
the circuit MLPs.

## Guards

Construction runs four empirical checks and refuses to patch on failure
(`GuardError` names the likely cause; measured errors are kept in
`engine.guard_report`):

| guard | catches |
|---|---|
| G1 additivity: final residual = embedding + Σ branch trunk contributions | post-LN trunks, wrong capture points |
| G2 branch wiring: re-evaluating each MLP branch on its derived input reproduces the cached branch output | mis-declared block order (sequential vs parallel), wrong pre-norm module |
| G3 patch-nothing closure: reassembled tail reproduces the model's logits | wrong final norm / LM head / missing softcapping |
| G4 patch-everything closure: patching every direct edge reconstructs the counterfactual logits | wrong capture points, inconsistent caches |

Note that G4 is a *direct-edge sum* and is numerically order-invariant — the
block-order teeth are G2's per-layer receiver re-evaluation.

Block order is resolved from the HF config (`use_parallel_residual`) and can
be overridden (`resolve_descriptor(model, block_order=...)`) — the override
exists so tests can prove the guards catch a lie. On parallel-residual models
(Pythia) the MLP reads the block *input*, so a same-layer head never feeds
the same-layer MLP; the engine's edge routing respects this.

## Capability contract

causalab's contract is "a model works iff its family is in pyvene's mapping".
This method keeps it exactly: `check_capability` verifies at construction
that every pyvene unit the requested operation needs exists in the family's
mapping, else raises `UnsupportedArchitectureError` naming the missing units
and the operation. There are **no raw torch hooks anywhere** (test-enforced).
Capture points beyond pyvene's named vocabulary use pyvene's dotted
module-path resolution — still `IntervenableModel` interventions.

- `engine.provenance` documents which pyvene units the engine uses
  (named / dotted-path / direct module call); validation runs write it into
  their results JSONs.
- `coverage_table()` reports supported/unsupported per family × component
  from the *installed* pyvene; it is committed as a test artifact
  (`tests/test_methods/artifacts/path_patching_coverage.json`) and
  regenerated in CI, so a pyvene pin bump surfaces as a table diff rather
  than a silent behavior change.

## The reference twin

`reference_patched_logits` implements the same freeze recipe as *live
intervened forwards* through pyvene (sender substitution, downstream freezes,
excluded-edge cancellation), mechanistically independent of the engine's
cache arithmetic, so engine-vs-twin agreement is an informative check rather
than a tautology. It is slow and meant for validation.

One pyvene mechanic to know about when writing norm-input interventions
anywhere in causalab: **setter interventions scatter in place**, and a
pre-norm's input tensor *is* the residual-stream tensor, so modifying it
changes the trunk downstream, not just what the norm sees. The twin pairs
every pre-norm cancellation with a trunk-output restoration on the same
receiver (`_TrunkOutputAdjust`) to keep the recipe exact.

## K/V-side attention detail (gated)

`build_attn_detail_cache` captures per-head q/k/v (pyvene
`query/key/value_output`; scores are derived arithmetic on cached q/k — the
score tensor is not a module boundary and never needs intervening on). It is
**gated to GPT-2-style attention** (`fused-qkv-absolute`) and raises
`NotImplementedError` on rotary/GQA models, with the agreed generalization
direction documented in `kv.py`: rotate key deltas by the patched position's
angle before re-scoring (pre-rotation keys via pyvene + rotation in engine
arithmetic is pyvene-native); use KV heads, not query heads, as the edge unit
under GQA; keep the K/V twin as pyvene replace interventions on
`head_key_output`/`head_value_output`. K/V capture requires
`attn_implementation="eager"` (the `LMPipeline` default) — fused kernels do
not materialize what pyvene's attention units need.

## Worked example: an iterative path-patching sweep

With any set of clean/counterfactual prompt pairs (templated prompts that
differ in one token, and a task metric over the logits), the full iterative
path patching of Hanna et al. (2023) §3.1 is a loop over senders × path
modes:

```python
modes = {
    "logits":    PathSpec.cascade(),
    "via_mlp10": PathSpec.cascade([10, 11], direct_to_logits=False),
    # ...
}
cell = prob_diff(engine.patched_logits(("head", l, h), spec)) - clean_prob_diff
```

During development the engine was validated against a hook-based
reimplementation of Hanna et al.'s greater-than analysis on GPT-2, matching
it cell-for-cell at float precision — at a fraction of the per-patch cost of
hook-based patching.
