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

## K/V-side attention detail

Edges that end at an attention head's **keys or values at one patched token
position** (the App. 8 analyses of Hanna et al.: what feeds a head's k/v at
an earlier prompt position). General across the same families as the engine, under the same
support policy: available iff pyvene's mapping has `query/key/value_output`
(collection) and `head_key_output`/`head_value_output` (twin) for the family.
gpt_neox has none of them (its fused QKV projection interleaves q/k/v per
head, and pyvene's mapping comments the units out), so Pythia-style models
raise `UnsupportedArchitectureError` at construction — no hook fallback.

```python
from causalab.methods.path_patching import (
    KVEdge, KVHead, KVPatchEngine, build_attn_detail_cache,
)

# "src": the patched upstream position (say, the prompt's subject token)
positions = {"end": -1, "src": -5}
clean = build_patch_cache(pipeline, desc, clean_texts, positions)
cf = build_patch_cache(pipeline, desc, cf_texts, positions)
engine_end = PatchEngine(desc, clean, cf, position="end")
engine_src = PatchEngine(desc, clean, cf, position="src", run_guards=False)
det_clean = build_attn_detail_cache(pipeline, desc, clean_texts, positions)
det_cf = build_attn_detail_cache(pipeline, desc, cf_texts, positions)
kv = KVPatchEngine(engine_end, engine_src, det_clean, det_cf,
                   position_patch="src")   # runs guard K1

# sender m0's delta at src entering a7.h10's values, then on to the logits:
delta_src = engine_src.trunk_delta(("mlp", 0))
edges = [KVEdge(KVHead.for_query_head(desc, 7, 10), patch_v=True)]
logits = engine_end.patched_logits(kv.kv_trunk_delta(edges, delta_src),
                                   PathSpec.cascade())
```

Design points, each enforced or verified at construction:

* **Pre-rotation capture, model-applied rotation.** pyvene's
  `key_output`/`value_output` hook the projection outputs; on rotary
  families that is before RoPE, so cached keys are position-disentangled.
  Score reconstruction applies the model's own rotary module at the actual
  position ids; rotation is linear, so rotating a patched key equals
  rotating the key delta. `rotate_key_delta=False` exists purely as a
  negative control (it must disagree with the twin on rotary families).
* **KV heads are the edge unit.** `KVHead(layer, kv_index)` is what is
  causally separable in the weights; z-deltas fan out across the query-head
  group (`desc.query_heads_of_kv`). Standard MHA is group size 1.
  Per-query-head value paths do not exist in the weights; a
  TransformerLens-style expansion would be pure cache arithmetic on top and
  is deliberately not a default.
* **Eager contract.** Construction refuses non-eager loads by name
  (`LMPipeline(..., attn_implementation="eager")` /
  `attn_implementation: eager` in the model YAML). No silent re-loading.
* **Sliding windows** (Gemma-2): an edge whose patched position the end
  position cannot attend to within the layer's window raises
  `SlidingWindowError` naming the layer and window.
* **Guard K1**: reconstructed per-head z at the end position must match the
  cached `attention_value_output` on both caches — rotation, GQA fan-in,
  scaling, softcapping, masks and softmax verified in one check (tight in
  float32; bf16 floor recorded, as with G4).
* **Twin.** `reference_patched_logits` accepts a
  `("kv", layer, kv_index, payload)` sender: pyvene replaces the KV head's
  pre-rotation k/v at the patched position and the model's own attention
  recomputes — scoring, softmax, softcapping, windows, GQA fan-out all come
  from the model. The twin validates atomic K/V edges; multi-edge
  composition is anchored against the GPT-2 numbers of a hook-based
  replication of Hanna et al. (2023).

Granularity ceiling unchanged: edges are atomic (no treeified patching), and
the score tensor is never a module boundary — analytic K/V patching
recomputes scores from cached q/k.

**Anchor.** During development the K/V machinery was anchored against a
hook-based replication of Hanna et al. (2023): it reproduces the paper's
appendix analyses (Figs 11–13 and the full-circuit evaluation) at float
precision (all 243 Fig 12–13 sweep cells ≤4.2e-7 and the 97 Fig 11 cells
≤1.6e-6; full-circuit probability difference to 1.2e-7) when composed with
that analysis's conventions — head
z-deltas routed into MLPs 8–9 only while the MLP cascade runs over 8–11, the
full-circuit K/V restoration built from summed cached component deltas, and
base-side statistics with unrestored queries. Those are analysis-level
composition choices, not engine semantics: they live in the anchoring
analysis's driver, and every one of them is expressible through the public
primitives above (`kv_trunk_delta` with `base_cache`, `trunk_delta` over a
component group, receiver routing in the caller).

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
