# Running experiments

Causalab runs neural-network interventions from a **document**: a JSON file that
names the model, the data, the activations to read, the edits to make, and the
numbers to save. The document is the experiment — there is no Python config
layer, and nothing about a run is decided by code you write.

This page is the path from "I have a hypothesis" to "I have a saved,
digest-stamped result", plus the table of every hookpoint the
[Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) architecture
exposes.

Reference material this page points at rather than repeats:
[`intervention_protocol.md`](intervention_protocol.md) (the normative spec),
[`workflow_protocol.md`](workflow_protocol.md) (chaining documents),
[`CODEBASE.md`](CODEBASE.md) (module map), [`TESTS.md`](TESTS.md) (test tiers).

## Setup

```bash
uv sync                       # the nnsight engine ships in the dev group
uv run causalab --help
```

## 1. Serialize a dataset

A document names a dataset **by ref**; a ref resolves by reading bytes under
`--data-root`. Nothing is generated during a load, so `validate` needs no task
code, no tokenizer and no network — and a document's digest is a function of
committed bytes.

```bash
uv run python scripts/build_task_dataset.py \
    --task MCQA --n 32 --seed 0 --target-variable answer \
    --out data/mcqa.json
# wrote data/mcqa.json (32 rows, digest 688a10e4cb7f…)
```

The sidecar `data/mcqa.manifest.json` records the parameters the bytes came
from: the table is a build product, the manifest is the recipe.

## 2. Write the document

An interchange intervention — read the answer-slot residual stream from the
counterfactual prompt, patch it into the base prompt, score what changed. Save
as `patch.json`:

```json
{
  "version": "1",
  "description": "Interchange the answer-slot residual stream at one layer.",
  "model": {"key": "Qwen/Qwen3.6-35B-A3B", "revision": "main"},
  "data": {
    "base":           {"dataset": "mcqa", "field": "input"},
    "counterfactual": {"dataset": "mcqa", "field": "counterfactual_inputs[0]"}
  },
  "sites": {
    "target":  {"component": "block_output", "layer": 20},
    "lm_head": {"component": "lm_head"}
  },
  "reads": {
    "v_cf":   {"site": "target",  "pos": -1, "model": "original", "input": "counterfactual"},
    "logits": {"site": "lm_head", "pos": -1, "model": "patched",  "input": "base"}
  },
  "writes": {
    "patch": {"site": "target", "pos": -1, "do": {"swap": "v_cf"}}
  },
  "intervened_models": {
    "patched": {"input": "base", "writes": ["patch"]}
  },
  "metrics": {
    "iia":        {"kind": "match",      "of": "logits", "expected": "label"},
    "logit_diff": {"kind": "logit_diff", "of": "logits", "a": "cf_answer", "b": "base_answer"}
  },
  "save": [
    {"value": "iia",        "model": "patched", "input": "base", "file_path": "iia.json"},
    {"value": "logit_diff", "model": "patched", "input": "base", "file_path": "logit_diff.json"}
  ]
}
```

Reading it in section order (the order is enforced; `save` is always last):

- **`sites`** is the complete tap inventory. Every address a read or write
  names, `lm_head` included. There are no implicit site names.
- **`reads`** produce values, each bound to one (site, position, model, input).
- **`writes`** are inert definitions — an address and a mechanism, no model.
  They do nothing until an intervened model lists them, which is what makes one
  write reusable across several.
- **`intervened_models`** is where a write comes into force. `original` is
  reserved for the un-intervened model and is never declared.
- **`metrics`** reduce a read against dataset columns.
- **`save`** is the complete manifest of what leaves the run.

`v_cf` is read in one model and consumed by a write in force in another: that is
the single channel for cross-model data flow, and the graph it induces is the
execution schedule.

### Document sections

| section | required | declares |
|---|---|---|
| `version` | ✓ | `"1"` |
| `description` | – | intent, free text |
| `model` | ✓ | the network as a name: `key`, `revision`, `dtype`, `quantization` |
| `data` | ✓ | input rows: `base`, optional `counterfactual` — dataset ref + field |
| `positions` | – | named token-position specs |
| `sites` | ✓ | named activation addresses — the complete tap inventory |
| `featurizers` | – | named feature-space maps |
| `params` | – | free/constant tensors owned by no featurizer |
| `reads` | ✓ | value producers: (site, pos, model, input) [+ featurizer, dims] |
| `writes` | – | inert effect definitions: (site, pos, `do`) |
| `intervened_models` | –* | which writes are in force on which input (*required with `writes`) |
| `metrics` | – | closed reductions over read values |
| `train` | – | the fit: objective, params, optimizer, steps, batch, seed |
| `save` | ✓ | the output manifest — non-empty, last |

### Closed vocabularies

Anything outside them is a load error, not a fallback.

| vocabulary | values |
|---|---|
| `sites.component` | the 62 hookpoints in [§5](#5-hookpoints-on-qwen36-35b-a3b) |
| `sites.stream` | `full_attention` · `linear_attention` — a per-layer fact on a hybrid tower, refused at load if the layer carries the other one |
| `sites.head` / `sites.expert` | sub-axis selectors, legal only where the component has that axis |
| `writes.do` | `swap` · `add_scaled` · `lerp` · `affine` · `gaussian` · `renormalize` · `clamp` · `pytorch_fn` (local-only) |
| `metrics.kind` | `logit_diff` · `token_logit` · `cross_entropy` · `kl` · `class_probs` · `top_k` · `match` · `decode` |
| `featurizers.kind` | `identity` · `subspace` · `pca` · `sae` · `standardize` · `gate` |
| `pos` forms | `-1` (sugar for `{"index": n}`) · `"all"` · `{"variable": v}` · `{"column": c}` · `{"span": [a, b]}` (half-open), modified by `scope` / `relative_to` / `generated` |
| save formats | `.json` (per-example tables) · `.safetensors` (dense numerics) |

Three rules that catch most authoring mistakes:

| rule | consequence |
|---|---|
| one global namespace over the named sections | every name unique; `base`, `counterfactual`, `counterfactual[j]`, `original` are reserved |
| at most one **absolute** write per (site, overlapping pos, model) | any number of additive writes; absolute applies first, then the summed deltas — so write sets are order-free |
| `{"sweep": [v, …]}` / `{"sweep": {"range": [a, b]}}` is the only axis | bare arrays are never axes; axis identity is name identity, so sweeping `sites.target.layer` moves the read, the write and the metric together |

Derived, never authored: feature widths, `num_forwards`, the point count, the
`requires` capability set, digests. If it can be computed from the document, the
document must not say it.

## 3. Check it before you spend a GPU

Both verbs are pure — no weights, no network, no accelerator — so they cost a
second and catch every load error.

⚠️ They are also **registry-only**: they derive featurizer widths from the
static metadata in `causalab/protocol/registry.py` rather than fetching a
config, so a digest never depends on connectivity. The A3B is not one of the
six built-in entries, so a document naming it refuses:

```bash
uv run causalab validate patch.json --data-root data --data
# refused: [V4] at model.key model 'Qwen/Qwen3.6-35B-A3B' is not in the protocol
#          model registry — register its static config
#          (causalab.protocol.registry.register_model, or model_info_from_hf_config
#          on a loaded HF config)
```

Two ways forward, and which you want depends on what you are checking:

- **checking the document** — point the pure verbs at a registered key. The
  structure, the reference graph and the save manifest are model-independent;
  only widths and layer bounds are not:

  ```bash
  uv run causalab validate patch.json --data-root data --data \
      --set model.key=Qwen/Qwen3-4B-Instruct-2507
  # OK: patch.json — 1 point, digest cc2e2500fac13029…

  uv run causalab explain patch.json --data-root data \
      --set model.key=Qwen/Qwen3-4B-Instruct-2507
  # digest    cc2e2500fac130298f0513e6f836da2d16056660147fdbd03f1e37e65427e4cf
  # model     Qwen/Qwen3-4B-Instruct-2507@main fp32
  # points    1
  # requires  ['component:block_output', 'component:block_output:write',
  #            'component:lm_head', 'paired_forward']
  # forwards  2 per point
  #   original on counterfactual: v_cf
  #   patched on base: logits
  # save
  #   iia (model=patched, input=base) -> iia.json
  #   logit_diff (model=patched, input=base) -> logit_diff.json
  ```

- **running it** — just run. `run` touches the model anyway, so it resolves an
  unregistered key from its HF config and registers it before canonicalizing.

`explain`'s `points` and `forwards` are what to size a job against: a sweep of
40 layers is 40 points, and the run cost is roughly points × forwards.
`requires` is the capability set routing matches engines on — every component
the document names appears there, `:write` suffixed where a write targets it.

## 4. Run it

Smoke it on a tiny random model of the same architecture — four layers, hidden
8, hybrid DeltaNet/attention tower, sparse MoE in every layer:

```bash
uv run causalab run patch.json --data-root data --out runs/patch \
    --set model.key=tiny-random/qwen3.5-moe \
    --set sites.target.layer=1 \
    --device cpu
# saved iia.json -> runs/patch/iia.json
# saved logit_diff.json -> runs/patch/logit_diff.json
```

The numbers are meaningless — random weights answer nothing. What this proves is
that the document loads, plans, executes and saves. `--set` is for exploration
only: anything that matters about an experiment belongs in the file, where it
enters the digest.

That run also prints a warning worth reading rather than skipping: MCQA's
answers are single letters, and ` Z` and `Z` are *different* tokens that both
exist. `token_form="auto"` takes the space-prefixed form and says so; set the
metric's `token_form` to `bare` or `space_prefixed` once you know which one the
model actually emits.

Then the real thing, on an accelerator:

```bash
uv run causalab run patch.json --data-root data --out runs/patch \
    --device cuda --dtype bf16 --engine auto
```

| flag | why |
|---|---|
| `--device` | placement is execution, not a document fact |
| `--dtype` | shorthand for `--set model.dtype=…`; precision **is** a document fact, so it enters the digest |
| `--engine` | `pytorch_hooks` (default), `nnsight`, or `auto` — see [§6](#6-engines-and-routing) |
| `--points START:STOP` | execute one half-open slice of an expanded sweep; the seam to shard a campaign on |
| `--resume` | reuse completed outputs whose inputs and code hash are unchanged |


## 5. Hookpoints on Qwen3.6-35B-A3B

The tower is **40 layers on a repeating 3+1 schedule**
(`full_attention_interval: 4`): layers 3, 7, … 39 carry a gated full-attention
mixer, the other 30 carry a Gated DeltaNet (linear-attention) mixer. Both kinds
carry the **same** MLP — a sparse MoE of 256 experts routed top-8, plus a shared
expert that runs on every token.

| | |
|---|---|
| layers | 40 — 30 `linear_attention`, 10 `full_attention` |
| hidden size | 2048 |
| full attention | 16 query heads, 2 KV heads (GQA), `head_dim` 256, output-gated, partial RoPE (0.25) |
| Gated DeltaNet | 16 key heads, 32 value heads (GVA), `d_k` = `d_v` = 128, causal conv kernel 4, 64-token chunked kernel |
| MoE | 256 experts, top-8, `moe_intermediate_size` 512; shared expert 512 |

Which mixer a layer carries is read off the module that is really there, never
off a config flag, and a site that names the wrong one is refused at load:

```json
{"component": "attention_probs", "layer": 3}    // ✓ layer 3 is full attention
{"component": "attention_probs", "layer": 4}    // ✗ [P4] a Gated DeltaNet block
                                                //     computes no attention matrix
{"component": "block_output", "layer": 4, "stream": "linear_attention"}  // ✓ optional, checked
```

### The table

**Reading the columns.** *blocks* is which of the two block types the tensor
exists in (and how many such layers the tower has). *shape* is the component's
declared axes — `head·feature` means head-major and already flattened,
`batch·position` means the MoE block's flattened token axis. *tap* is the
mechanism the engine reaches it by, which is what the engine column follows
from. *write* is the policy: a refusal names the alternative rather than just
saying no.


**Model boundary (no `layer`)**

| component | blocks | shape | tap | engines | write |
|---|---|---|---|---|---|
| `input_ids` | — (layer-less) | `(batch, position)` | module input | both | read-only — token ids are not an activation — edit the row's text, or write `embeddings` |
| `embeddings` | — (layer-less) | `(batch, position, feature)` | module output | both | any mechanism |
| `ln_final` | — (layer-less) | `(batch, position, feature)` | module output | both | any mechanism |
| `lm_head` | — (layer-less) | `(batch, position, feature)` | module output | both | any mechanism |

**Residual stream — every layer**

| component | blocks | shape | tap | engines | write |
|---|---|---|---|---|---|
| `block_input` | every layer (40) | `(batch, position, feature)` | module input | both | any mechanism |
| `attention_input_norm` | every layer (40) | `(batch, position, feature)` | module output | both | any mechanism |
| `attention_output` | every layer (40) | `(batch, position, feature)` | module output | both | any mechanism |
| `block_mid` | every layer (40) | `(batch, position, feature)` | module input | both | any mechanism |
| `mlp_input_norm` | every layer (40) | `(batch, position, feature)` | module output | both | any mechanism |
| `mlp_input` | every layer (40) | `(batch, position, feature)` | module input | both | any mechanism |
| `mlp_output` | every layer (40) | `(batch, position, feature)` | module output | both | any mechanism |
| `block_output` | every layer (40) | `(batch, position, feature)` | module output | both | any mechanism |

**Full-attention mixer interior — the 10 `full_attention` layers**

| component | blocks | shape | tap | engines | write |
|---|---|---|---|---|---|
| `attention_query_pre_rope` | full-attn (10) | `(batch, position, head·feature)` | module output | both | any mechanism |
| `attention_key_pre_rope` | full-attn (10) | `(batch, position, head·feature)` | module output | both | any mechanism |
| `attention_value_states` | full-attn (10) | `(batch, position, head·feature)` | module output | both | any mechanism |
| `attention_gate` | full-attn (10) | `(batch, position, head·fused·feature)` | module output | both | any mechanism |
| `attention_query` | full-attn (10) | `(batch, head, position, feature)` | attention-function slot | both | any mechanism |
| `attention_key` | full-attn (10) | `(batch, head, position[key], feature)` | attention-function slot | both | any mechanism |
| `attention_scores` | full-attn (10) | `(batch, head, position[query], key_position[key])` | attention-function slot | both | any mechanism |
| `attention_probs` | full-attn (10) | `(batch, head, position[query], key_position[key])` | module output | both | `swap` only — its rows are a distribution and nothing renormalizes after an edit — use `attention_scores` for arithmetic |
| `attention_z` | full-attn (10) | `(batch, position, head, feature)` | attention-function slot | both | any mechanism |
| `attention_premix` | full-attn (10) | `(batch, position, head·feature)` | module input | both | any mechanism |
| `attention_result` | full-attn (10) | `(batch, position, head·feature)` | derived from `attention_premix` | both | read-only — derived, never formed by the model — write `attention_premix` with the same `head` |

**Gated DeltaNet mixer interior — the 30 `linear_attention` layers**

| component | blocks | shape | tap | engines | write |
|---|---|---|---|---|---|
| `delta_qkv` | DeltaNet (30) | `(batch, position, feature)` | module output | `pytorch_hooks` | any mechanism |
| `delta_gate` | DeltaNet (30) | `(batch, position, head·feature)` | module output | `pytorch_hooks` | any mechanism |
| `delta_conv` | DeltaNet (30) | `(batch, feature, position)` | delta-kernel boundary | `pytorch_hooks` | any mechanism |
| `delta_query` | DeltaNet (30) | `(batch, position, head, feature)` | delta-kernel boundary | `pytorch_hooks` | any mechanism |
| `delta_key` | DeltaNet (30) | `(batch, position, head, feature)` | delta-kernel boundary | `pytorch_hooks` | any mechanism |
| `delta_value` | DeltaNet (30) | `(batch, position, head, feature)` | delta-kernel boundary | `pytorch_hooks` | any mechanism |
| `delta_beta` | DeltaNet (30) | `(batch, position, head)` | delta-kernel boundary | `pytorch_hooks` | any mechanism |
| `delta_decay` | DeltaNet (30) | `(batch, position, head)` | delta-kernel boundary | `pytorch_hooks` | any mechanism |
| `delta_kernel_output` | DeltaNet (30) | `(batch, position, head, feature)` | delta-kernel boundary | `pytorch_hooks` | any mechanism |
| `delta_premix` | DeltaNet (30) | `(batch, position, head·feature)` | module input | `pytorch_hooks` | any mechanism |
| `delta_kv_mem` | DeltaNet (30) | `(batch, position, head, feature)` | delta-kernel boundary | `pytorch_hooks` | read-only — a memory readout, recomputed each step — write `delta_state` or `delta_value` |
| `delta_state_update` | DeltaNet (30) | `(batch, position, head, feature)` | delta-kernel boundary | `pytorch_hooks` | read-only — lowers onto a state edit; deferred — write `delta_state` |
| `delta_state` | DeltaNet (30) | `(batch, position[steps], head, state, state)` | delta-kernel boundary | `pytorch_hooks` | any mechanism |
| `deltanet_qkv` | DeltaNet (30) | `(batch, position, feature)` | `.source` line (fused forward) | `nnsight_tracing` | any mechanism |
| `deltanet_gate` | DeltaNet (30) | `(batch, position, head, feature)` | `.source` line (fused forward) | `nnsight_tracing` | any mechanism |
| `deltanet_qkv_conv` | DeltaNet (30) | `(batch, feature, position)` | `.source` line (fused forward) | `nnsight_tracing` | any mechanism |
| `deltanet_query` | DeltaNet (30) | `(batch, position, head, feature)` | `.source` line (fused forward) | `nnsight_tracing` | any mechanism |
| `deltanet_key` | DeltaNet (30) | `(batch, position, head, feature)` | `.source` line (fused forward) | `nnsight_tracing` | any mechanism |
| `deltanet_value` | DeltaNet (30) | `(batch, position, head, feature)` | `.source` line (fused forward) | `nnsight_tracing` | any mechanism |
| `deltanet_beta` | DeltaNet (30) | `(batch, position, feature)` | `.source` line (fused forward) | `nnsight_tracing` | any mechanism |
| `deltanet_decay` | DeltaNet (30) | `(batch, position, feature)` | `.source` line (fused forward) | `nnsight_tracing` | any mechanism |
| `deltanet_core_out` | DeltaNet (30) | `(batch, position, head, feature)` | `.source` line (fused forward) | `nnsight_tracing` | any mechanism |
| `deltanet_gated_out` | DeltaNet (30) | `(batch, position, head·feature)` | `.source` line (fused forward) | `nnsight_tracing` | any mechanism |
| `deltanet_state` | DeltaNet (30) | `(batch, position[chunk], head, feature)` | `.source` line (fused forward) | `nnsight_tracing` | any mechanism |

**Sparse MoE + shared expert — every layer**

| component | blocks | shape | tap | engines | write |
|---|---|---|---|---|---|
| `router_logits` | every layer (40) | `(batch·position, feature)` | module output | both | read-only — the MoE block discards them — write `router_scores` or `expert_idx` |
| `router_scores` | every layer (40) | `(batch·position, topk)` | module output | both | any mechanism |
| `expert_idx` | every layer (40) | `(batch·position, topk)` | module output | both | `swap` only — integer expert ids: arithmetic on labels routes to arbitrary experts |
| `expert_permutation` | every layer (40) | `(batch·position, topk)` | `.source` line (fused forward) | `nnsight_tracing` | read-only — the serving kernel's row bookkeeping — write `expert_idx` or `router_scores` |
| `expert_gate_proj` | every layer (40) | `(batch·position, topk·fused·feature)` | grouped-experts dispatch | both | any mechanism |
| `expert_up_proj` | every layer (40) | `(batch·position, topk·fused·feature)` | grouped-experts dispatch | both | any mechanism |
| `expert_activation` | every layer (40) | `(batch·position, topk·feature)` | grouped-experts dispatch | both | any mechanism |
| `expert_output` | every layer (40) | `(batch·position, topk·feature)` | grouped-experts dispatch | both | any mechanism |
| `routed_output` | every layer (40) | `(batch·position, feature)` | module output | both | any mechanism |
| `shared_expert_gate_proj` | every layer (40) | `(batch·position, feature)` | module output | both | any mechanism |
| `shared_expert_up_proj` | every layer (40) | `(batch·position, feature)` | module output | both | any mechanism |
| `shared_expert_activation` | every layer (40) | `(batch·position, feature)` | module input | both | any mechanism |
| `shared_expert_output` | every layer (40) | `(batch·position, feature)` | module output | both | any mechanism |
| `shared_expert_gate` | every layer (40) | `(batch·position, feature)` | module output | both | any mechanism |

### One entry in the vocabulary that this architecture has no tensor for

`mlp_activation` is a Llama-family box: the output of the MLP's `act_fn`. The
A3B's MLP is a sparse-MoE block whose children are
`['experts', 'gate', 'shared_expert', 'shared_expert_gate']` — there is no
`act_fn` to tap, and both engines refuse it by name. Its analogues here are
`expert_activation` (inside the routed experts) and `shared_expert_activation`
(the shared expert's `down_proj` input).

### `delta_*` and `deltanet_*` are the same tensors

The DeltaNet interior appears twice in the table because the two engines reach
it by unrelated mechanisms and each names what it can serve. The reference
engine swaps the modeling file's kernel globals for the extent of one mixer
forward (`delta_*`); the nnsight engine drills `.source` inside the fused
forward (`deltanet_*`). 📐 Measured on the fixture and asserted in both test
tiers, the pairs agree:

| `pytorch_hooks` | `nnsight_tracing` | how they line up |
|---|---|---|
| `delta_qkv` `delta_conv` `delta_gate` `delta_value` `delta_beta` `delta_decay` `delta_kernel_output` `delta_premix` | `deltanet_qkv` `deltanet_qkv_conv` `deltanet_gate` `deltanet_value` `deltanet_beta` `deltanet_decay` `deltanet_core_out` `deltanet_gated_out` | identical — same shape, max abs diff 0.0 |
| `delta_query` `delta_key` | `deltanet_query` `deltanet_key` | `delta_*` is **post** GVA `repeat_interleave` (32 value heads); `deltanet_*` is **pre** (16 key heads). Exact after tiling. |
| `delta_state` | `deltanet_state` | per **step** vs per 64-token **chunk**; the chunk's state is the step-state at the chunk's last position |

So pick the vocabulary your engine serves, not the one that sounds right — and
if you need per-step state, that is `delta_state` on the reference engine only.

### Reading state and attention is expensive

Two components have a position axis that is not the token axis, and both cost
real memory on the A3B:

- `delta_state` is one `d_k × d_v` matrix per head per step — 30 layers ×
  seq × 32 × 128 × 128 floats if you ask for every layer at `pos: "all"`.
  Address positions in the read, not afterwards: the gather runs before
  anything is kept.
- `attention_scores` / `attention_probs` have **two** position axes (query and
  key), so an integer `pos` is ambiguous and refused. Read them whole.

## 6. Engines and routing

Two engines implement the same protocol. A document does not name one — it
declares what it needs, and `choose_engine` takes the first engine in the list
whose capabilities cover it (`--engine auto` puts the reference engine first).

| | `pytorch_hooks` | `nnsight_tracing` |
|---|---|---|
| how | `register_forward_hook` / pre-hook, plus global swaps for the delta kernel and the experts dispatch | one trace over an envoy tree, `.source` for fused-forward interiors |
| capabilities | `grad` `paired_forward` `full_logits` `generate` `pytorch_fn_local` `quantized_weights` `writable_attention_probs` | `paired_forward` `full_logits` `generate` `pytorch_fn_local` `writable_attention_probs` |
| components | 50 of 62 — everything but `deltanet_*` and `expert_permutation` | 49 of 62 — everything but `delta_*` |
| serves alone | the Gated DeltaNet kernel interior (`delta_*`), training (`train` documents need `grad`), quantized weights | the fused-forward interiors: `deltanet_*`, `expert_permutation` |
| install | always | `uv sync` (dev group) or the `nnsight` extra |

Both engines run **one device per run** (no `device_map` sharding) and one batch
per forward group. A `train` document routes to the reference engine because
only it declares `grad`.

The two engines' answers are asserted to agree over the whole shared vocabulary,
read and written, at both test tiers —
`tests/neural/engines/nnsight_tracing/test_parity_a3b_sweep.py` on the tiny
fixture and `tests/golden/test_a3b_engine_parity.py` on the real checkpoint.

## 7. Running at scale

Scale is not document vocabulary: a document never names a device, a host or a
scheduler. Sharding is `--points`, and job dispatch is site tooling.

```bash
#!/usr/bin/env bash
#SBATCH --job-name=mcqa-patch
#SBATCH --gres=gpu:2          # ~70 GB of bf16 weights + KV/state headroom
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x_%j.out
set -euo pipefail

uv run causalab run patch.json \
    --data-root data --out "runs/patch" \
    --device cuda --dtype bf16 --engine auto
```

Shard a sweep by point range — `explain` tells you how many there are:

```bash
#SBATCH --array=0-9           # explain says 40 points -> 10 shards of 4
START=$(( SLURM_ARRAY_TASK_ID * 4 ))
uv run causalab run scan.json \
    --data-root data --out "runs/scan/shard_${SLURM_ARRAY_TASK_ID}" \
    --points "${START}:$(( START + 4 ))" \
    --device cuda --dtype bf16
```

Each point's digest is the provenance unit, so shards are independent and their
outputs merge by coordinate.

## 8. Chaining documents: workflows

A workflow document chains protocol steps with `script` steps between them —
select the best layer from a scan, fit a PCA, plot a curve — and the runner
resolves the dependency graph. See
[`workflow_protocol.md`](workflow_protocol.md).

The shipped one, `causalab/configs/workflows/weekdays_8b.json`: scan 64 points
for the layer that carries the variable, select it, fit a DAS rotation over a
subspace sweep, select the best `k`, apply it — with two figures along the way.

```bash
uv run causalab explain causalab/configs/workflows/weekdays_8b.json \
    --data-root tests/protocol/fixtures/data
# digest    2cf5fd55f79d4c97fa70993248db3734255ad137ea32903cd287d06b584d71da
# schedule  5 levels
#   level 0: locate
#   level 1: best, scan_heatmap
#   level 2: fit
#   level 3: best_fit, iia_by_k
#   level 4: apply
#   locate: intervention_protocol ../protocols/weekdays_locate_scan.json — 64 point(s), campaign digest 83e27be6b471895c…
#   best: script causalab.workflow.scripts.select -> values.json
#   fit: intervention_protocol ../protocols/weekdays_das_sweep.json — 9 point(s), authored digest 890ba99b6ee1c860…
#   best_fit: script causalab.workflow.scripts.select -> values.json
#   apply: intervention_protocol ../protocols/weekdays_das_apply.json — 1 point(s), authored digest 894548f40b29ac94…
#   scan_heatmap: script causalab.io.plots.workflow_figures -> scan_iia.json, scan_iia.png
#   iia_by_k: script causalab.io.plots.workflow_figures -> iia_by_k.json, iia_by_k.png

uv run causalab run causalab/configs/workflows/weekdays_8b.json \
    --data-root tests/protocol/fixtures/data \
    --out runs/weekdays --device cuda --dtype bf16
```

`explain` on a workflow is the same pre-flight as on a document, one level up:
the schedule is derived from the steps' references, so a level is what can run
in parallel, and each protocol step reports its own point count — 64 + 9 + 1
forward groups is what to size the job against.

`--resume` reuses a step whose inputs *and* script content hash are unchanged;
editing a script busts its reuse, which is why the hash is in the digest.

## 9. Where to look next

| you want | read |
|---|---|
| the normative document spec | [`intervention_protocol.md`](intervention_protocol.md) |
| chaining documents | [`workflow_protocol.md`](workflow_protocol.md) |
| the module map and layering rules | [`CODEBASE.md`](CODEBASE.md) |
| test tiers and pinned-artifact discipline | [`TESTS.md`](TESTS.md) |
| worked documents | `causalab/configs/protocols/*.json`, `causalab/configs/workflows/` |
| a picture of the hookpoints above | [`../playground/qwen36-35b-a3b-architecture.html`](../playground/qwen36-35b-a3b-architecture.html) |
