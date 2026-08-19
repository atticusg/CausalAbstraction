# Intervention Protocol — specification v1

The **Intervention Protocol** defines causal intervention experiments on neural networks, with some useful properties:

- **Serializable JSON document that's easily shareable and reproducible**: The intervention protocol is one JSON document that fully describes an experiment: it can be hashed, diffed, shared, and re-run. It is self-contained and enables exact reproduction.
- **Agnostic to neural network interfaces.**: The intervention protocol defines the experiment, not the implementation. The protocol is passed to a separate parser/compiler compiling the protocol into runnable code of intervention backends (like pytorch-native hooks, NNsight/NNterp, SGLang, Megatron, etc.). The protocol says *what*; backend specific parser/planner derives *how* (forward count, fusion, batching, sweep parallelization) and execute an intervention protocol.

## 1. Document layout

Sections in this order (order enforced; `save` last):

| # | key | required | content |
|---|---|---|---|
| 1 | `version` | ✓ | `"1"` |
| 2 | `description` | – | free text, the file's intent |
| 3 | `model` | ✓ | the neural network ℒ (alias: `neural_model`) |
| 4 | `causal_model` | – | the high-level algorithm ℋ, provenance only |
| 5 | `data` | ✓ | input rows: `base` (+ `source`) |
| 6 | `positions` | – | named token-position specs |
| 7 | `sites` | ✓ | named activation addresses — the complete tap inventory |
| 8 | `featurizers` | – | named feature-space maps |
| 9 | `params` | – | free/constant tensors owned by no featurizer |
| 10 | `reads` | ✓ | value producers |
| 11 | `edits` | – | effect definitions (inert until listed) |
| 12 | `intervened_models` | –* | which edits are in force on which input (*required if `edits` present) |
| 13 | `metrics` | – | closed reductions over read values |
| 14 | `train` | – | the fit, declared |
| 15 | `save` | ✓ | the complete output manifest — non-empty, last |

- **One global namespace**: every name in sections 6–13 must be unique across
  all of them; reserved names: `base`, `source`, `source[j]`, `original`.
- All cross-references must resolve; references are by name, never inline
  duplication.
- **Artifact-valued fields**: anywhere a scalar or position is expected,
  `{"artifact": "<path>", "key": "<field>"}` reads one value from a prior
  run's artifact at load. Missing artifact = load error.

## 2. Section reference

### 2.1 `model`, `causal_model`

| field | meaning |
|---|---|
| `model.key` | model name (HF key or registry name) — the network as a *name* |
| `model.revision` | checkpoint revision |
| `causal_model.key` | name of the high-level causal model (provenance; not executed) |

- `neural_model` is accepted as an alias of `model`; canonical form uses `model`.

### 2.2 `data`

```json
"data": {
  "base":   {"dataset": "weekdays/train", "field": "input"},
  "source": {"dataset": "weekdays/train", "field": "counterfactual_inputs[0]"}
}
```

- `dataset`: a **local path or HF key — no digest**. The parser resolves it at
  load and stamps the content digest into the canonical form (sec. 7).
- Roles are the keys: `base` (required) and `source` (optional). `source` is
  singular; if its value is an array, references index it as `source[j]`.
  Rows are paired: one base row + its source row(s) form one example.
- `field` selects the column; `[j]` indexes list-valued columns.
- Dataset **columns** referenced by metrics are checked against the table at
  run time (`validate --data`), not at load.

### 2.3 `positions`

Named entries; a read/edit `pos` is a name here, or an inline spec.

| form | resolves to |
|---|---|
| `-1` (bare int, sugar) | `{"index": -1}` |
| `{"index": n}` | one token per row. `n < 0` counts from the end of the sequence; `n ≥ 0` is rebased past any chat prefix |
| `{"variable": "x"}` | all tokens of prompt variable `x` — a per-row window, ragged across rows |
| `{"span": [a, b]}` | fixed window `[a, b)` |
| + `"scope": {"variable": "x"}` | interpret the index/span inside `x`'s span |
| + `"relative_to": {"variable": "x"}` | offset from `x`'s span |

- Positions are **never resolved to integers in the document**. Resolution is
  a backend service against a `PositionFrame` (pad side, packing, sequence
  shard map) — sec. 8.

### 2.4 `sites`

```json
"target": {"component": "block_output", "layer": 18}
```

| field | meaning |
|---|---|
| `component` | one of the vocabulary below |
| `layer` | depth index (where the component has one) |
| `head` / `expert` / `stream` | optional sub-axes: attention head, MoE expert, residual-stream index |

Component vocabulary (per-backend `SiteResolver` maps each to a tap):

`embeddings` · `block_input` · `block_output` · `attention_output` ·
`attention_value` · `attention_probs` · `mlp_input` · `mlp_output` ·
`mlp_activation` · `router_logits` · `expert_output` · `ln_final` · `lm_head`

- **`sites` is the complete inventory**: every site a read or edit references
  must be declared here, including `lm_head` (`{"component": "lm_head"}`).
  There are no implicit site names.
- Sites are pure data — no behavior, no model handles.

### 2.5 `featurizers`

`featurize(x) → (f, err)`; `inverse(f, err) → x̂`; both defined per kind.

| kind | featurize | param slots | authored fields |
|---|---|---|---|
| `identity` (default) | `(x, 0)` | — | — |
| `subspace` | `(Qᵀx, 0)` | `weight` | `k`, `parametrization` ∈ `cayley` \| `matrix_exp` \| `stiefel` |
| `pca` | `(Pᵀx, 0)` | `weight` | `k`, `file_path` |
| `sae` | `(enc(x), x − dec(enc(x)))` | `enc, dec, b_enc, b_dec` | `file_path` |
| `standardize` | `((x−μ)/σ, 0)` | `mu, sigma` | `file_path` |
| `gate` | `(σ(θ)⊙x, (1−σ(θ))⊙x)` | `theta` | — |

- **Widths are derived** from (model, site) — never authored. Only choices
  (`k`, `parametrization`, `dtype`, `init`) are authored.
- **Params are auto-declared** per kind, named `<featurizer>.<slot>`.
- **Composition**: a `featurizer` reference may be a list `["rot", "gate"]`,
  applied left-to-right with a per-stage `err` list.
- **Error-term contract**: `err` and unselected dims always come from the
  pre-edit value at the address — so a zero write ablates only the feature
  contribution, and a `dims` write is a subspace swap.
- **`file_path`** (optional): load a fitted artifact instead of computing.
  Its `ArtifactIdentity` (sec. 8) is checked; mismatch refuses. A loaded
  featurizer may not appear in `train.params`.

### 2.6 `params` (optional)

Free tensors owned by no featurizer (steering vectors, a free written value):

| field | meaning |
|---|---|
| `file_path` | constant tensor, loaded |
| `shape`, `init` | trainable free tensor (must then appear in `train.params`) |

### 2.7 `reads`

```json
"v_src":  {"site": "target",  "pos": -1, "model": "original", "input": "source"},
"logits": {"site": "lm_head", "pos": -1, "model": "patched",  "input": "base"}
```

| field | meaning |
|---|---|
| `site`, `pos` | the address |
| `model` | `original` (un-intervened) or a declared intervened_model |
| `input` | `base` \| `source` \| `source[j]`. For an intervened model this is redundant with the IM's own `input` and is **cross-checked** — mismatch is a load error |
| `featurizer` | optional; value is read in feature space |
| `dims` | optional static index list into the feature axis; default = all |

- Value = `featurize(activation at (site, pos) in model)[dims]`.
- A read in model `M` sees the activation **with all of `M`'s edits applied**
  (upstream and at the same address). To read an un-edited value, read in
  `original` (or an IM without that edit).
- Reads never carry `do`.

### 2.8 `edits` and the `do` algebra

```json
"patch": {"site": "target", "pos": -1, "featurizer": "rot", "do": {"swap": "v_src"}}
```

- An edit is an **inert definition**: no `model`, no `input`, no conditions.
  It executes inside every intervened_model that lists it.
- Effect at its address: `write(inverse(scatter(do(f[dims]) into f), err))` —
  untouched dims and `err` from the pre-edit value (sec. 2.5).
- `Operand` = a read name · a param name (`rot.weight`, or a `params` entry) ·
  a literal scalar. **Never a tensor, never a closure** — constant vectors
  enter as `params` entries.

Closed mechanism set (`do` has exactly one key):

| `do` | write | class |
|---|---|---|
| `{"swap": op}` | `f ← op` | absolute |
| `{"add_scaled": {"op": op, "alpha": a}}` | `f ← f + a·op` | additive |
| `{"lerp": {"op": op, "alpha": a}}` | `f ← (1−a)·f + a·op` | absolute |
| `{"affine": {"A": param, "b": param}}` | `f ← Af + b` | absolute |
| `{"gaussian": {"seed": s, "scale": c, "axis": "tp_duplicated" \| "tp_split"}}` | `f ← f + c·randn(s)` | additive |
| `{"renormalize": true}` | `f ← f·‖f₀‖/‖f‖` | absolute |
| `{"clamp": {"lo": a, "hi": b}}` | `f ← clip(f, a, b)` | absolute |
| `{"pytorch_fn": {"qualname": "…"}}` | arbitrary | absolute; **local-only** — refused at load by any non-local backend |

- Per (site, overlapping pos, model): **at most one absolute edit**; any
  number of additive edits. Application order: absolute first, then additive
  deltas summed. This replaces any commutativity analysis and makes edit sets
  order-free.
- `gaussian.axis` tells a tensor-parallel backend whether the draw is
  replicated or sharded across ranks; `seed` is part of the hash.

### 2.9 `intervened_models`

```json
"patched": {"input": "base", "edits": ["swap_sender", "freeze_10", "freeze_11"]},
"final":   {"input": "base", "edits": ["inject"]}
```

| field | meaning |
|---|---|
| `input` | **mandatory** — `base` \| `source` \| `source[j]` |
| `edits` | the edits in force; **unordered** (canonical form sorts) |

- `original` is the reserved name for the un-intervened model (on any input);
  it is never declared.
- **Membership rule**: every declared edit appears in ≥ 1 intervened_model.
- **Cross-model data flow has exactly one channel**: a read in model A may be
  the operand of an edit in force in model B. No direct IM→IM wiring, no
  inheritance. The graph (IM → edits → operand reads → IMs) must be acyclic —
  it is the execution schedule's skeleton.

### 2.10 `metrics`

Closed vocabulary; `of` names a read; other value fields name dataset columns.

| kind | fields | result per example |
|---|---|---|
| `logit_diff` | `of, a, b` | `logits[a] − logits[b]` |
| `token_logit` | `of, token` | `logits[token]` |
| `cross_entropy` | `of, target` | CE against target |
| `kl` | `of, target` (a read) | KL between two reads' distributions |
| `class_probs` | `of, groups` | summed probability per group |
| `top_k` | `of, k` | top-k tokens + probs |
| `match` | `of, expected` | exact-match indicator |

- A metric binds to exactly one read → one (model, input). Same metric in two
  models = two reads + two metrics.
- Metrics are gather-then-reduce over read values and dataset columns —
  nothing else. Cross-read arithmetic (differences of saved metrics) is
  post-hoc analysis. The vocabulary stays closed so backends can lower kinds
  to fused/vocab-parallel implementations.

### 2.11 `train`

| field | meaning |
|---|---|
| `objective` | `[[weight, metric_or_reg], …]`; reg = `{"l1": name}` \| `{"l2": name}` where name is a featurizer (all its params) or a dotted slot |
| `params` | what is optimized: featurizer names (all slots) or dotted slots; the **only** trainability declaration |
| `optimizer` | `{name, lr, …}` — lr/schedule/clip live here |
| `steps` | `{"epochs": n}` or `{"updates": n}` |
| `batch` | `{"pairs": n}` — counts base+source **pairs**, not rows |
| `anneal` | dotted-path schedules, e.g. `{"gate.theta.temperature": [start, end, frac]}` |
| `precision` | `{feature, loss, model}` dtypes |
| `eval` | `{every, split, metrics}` |
| `early_stop` | `{metric, patience, mode}` |
| `checkpoint` | transient training state (resume); the final artifact is the `save` entry |
| `seed` | init + data order |

- `train` present ⇒ the run requires gradients (sec. 8).
- Every trained featurizer must have a `save` entry (sec. 2.12).

### 2.12 `save`

Mandatory, non-empty, the last section. The **complete manifest** of
everything that leaves the run. Three saveable kinds, two entry shapes:

| kind | entry |
|---|---|
| read / metric | `{"value": name, "model": …, "input": …, "file_path": …}` |
| trained featurizer | `{"value": name, "site": …, "file_path": …}` |

- `model`/`input` (resp. `site`) **restate** the binding resolved from the
  declarations and are **cross-checked** — mismatch is a load error. They are
  drift-protected documentation, never a second source of truth.
- `file_path` is relative to the run's output directory. Tensors →
  `.safetensors`; per-example metric tables → `.parquet`. In swept documents
  the path is unchanged; axis coordinates become columns / keyed entries.
- Rules (all load errors): every metric saved (no objective/eval exception —
  the loss trajectory is always in the results) · every trained featurizer
  saved · untrained or `file_path`-loaded featurizers not saveable · edits
  and intervened_models not saveable.

## 3. Sweeps

- **Every axis is an explicit wrapper** on a field of a named table entry:
  `{"sweep": [v1, v2, …]}` or `{"sweep": {"range": [start, stop, step?]}}`.
  Bare arrays are never axes. Works on scalar- and list-typed fields alike.
- **Axis identity = name identity**: sweeping `sites.target.layer` moves every
  read/edit/metric referencing `target` together. The same list written on
  two fields is two axes (a cross) — share by referencing one name.
- Axes **propagate through the reference graph**; entities off the axis stay
  singletons shared by all points.
- Multiple axes form the **cross product**; nothing else (no zip, no
  conditionals; dependent axes are a generator's job — emit the JSON).
- Coordinates suffix derived names (`rot[k=8]`) and key results.
- Expansion is **deterministic at load**: one document ⇒ a set of point
  protocols. The document digest names the campaign; each point's digest is
  the provenance unit. The planner content-dedups sub-values shared across
  points — shared harvests and forwards fall out automatically (identical
  reads intern to one read).

## 4. Execution semantics

- **Models → forwards.** For each expanded point, the models are: `original`
  on every input it is read on, plus each intervened_model. Each (model)
  is one forward group over its input rows; fusion, batching, and staging
  across groups are the backend's choice. `num_forwards` is derived, never
  authored.
- **Within a model**: apply each in-force edit at its address (absolute
  first, then additive sum); reads see the fully edited state.
- **Across models**: operand values flow along the acyclic model graph;
  the backend stages them (fused multi-pass, saved constants, or microbatch
  wiring — its call).
- **Elision**: a model whose reads are all satisfied may stop its forward
  after the deepest tap; a full-depth pass is never owed.
- **Determinism**: `gaussian` draws from its declared seed; sweep expansion
  and canonicalization are pure functions of the document.

## 5. Validation — load-error checklist

A conforming loader rejects the document unless all of these hold:

1. Strict keys: unknown fields anywhere are errors; closed enums reject with
   suggestions. Derived fields (sec. 7) may not be authored.
2. Section order per sec. 1; `save` last.
3. Global namespace: no duplicate names across sections 6–13; no reserved
   names (`base`, `source`, `source[j]`, `original`) declared.
4. Every reference resolves: sites (declared inventory only), positions,
   featurizers, params, reads, edits, intervened_models, metrics.
5. Reads: `model` ∈ `original` ∪ IMs; `input` a valid role; if `model` is an
   IM, `input` equals the IM's `input`.
6. Edits carry no `model`/`input`/conditions; operands name reads, params, or
   literal scalars.
7. Every edit is in ≥ 1 intervened_model; every IM has a mandatory valid
   `input`; the model graph is acyclic.
8. Per (site, overlapping pos, model): ≤ 1 absolute edit.
9. `dims` selections co-occurring at one address in one model are disjoint.
10. `save` non-empty; entry shapes exact; bindings match resolution; every
    metric and every trained featurizer saved; nothing else saveable.
11. Sink rule: every read is saved, a metric input, or an operand.
12. Loaded featurizers (`file_path`) are not trained; trained featurizers are
    declared kinds with trainable slots.
13. `pytorch_fn` present ⇒ refused unless the selected backend is local.
14. Sweep wrappers well-formed; the expanded point count is reported (and may
    be capped without an explicit override flag).
15. Artifact-valued fields resolve (missing artifact = error, never a
    default).

## 6. Derived — never authored

| property | derivation |
|---|---|
| featurizer widths, param shapes | from (model config, site); parametrization internals are not authored |
| param slots | per featurizer kind (sec. 2.5) |
| `requires` | capability set, sec. 8 |
| `num_forwards`, fusion, staging | from the model graph; a compile property |
| dataset content digest | resolved + stamped at load |
| point protocols + digests | deterministic sweep expansion |
| `ArtifactIdentity` | stamped into artifacts, sec. 8 |

## 7. Canonical form and digests

- **An experiment is a value, not a program.** One JSON document fully
  describes an experiment: it can be hashed, diffed, shared, and re-run.
  It never contains tensors, closures, resolved token indices, module
  references, or anything only one backend could interpret.
- **The parser owns execution.** The document says *what*; the parser/planner
  derives *how* (forward count, fusion, batching, sweep parallelization) and
  `explain` reports it.
- **Everything declared must reach a sink; everything derivable is derived.**
  Dead declarations are load errors. Authored files are minimal; the stamped
  canonical form materializes every default (sec. 7).
- **Format**: strict JSON (unknown keys = error). YAML is accepted at the
  authoring surface; the object model is normative. JSON has no comments —
  use `description`.
- **v1 scope**: prefill-only. No generation, no decode-step edits, one neural
  model per document.
- **Canonical-stamp principle**: the authored file may be minimal; the
  canonical form materializes *everything* — every default (constant LR,
  optimizer betas, dtypes), every resolved reference (dataset digests,
  artifact values), every derived width, sugar expanded (int positions,
  alias `neural_model` → `model`), unordered lists sorted (IM edit lists),
  sweeps expanded to points.
- `digest = sha256(canonical bytes)` — sorted keys, canonical floats; each
  param replaced by its content hash. Document digest = campaign; point
  digest = provenance unit, stamped on every artifact as `produced_by`.
- Any change to canonical form bumps `version` and ships a loader migration.
  Pin a golden corpus (canonical form + digest per example) in tests.

## 8. Backend contract

A backend implements these services:

| service | contract |
|---|---|
| `SiteResolver` | site record → tap in its execution engine (component vocabulary, sec. 2.4) |
| position resolution | Pos spec + `PositionFrame` (pad side, packing, sequence shard map) → indices; supports flat, per-row, and ragged windows |
| planner | model graph → forward groups; fusion/batching/staging; elision |
| mechanisms | the closed `do` set, class order per address; refuse `pytorch_fn` if non-local |
| featurizers | kinds table with declared dtypes; error-term contract |
| metrics | lower kinds to native ops; derive minimal logit materialization (`logits_to_keep`, vocab-parallel CE) from `save` + metric needs |
| training | own the `train` loop (optimizer, accumulation, anneal, early stop, checkpoints) — the document never changes across backends |
| RNG | realize `gaussian` per declared seed + axis semantics, bit-stable across parallelism layouts |
| stamping | write canonical point protocols + digests; `ArtifactIdentity` into every featurizer bundle's safetensors header |

`ArtifactIdentity` (stamped, checked on any `file_path` load; mismatch
refuses): `produced_by` digest · model key + revision · tokenizer · site
record · `k` · parametrization · dtype · trained-on data ref + digest ·
backend · code commit.

**Capabilities.** `requires` is derived from the document; a backend declares
what it supports; `choose_backend = first b where requires ⊆ b.capabilities`;
refusal messages generate from the missing capability.

| capability | required when |
|---|---|
| `grad` | `train` present |
| `paired_forward` | an edit's operand read has a different `input` than the edit's model |
| `full_logits` | a full `lm_head` read is saved, or a metric needs the full vocab (`top_k`, `class_probs`) |
| `editable_attention_probs` | an edit writes `attention_probs` |
| `pytorch_fn_local` | any `pytorch_fn` |

Reference matrix:

| capability | nnsight (HF) | Megatron | SGLang/serving |
|---|---|---|---|
| `grad` | ✓ (≤ ~1 node) | ✓ (the point of it) | ✗ |
| `paired_forward` | ✓ fused invokes | ✓ pairs per microbatch | ✗ |
| arbitrary writes | ✓ | ✓ | additive steering only |
| `full_logits` | ✓ | ✗ vocab-parallel only | ✓ |
| `pytorch_fn_local` | ✓ | ✗ | ✗ |

**Execution scale.** Documents and workflows are scheduler-agnostic — they
never name devices, hosts, or job systems. The division of labor:

- A **backend** owns all intra-run execution: device placement, dtype,
  batching, and any parallelism across a campaign's points or across its own
  accelerators — declared, like everything else, through its capability set
  and constructor. The reference backend takes `device`/`dtype` and runs
  points serially; sharded and multi-device backends are backend work, not
  document vocabulary.
- **Job dispatch is site tooling outside this repository.** The one seam it
  needs is the CLI's `--points START:STOP` selector: an external scheduler
  expands nothing itself, launches `run` per index range, and recombines by
  digest — every shard stamps its artifacts as members of the same campaign
  (`document_digest` is unaffected by slicing).

## 9. CLI

| verb | effect |
|---|---|
| `run <doc>` | validate, expand, plan, execute, stamp |
| `validate <doc> [--data]` | sec. 5 checks; `--data` also checks column references |
| `explain <doc>` | models + forward plan, expanded point count, derived `requires`, resolved bindings, digest, what `save` produces |
| `digest <doc>` | the campaign digest |
| `--set path=value` | ad-hoc override — exploration only; promote anything that matters into the file |
| `--device`, `--dtype` (run) | reference-backend placement: any torch device string (`cpu` default, `cuda`, `cuda:1`, `mps`) and `fp32` (default) \| `bf16` \| `fp16` |
| `--points START:STOP` (run) | execute one half-open point-index shard of the expanded campaign (sec. 8, execution scale); document runs only — digests and stamps are unaffected |

## 10. Worked examples

Path patching (sender → receiver, off-path frozen; shows cross-model flow):

```json
{
  "version": "1",
  "model": {"key": "meta-llama/Llama-3.1-8B", "revision": "main"},
  "data": {
    "base":   {"dataset": "ioi/test", "field": "input"},
    "source": {"dataset": "ioi/test", "field": "counterfactual_inputs[0]"}
  },
  "sites": {
    "sender":   {"component": "attention_value",  "layer": 9, "head": 9},
    "receiver": {"component": "block_input",      "layer": 12},
    "a10":      {"component": "attention_output", "layer": 10},
    "a11":      {"component": "attention_output", "layer": 11},
    "lm_head":  {"component": "lm_head"}
  },
  "reads": {
    "v_sender":   {"site": "sender",   "pos": -1, "model": "original", "input": "source"},
    "v_a10":      {"site": "a10",      "pos": -1, "model": "original", "input": "base"},
    "v_a11":      {"site": "a11",      "pos": -1, "model": "original", "input": "base"},
    "v_receiver": {"site": "receiver", "pos": -1, "model": "patched",  "input": "base"},
    "logits":     {"site": "lm_head",  "pos": -1, "model": "final",    "input": "base"}
  },
  "edits": {
    "swap_sender": {"site": "sender",   "pos": -1, "do": {"swap": "v_sender"}},
    "freeze_10":   {"site": "a10",      "pos": -1, "do": {"swap": "v_a10"}},
    "freeze_11":   {"site": "a11",      "pos": -1, "do": {"swap": "v_a11"}},
    "inject":      {"site": "receiver", "pos": -1, "do": {"swap": "v_receiver"}}
  },
  "intervened_models": {
    "patched": {"input": "base", "edits": ["swap_sender", "freeze_10", "freeze_11"]},
    "final":   {"input": "base", "edits": ["inject"]}
  },
  "metrics": {
    "logit_diff": {"kind": "logit_diff", "of": "logits", "a": "answer", "b": "cf_answer"}
  },
  "save": [
    {"value": "logit_diff", "model": "final", "input": "base", "file_path": "logit_diff.parquet"}
  ]
}
```

DAS with a k × seed sweep (9 fits from one harvest; shows axes + train +
featurizer save):

```json
{
  "version": "1",
  "model": {"key": "meta-llama/Llama-3.1-8B", "revision": "main"},
  "causal_model": {"key": "weekdays.causal_model"},
  "data": {
    "base":   {"dataset": "weekdays/train", "field": "input"},
    "source": {"dataset": "weekdays/train", "field": "counterfactual_inputs[0]"}
  },
  "sites": {
    "target":  {"component": "block_output", "layer": 18},
    "lm_head": {"component": "lm_head"}
  },
  "featurizers": {
    "rot": {"kind": "subspace", "k": {"sweep": [8, 16, 32]}, "parametrization": "cayley"}
  },
  "reads": {
    "v_src":  {"site": "target",  "pos": -1, "model": "original", "input": "source", "featurizer": "rot"},
    "logits": {"site": "lm_head", "pos": -1, "model": "patched",  "input": "base"}
  },
  "edits": {
    "patch": {"site": "target", "pos": -1, "featurizer": "rot", "do": {"swap": "v_src"}}
  },
  "intervened_models": {
    "patched": {"input": "base", "edits": ["patch"]}
  },
  "metrics": {
    "iia": {"kind": "logit_diff",    "of": "logits", "a": "cf_answer", "b": "base_answer"},
    "ce":  {"kind": "cross_entropy", "of": "logits", "target": "label"}
  },
  "train": {
    "objective":  [[1.0, "ce"]],
    "params":     ["rot"],
    "optimizer":  {"name": "adamw", "lr": 1e-3},
    "steps":      {"epochs": 10},
    "batch":      {"pairs": 16},
    "eval":       {"every": {"epochs": 1}, "split": "weekdays/test", "metrics": ["iia"]},
    "early_stop": {"metric": "iia", "patience": 3, "mode": "max"},
    "seed":       {"sweep": [0, 1, 2]}
  },
  "save": [
    {"value": "iia", "model": "patched", "input": "base", "file_path": "iia.parquet"},
    {"value": "ce",  "model": "patched", "input": "base", "file_path": "ce.parquet"},
    {"value": "rot", "site": "target", "file_path": "rot.safetensors"}
  ]
}
```

## 11. Glossary (Geiger et al., arXiv:2301.04709)

| this spec | causal abstraction |
|---|---|
| `model` / `causal_model` | low-level model ℒ / high-level model ℋ |
| `intervened_models.<name>` | ℒ_{b∪𝕀} — the intervened model |
| an edit's `do` | an interventional 𝕀_X |
| `swap` from a source read | interchange intervention (`IntInv`; `DistIntInv` when featurized) |
| site + pos + dims | the target variable set **X** |
| featurizer | the translation τ |
| `match` metric | interchange-intervention accuracy (IIA) |
