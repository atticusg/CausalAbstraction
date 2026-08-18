# Intervention Protocol — cheatsheet

> **Status: proposed, not implemented.** Nothing in this document exists in the tree at
> `a50637c`. It is the target language that would replace `neural/plan.py` as the
> interface. See §10 for what each current object becomes, and `docs/CODEBASE.md` /
> `causalab/neural/README.md` for what actually runs today.

## 1. Why

- `Plan` cannot be shipped, hashed, or retargeted: it holds tokenized device tensors
  (`Plan.inputs`), binds models **by object identity** (`Plan.models`), and its nodes
  execute themselves against nnterp accessors (`Site._proxy`, `FeaturizedSite._rewrite`).
- Three consumers want the same fix, independently:

| Consumer | Needs |
|---|---|
| Megatron / MegaFire | model by name; positions unresolved (SP shards the sequence axis, CP permutes it); no Python closures in the forward |
| NDIF | a trace body from a whitelisted module set (`torch, numpy, transformers, accelerate, einops`); anything else ships by value |
| hydra → experiment | a YAML-round-trippable, hashable target so config→run→artifact is one closed loop |

- **`Protocol` is a description; an nnsight trace is a program.** A program can't answer
  *which sites · needs gradient? · how many forwards · which model produced this artifact*
  without running. Those four are the whole purchase.

## 2. Shape

```
DatasetConfig ─▸ Dataset ─▸ CounterfactualDataset{base, sources[]} ─▸ Input (base | source[j])
                                                                  │
Site ─┐                                                           │
Pos  ─┼──────────────────────────────────────────────▸ Interventional ──▸ Metric ──▸ outputs
Featurizer(+Param) ─┘                                     │  do: Mechanism    (off-model)
                                                          └─ conditions: {nodes}
Protocol{ModelConfig, CausalModelConfig?, Alignment?, …} ──compile──▸ Backend ──▸ Result
```

- **Everything the model touches is one node type** (`Interventional`).
- **Everything off-model is a `Metric`** — pure arithmetic over node values + dataset columns.
- `requires`, `num_forwards`, `digest` are **derived**, never written.

## 3. Primitives

### 3.1 Containers

| Primitive | Fields | Notes |
|---|---|---|
| `Protocol` | `version · model · high_level? · alignment? · counterfactual_dataset · positions · sites · featurizers · params · interventionals · metrics · outputs · objective? · train? · sweep · seeds` | pure, hashable, YAML-round-trippable, model-free |
| `TrainSpec` | `objective · params · optimizer · lr · schedule · steps · batch(pairs) · anneal · precision · clip · eval · early_stop · checkpoint · seed · resume` | the outer loop declared, so a distributed optimizer can own it |

- **No `ExecSpec`.** Backend choice, `device_map`, batch size, `remote=True` are
  `run(protocol, data, backend=…, **backend_kwargs)`. They are backend-shaped and would
  only ever be a union of per-backend dicts.
- **No `Pipeline`.** One protocol per run. Multi-step campaigns stay hydra multirun;
  chains are recoverable post hoc because every artifact stamps the producing digest.
- **Hash rule:** *in the hashed value iff it can change the numbers.* Training batch size
  is in (`TrainSpec`); eval batch size is out — but the backend stamps what it used into
  the result.

### 3.2 Naming rule

- **`*Config` = a handle to something outside the protocol** (a checkpoint on HF, a table
  on disk, an algorithm in `causalab.tasks`).
- Everything defined *inside* the protocol takes no suffix: `Site`, `Pos`, `Featurizer`,
  `Param`, `Interventional`, `Metric`.

### 3.3 Model and data

| Primitive | Fields | Meaning |
|---|---|---|
| `ModelConfig` | `key · revision · tokenizer` | ℒ — the network, **as a name** (nnsight's `model_key`) |
| `CausalModelConfig` | `key · revision` | ℋ — the algorithm, as a name |
| `Alignment` | `Π · π` | which low-level variables realize which high-level variable |
| `DatasetConfig` | `name · digest` | identifies the *table* |
| `Dataset` | `config · field · tokenizer? · chat?` | one *column*, tokenized. `field: "counterfactual_inputs[0]"` |
| `CounterfactualDataset` | `base: Dataset · sources: tuple[Dataset, …]` | schema of one base + k sources. `k` is **derived** from the highest `source[j]` referenced |
| `Input` | `base` \| `source[j]` | one input sequence to the LM |

- `CounterfactualDataset` is the **schema**; `CounterfactualInput` is the **row**.
- `CounterfactualDataset.source` returns the single source and raises when `k ≠ 1`;
  `.sources` always returns the tuple. `k=1` in YAML is sugar that expands on load.
- Two `Dataset`s routinely share one `DatasetConfig` (base and source are two columns of
  one table).

### 3.4 Addressing

| Primitive | Fields | Meaning |
|---|---|---|
| `Site` | `component · layer? · head? · expert? · stream` | where in the depth axis. **Pure data** — a per-backend `SiteResolver` maps it to a tap |
| `Pos` | see §3.4.1 | where on the sequence. **Never resolved to ints inside the IR**; resolution is a backend service against a declared `PositionFrame(pad_side, packed, seq_shard_map)` |
| `Featurizer` | `kind · shape? · parametrization? · constraint? · dtype · init? · params: (ParamName, …)` | the feature space, declared. Weights live in `params`, by name, so two sites can share one rotation |
| `Param` | `name · shape · trainable · constraint? · dtype · init? · sha256?` | a named tensor in `weights.safetensors`. `sha256` is `None` until fitted |

- `Site.component` vocabulary extends today's seven with `router_logits`,
  `expert_output(e)`, `ln_final`, `lm_head`, a residual-stream index, and a per-head → TP-rank map.
- `dims: tuple[int, ...] | None` lives on the **`Interventional`**, not on `Site` or
  `Featurizer` — it is the only one of the three that varies per use. Ranges are YAML sugar.
- **`dims` must be static.** Per-example token ids (e.g. `logit_diff` over `cf_answer`)
  are a `Metric` concern, not a dim selection.

#### 3.4.1 `Pos` vocabulary

A `Pos` resolves to **one or more** token indices per row. Single indices and windows are
the same kind of thing; nothing downstream distinguishes them.

| Spec | Resolves to | Exists at `a50637c` |
|---|---|---|
| `{type: index, position: -1}` | one index; negative counts from the end (the generation slot), non-negative is rebased past any chat prefix | ✅ |
| `{type: index, position: 1, scope: {variable: x}}` | the 2nd token of variable `x` | ✅ |
| `{type: index, position: +1, relative_to: {variable: x}}` | the token after `x` | ✅ |
| `{type: variable, name: x}` | **all** tokens of `x` — a per-row window, ragged across rows | ✅ |
| `{type: span, start: a, stop: b}` | a fixed absolute window `[a, b)` | ❌ **new** |
| `{type: span, scope: {variable: x}, start: 1, stop: -1}` | a window inside a variable's span | ❌ **new** |
| char range → tokens | via `get_tokens_in_char_range` / `get_substring_token_ids` | ✅ |

- **Windows already work.** The resolution machinery carries three index forms — flat row
  broadcast over the batch, per-row `(batch, k)`, and **ragged** `RaggedIndex(row_ids,
  col_ids, widths)` gathered flat — and reads *and* writes are pinned against the raw-hook
  oracle for all three (`tests/neural/test_positions.py`, `test_site.py`).
- The two `span` rows are the only gap: `_build_factory` (`token_positions.py:979`) accepts
  just `index` and `variable`. Adding `span` is one branch returning `range(a, b)` plus the
  same chat-prefix rebasing the absolute-index factory does — ~15 lines and a test.
- **Follow-up, not a blocker:** `AttentionProbabilitiesSite` refuses per-row and ragged
  positions, so a *ragged* window on an attention-pattern site stays unsupported.

### 3.5 `Featurizer.kind`

| kind | `featurize(x) -> (f, err)` | Fits |
|---|---|---|
| `identity` | `(x, 0)` | default |
| `subspace` | `(Qᵀx, 0)` | DAS. `parametrization ∈ {cayley, matrix_exp, stiefel}` — measured **7.2×** between `matrix_exp` and `cayley` at h=8192 |
| `pca` | `(Pᵀx, 0)` | fitted off-model |
| `sae` | `(f_enc(x), x − f_dec(f_enc(x)))` | reconstruction error is the `err` term |
| `standardize` | `((x−μ)/σ, 0)` | |
| `gate` | `(σ(θ)⊙x, (1−σ(θ))⊙x)` | **DBM.** A plain `Swap` in this space computes `σ⊙x_src + (1−σ)⊙x_base` |

- `inverse(f, err) = f + err` for `gate`; round-trips exactly.
- Compose with `a >> b` (per-stage error list). `subspace >> gate` = DAS + DBM stacked.
- **Error-term contract:** the reconstruction error and unselected dims always come from
  **base**. That is what makes a zero write ablate the feature contribution only, and a
  `dims` write a subspace swap.

### 3.6 The one node type

```python
@frozen class Interventional:
    site:       SiteName
    position:   PosName
    input:      Input                            # base | source[j]
    featurizer: FeaturizerName = "identity"
    dims:       tuple[int, ...] | None = None    # None = every dim of the space
    do:         Mechanism = Identity()
    conditions: tuple[InterventionalName, ...] = ()
```

- `conditions` is a **set**, not a sequence — order is never meaningful, because
  everything in it is in force simultaneously (rule 3 forbids order-dependent overlap).
  It is typed as a `tuple` rather than a `frozenset` so it serializes as a YAML list and
  hashes deterministically. Canonical form = the **transitive closure, sorted**; the
  authored form may be the minimal set and is expanded on load.

Semantics — let `s` = the total setting on `input` under `conditions`:

| | |
|---|---|
| **value** | `featurize(read(site, position) in s)[dims]` — the **pre**-transform read |
| **effect** | `write(site, position, inverse(scatter(do(value, *operands), into f), err))` |
| `do=Identity` | the write is a no-op ⇒ pure observation; compiler elides it and may stop the forward after the deepest one |
| visibility | the effect is visible **only** to nodes naming this node in their own `conditions`. There is no ambient "all effects on this input" |

- **Pre-transform is load-bearing.** Post-transform would make "read before the write at
  the same address" inexpressible, since both would be the same `input`, the same
  `conditions` and the same address.
- This deletes the collect-after-edit declaration-order rule: *read the un-edited value* =
  the interventional itself; *read the edited value* = any interventional that lists it in
  `conditions`.

### 3.7 `conditions` — the crux

- `conditions` = **which interventionals are in force when I run.**
  `do`'s operands = **which values I use.** They are independent.
- **Consuming a value does not put you under the conditions it was harvested under.**
  Path patching is exactly where the two differ, and it is why "an ordered list of edits"
  cannot express it.
- An effect is visible **only** to interventionals that list it in `conditions`. There is
  no ambient "everything declared on this input".
- Default is `()` — un-intervened.
- `(input, conditions)` is what the compiler groups by: two interventionals sharing both
  can share a forward pass; differing in either cannot. That grouping is the *only* place
  the forward count comes from, and it is derived, never declared.

### 3.8 The `do:` algebra — `Mechanism`

Closed set. `Operand = ValueName | ParamName | literal` — never a tensor, never a closure.

| Member | Effect | Causal-abstraction kind |
|---|---|---|
| `Identity` | none | the empty interventional |
| `Swap(op)` | `f ← op` | hard intervention (Def. 9); `IntInv` (Def. 49) when `op` is a source read; `DistIntInv` (Def. 51) when featurized |
| `AddScaled(op, α)` | `f ← f + α·op` | interventional (Def. 11) — reads the old mechanism |
| `Lerp(op, α)` | `f ← (1−α)·f + α·op` | interventional |
| `Affine(A, b)` | `f ← Af + b` | interventional |
| `Gaussian(NoiseSpec)` | `f ← f + scale·randn(seed)` | interventional; the seed is formally an extra **input variable** |
| `Renormalize` | `f ← f·‖base‖/‖f‖` | interventional |
| `Clamp(lo, hi)` | `f ← clip(f, lo, hi)` | interventional |
| `PyTorchFn(qualname, local_only=True)` | arbitrary | escape hatch; **refused at construction** by any non-local backend |

- `NoiseSpec(seed, scale, axis_semantics)` — declared, so a TP backend can distinguish
  TP-duplicated from TP-split randomness. `seeds` is inside the hash.
- No `Blend`: DBM is `Swap` through a `gate` featurizer (§3.5).

### 3.9 Presets

Thin aliases over `do`. Each returns exactly one `Interventional`.

| Preset | Expands to |
|---|---|
| `collect(site, position, input)` | `do=Identity` — the only one that names something the algebra doesn't |
| `replace(site, position, input, source)` | `do=Swap(source)`; `source` is any `Operand` |
| `steer(site, position, input, source, α)` | `do=AddScaled(source, α)` |
| `interpolate(site, position, input, source, α)` | `do=Lerp(source, α)` |
| `noise(site, position, input, seed, scale)` | `do=Gaussian(...)` |

Named *methods* are compositions, and ship as protocol YAML under `causalab/methods/`
(§9), not as primitives here:

| Method | Composition |
|---|---|
| interchange intervention (activation patch) | `collect` on `source[j]` → `replace` on `base` with that value |
| DBM | `replace` through a `gate` featurizer |
| path patching | `collect` ×k → `replace` ×k → `collect` under those conditions → `replace` (§6) |
| logit lens / patchscope | `replace` into `ln_final`'s input → `collect` at `lm_head` |

`neural/` provides the primitives; `methods/` names the experiments as YAML. The preset table
being one-to-one with `Mechanism` is the sign the algebra is the real primitive set.

### 3.10 Metrics — off-model

Pure functions of node values + dataset columns. **No `input:`, no `conditions:`** — they
inherit both from the interventional they read.

| Metric | Fields | Causal-abstraction kind |
|---|---|---|
| `logit_diff` | `of · a · b` | `Sim` |
| `cross_entropy` | `of · target` | `Sim`; IIA's loss twin |
| `class_probs` | `of · groups` | `Sim` |
| `kl` | `of · target` | `Sim` |
| `top_k` | `of · k` | reduction |
| `match` | `of · expected` | ⇒ **IIA** (Def. 53) |

- Extraction is **not** a metric — `LogitsAt` is a `collect` at `lm_head`, and
  `TokenLogits(ids)` is `dims: ids` at `lm_head` (the vocabulary *is* that site's feature
  axis). The backend derives `logits_to_keep` and vocab-parallel readout from that.
- `outputs: tuple[InterventionalName | MetricName, …]` names what comes back. Also drives early stop.
- `Objective = Sum[(weight, MetricName | Regularizer)]`; `Regularizer ∈ {L1(param), L2(param)}`.

## 4. Well-formedness — all checked before the model is touched

1. Every name resolves; every `source[j]` has `j < len(sources)`.
2. Acyclic: `n ∉ n.conditions` after closure; every operand of `n.do` is declared.
3. **Two interventionals sharing an `input` and the same closed `conditions` may not write
   overlapping addresses unless they commute** — the intervention-algebra condition
   (commutativity + left-annihilativity). Replaces the declaration-order tie-break.
   *Not checked today.*
4. `dims` selections co-occurring under the same `(input, conditions)` must be **disjoint**.
5. An interventional with `do ≠ Identity` that nothing lists in `conditions` is dead — warn.
6. A `PyTorchFn` under a non-local backend is refused at construction.
7. Sugar (`k=1` source, `conditions: all`, span ranges) **expands at load**; the canonical
   hashed form is always explicit, with `conditions` transitively closed and sorted.

## 5. Derived — never written by hand

| Property | Derivation |
|---|---|
| `requires` | `grad` ⇐ `params ≠ ∅ ∧ objective`; `paired_forward` ⇐ an effect depends on a value read on another input; also `full_logits`, `editable_attention_probs` |
| `num_forwards` | a property of a **compile**, ≥ the number of distinct `(input, conditions)` pairs; fusion is the backend's call |
| `digest` | `sha256(canonical_json(protocol))`, each `Param` replaced by its `sha256` |
| `denotes` | `control` if no effect is an interchange, else `explanation` |

- `choose_backend(protocol)` = `first b where protocol.requires ⊆ b.capabilities`.
  Refusal text generates from the missing capability.

## 6. Example — path patching

Sender → receiver, everything off-path frozen. Default `conditions` is `[]`, so only
the interesting line carries one.

```yaml
version: "1.0"
model: {key: meta-llama/Llama-3.1-8B, revision: main}

counterfactual_dataset:
  base:   {dataset: "ioi/test@3f1c", field: input}
  source: [{dataset: "ioi/test@3f1c", field: "counterfactual_inputs[0]"}]

positions: {last: {type: index, position: -1}}

sites:
  sender:     {component: attention_value, layer: 9, head: 9}
  receiver:   {component: block_input, layer: 12}
  freeze_a10: {component: attention_output, layer: 10}
  freeze_a11: {component: attention_output, layer: 11}

interventionals:
  # harvest, un-intervened
  v_sender:    {site: sender,     position: last, input: source[0]}
  v_a10:       {site: freeze_a10, position: last, input: base}
  v_a11:       {site: freeze_a11, position: last, input: base}

  # the patched condition set
  swap_sender: {site: sender,     position: last, input: base, do: {swap: v_sender}}
  freeze_10:   {site: freeze_a10, position: last, input: base, do: {swap: v_a10}}
  freeze_11:   {site: freeze_a11, position: last, input: base, do: {swap: v_a11}}

  # read the receiver UNDER those conditions — the collect-under-intervention
  v_receiver:  {site: receiver, position: last, input: base,
                conditions: [swap_sender, freeze_10, freeze_11]}

  # inject it back into an otherwise clean base
  inject:      {site: receiver, position: last, input: base, do: {swap: v_receiver}}
  logits:      {site: lm_head,  position: last, input: base, conditions: [inject]}

metrics:
  logit_diff: {kind: logit_diff, of: logits, a: answer, b: cf_answer}

outputs:  [logit_diff]
requires: [paired_forward]               # derived
```

- Four distinct `(input, conditions)` pairs: `(source[0], {})`, `(base, {})`,
  `(base, {swap_sender, freeze_10, freeze_11})`, `(base, {inject})`.
- `num_forwards` is absent by design. nnsight fuses and stages; Megatron makes microbatch rows.
- `sender` / `receiver` / `freeze` are the causal-abstraction paper's own three words for
  this shape (recursive interchange intervention, Def. 50).

## 7. Example — DAS, and DBM as a delta

```yaml
version: "1.0"
model:      {key: meta-llama/Llama-3.1-8B, revision: main}
high_level: {key: "weekdays.causal_model"}          # what IIA compares against

counterfactual_dataset:
  base:   {dataset: "weekdays/train@9ab2", field: input}
  source: {dataset: "weekdays/train@9ab2", field: "counterfactual_inputs[0]"}   # k=1 sugar

positions: {last: {type: index, position: -1}}
sites:     {L18: {component: block_output, layer: 18}}

featurizers:
  rot18: {kind: subspace, shape: [4096, 8], parametrization: cayley,
          constraint: orthogonal, dtype: fp32, init: pca_random, params: [rot18.weight]}

params:
  - {name: rot18.weight, shape: [4096, 4096], constraint: orthogonal,
     dtype: fp32, init: pca_random, sha256: null}

interventionals:
  v_src:  {site: L18, position: last, input: source, featurizer: rot18, dims: [0,1,2,3,4,5,6,7]}
  patch:  {site: L18, position: last, input: base,   featurizer: rot18, dims: [0,1,2,3,4,5,6,7],
           do: {swap: v_src}}
  logits: {site: lm_head, position: last, input: base, conditions: [patch]}

metrics:
  iia: {kind: logit_diff,    of: logits, a: cf_answer, b: base_answer}
  ce:  {kind: cross_entropy, of: logits, target: label}

outputs:   [iia, ce]
objective: {terms: [[1.0, ce]]}
seeds:     {init: 0}
requires:  [paired_forward, grad]           # the one line that routes this to MegaFire

train:
  params:     [rot18.weight]
  optimizer:  {name: adamw, weight_decay: 0.0, betas: [0.9, 0.999]}
  lr:         1.0e-3
  schedule:   {kind: linear, warmup: 0}
  steps:      {epochs: 10}
  batch:      {pairs: 16, accumulate: 1}      # PAIRS, not rows
  precision:  {feature: fp32, loss: fp32, model: bf16}
  eval:       {every: {epochs: 1}, split: "weekdays/test@9ab2", metrics: [iia]}
  early_stop: {metric: iia, patience: 3, mode: max}
  checkpoint: {every: {epochs: 1}, scope: params}     # the rotation — KB, not 16 GB
  seed:       0
  resume:     null
```

**DBM = three edits:**

```yaml
featurizers: {gate18: {kind: gate, shape: [4096], dtype: fp32, params: [gate18.theta]}}
interventionals: {patch: {..., featurizer: "rot18 >> gate18", do: {swap: v_src}}}
objective:   {terms: [[1.0, ce], [0.01, {l1: gate18.theta}]]}
train:       {anneal: {gate18.theta.temperature: [1.0, 0.01, 0.5]}}
```

## 8. Serialization

```
protocol.yaml            # the value. Tensors appear only as Param names.
weights.safetensors      # every Param, keyed by Param.name
weights.meta.json
```

- JSON is the commit point, safetensors carries the payload, **no pickle** — the pattern
  `save_site_specs` already uses.
- `digest = sha256(canonical_json(protocol))`, keys sorted, floats canonicalized, each
  `Param` replaced by its content hash. Stable under reformatting, sensitive to weights.
- `ArtifactIdentity` on every fitted featurizer: `produced_by: <digest>` + model key,
  revision, tokenizer, site record, hook semantics, width, k, parametrization, dtype,
  trained-on, backend, commit. `load` refuses a mismatch. **This is the MegaFire↔causalab
  contract.**

## 9. Surface

**YAML is the only authoring surface.** There is no Python builder and no
`methods.*(…) -> Protocol` constructor. A protocol is written, composed by hydra, and run.

```sh
causalab run protocol.yaml                              # one protocol, one run
causalab run protocol.yaml model=llama31_8b sites.L18.layer=24
causalab validate protocol.yaml                         # rules §4, no model touched
causalab digest protocol.yaml                           # the provenance key
```

Hydra composes **protocols**, not `DictConfig`s that a runner interprets — config groups
merge into one protocol document, and the merged document is the hashed value:

```
configs/
├── protocol/das_subspace.yaml     # interventionals · metrics · objective · train
├── model/llama31_8b.yaml          # -> protocol.model
├── task/weekdays.yaml             # -> protocol.counterfactual_dataset (+ high_level)
└── config.yaml                    # defaults: [protocol: das_subspace, model: …, task: …]
```

The Python surface is **runtime only** — load, check, hash, run. Not authoring:

| Call | Returns |
|---|---|
| `Protocol.from_yaml(text \| path)` | the frozen value; expands sugar, applies §4 |
| `protocol.to_yaml()` | canonical form — closed & sorted `conditions`, explicit `k`, expanded spans |
| `protocol.digest` | `sha256` of the canonical form |
| `protocol.requires` | the derived capability set |
| `run(protocol, data, backend="auto", **backend_kwargs)` | `Result` (+ artifacts, + `ArtifactIdentity`) |

Two consequences worth stating rather than discovering:

- **`causalab/methods/` ships protocol YAML, not Python builders.** The compositions in
  §3.9 become templates under `configs/protocol/`, plus their metric definitions. A layer
  scan is one template with `sweep: [{for: sites.L18.layer, in: [0, …, 31]}]`, not a loop.
- **Variable-arity shapes need `ForEach` or generated YAML.** A path patch with a
  *k*-node freeze set has *k* interventionals; `sweep` covers the regular cases (scans,
  grids), but an irregular freeze set is written out or generated. Not a blocker — worth
  knowing before someone reaches for a builder to solve it.

## 10. Migration — what each object at `a50637c` becomes

| Today | Becomes |
|---|---|
| `Site`, `HeadSite`, `AttentionProbabilitiesSite` | `Site` (pure data) + a per-backend `SiteResolver` |
| `Featurizer` (`nn.Module` pair) | `Featurizer` (declared) + `Param`s (bytes) |
| `FeaturizedSite` | the `site` + `featurizer` + `dims` fields of a node; error-term contract unchanged |
| `Edit`, `ReadSource`, `CollectOp`, `EditOp` | one `Interventional` |
| `modes.py` (7 constructors) | 5 presets over `Mechanism`; `interchange` and `mask` move up to `methods/` as YAML compositions (§3.9, §9) |
| `methods/*` Python builders (`run_layer_scan`, `build_edge_plan`, …) | protocol templates under `configs/protocol/`; the runner is generic |
| `CollectOp.step` / `EditOp.step` / `GenerateSpec` decode-step ops | **dropped.** No code in `causalab/` ever sets a nonzero `step` — only `tests/neural/{test_plan,test_staged,test_preflight,test_dataset}.py`. `GenerateSpec(max_new_tokens=…)` itself stays (used at `dataset.py:493`); only step-addressed ops go. What is lost: an edit that fires on decode pass *k* rather than the prefill — a capability Megatron lacks anyway |
| `SiteSpec` / `EditSpec` | absorbed into `Protocol` — they are already the closure-free layer |
| `Plan` / `staged.py` / `run_plan` | the nnsight backend's private `NNSightSchedule` + executor, below the seam |
| `save_logits` | a `collect` at `lm_head` |
| `GradientRequest` | `Objective` + `Param` + `TrainSpec` |
| `trainable.py`, `traced_label_loss` | deleted |
| `metric.py` scorers | rebuilt on `Metric`, so loss and metric share one vocabulary |
| `logit_lens` (outside the graph) | an ordinary write + read |
| `Plan.inputs` (tensors) | `CounterfactualDataset` (schema) + data bound at run time |
| `Plan.models` (identity) | `ModelConfig` |

## 11. Causal-abstraction glossary

Terms below are from Geiger et al., *Causal Abstraction: A Theoretical Foundation for
Mechanistic Interpretability* ([arXiv:2301.04709](https://arxiv.org/abs/2301.04709)).

| Ours | Theirs | Where |
|---|---|---|
| `ModelConfig` / `CausalModelConfig` | low-level model ℒ / high-level model ℋ | §2.3 |
| `Alignment` | ⟨Π, π⟩ | Def. 34 |
| `CounterfactualDataset.base` / `.sources` | base **b** / sources ⟨s₁…sₖ⟩ | Def. 49 |
| a `collect` + `replace` chain | interchange intervention, `IntInv`; `DistIntInv` when featurized | Def. 49, 51 |
| `Site` + `Pos` + `dims` | a target variable set **X** ⊆ **V** | Def. 49, 51 |
| `Featurizer` | bijective translation τ | Def. 30; §3.6 says PCA/SAE/probe/DAS *are* τ |
| `Interventional` (node) | `Proj_X(Solve(ℒ_𝕀))` + the assignment of 𝕀 to **X** | Def. 4, 8, 11 |
| `conditions` | the subscript 𝕀 in ℒ_{b∪𝕀} | §2.1 |
| `Mechanism` (the `do:` algebra) | an **interventional**, `𝕀_X : Soft_X → Soft_X` | Def. 11 |
| well-formedness rule 3 | commutativity + left-annihilativity | Def. 16–17 |
| `Metric` | `Sim`; `match` ⇒ IIA | Def. 45, 53 |
| `Objective` | 𝕊 over `Sim` — graded faithfulness | Def. 45, §3.2 |
| `Protocol` | ⟨ℒ, ℋ, ⟨Π,π⟩, `Domain(ω)`, `Sim`, ℙ, 𝕊⟩ | Def. 27, 45 |

- No counterpart, correctly: `Site.component` vocabulary, `Pos`, `Backend`,
  `digest`, `ForEach`. Causal abstraction is a semantics; it says nothing about addressing
  or execution.
- §2.2 argues for the closed algebra independently: *"The unconstrained space of
  interventionals `Func_V` is unruly … We want to characterize spaces of interventionals
  that 'act like hard interventions'."*
- §3.6.2 puts DBM in *aligning* features (learning Π) and DAS in *learning* the feature
  space (learning τ) — which is why the DBM gate is a featurizer, not a transform.

## 12. Build order

Each step is useful alone; none blocks on a second backend existing.

| # | Step | Cost |
|---|---|---|
| 1 | `to_yaml` / `from_yaml` / `digest` + the round-trip test | 1 day. `Plan` fails it by construction; that failure *is* the argument |
| 2 | `ArtifactIdentity` on featurizer bundles | 1 day. Makes a MegaFire-trained subspace safe to load |
| 3 | Closed `Mechanism` algebra; `mask` + `collect` declarative | required by NDIF independently of Megatron |
| 4 | `Metric` / `Objective` | deletes `metric.py` host slicing; derives `logits_to_keep` from the IR |
| 5 | `Param` + `Featurizer` split + `TrainSpec` | unblocks featurizer training on any backend; deletes `trainable.py` |
| 6 | `CounterfactualDataset` + symbolic positions + `conditions` | the step that demotes `Plan` |
| 7 | `Backend.capabilities` + `choose_backend` | only once a second backend exists |

**Settle it empirically first, no infra required:** run the existing plan tests under
`remote='local'` and see whether `barrier` / `stop` / `iter` survive serialization;
`nnsight.register(causalab)` and measure the payload; write one
`Protocol → YAML → hash → back` test.

## 13. Known costs

- **Structural conditioning is not nominal.** Omitting a name in `conditions` silently
  creates a second `(input, conditions)` group and a second forward — the failure is a
  wrong number, not an error. Mitigate with all three: intern condition sets at load;
  print the derived groups and forward count in the compile report; allow a *named*
  condition set as sugar that expands before hashing.
- **Datasets by reference** make the protocol only as reproducible as the ref — a digest in
  the ref is the minimum.
- **The closed algebra will hurt** the first time someone wants a weird `g`. `PyTorchFn`
  is the pressure valve and must refuse loudly at construction.
- **Metric vocabulary can grow without bound.** Hold the line: a metric is a
  gather-then-reduce over node values and dataset columns, nothing else.
- **The counter-position, fairly stated:** if Megatron came off the roadmap, "no IR — just
  functions that build nnsight traces" would be the better and cheaper design. The IR earns
  its keep through exactly two requirements — **retargetability** and **provenance**.

## 14. Open

- `Mechanism` as the name of the `do:` algebra — the node took `Interventional`, so the
  algebra needed a different word. Alternatives: `Do`, `Transform`.
- Whether `Featurizer` should carry the `Config` suffix (§3.2 says no: it is defined
  inline, not referenced).
- `dims: {soft: …}` was folded into the `gate` featurizer. Same math at the hard threshold,
  but it moves where the gradient attaches — needs a parity test against
  `tests/neural/test_modes.py`'s soft and hard mask cases before it is settled.
- `{type: span, …}` (§3.4.1) is not implemented. Small, but unwritten.
- `step` was dropped (§10). If decode-step editing is ever wanted back, it returns as a
  field on `Interventional` plus a `decode_step_edits` capability, not as a new type.
