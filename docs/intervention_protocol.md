# Intervention Protocol — specification v1

The **Intervention Protocol** defines causal intervention experiments on neural networks, with some useful properties:

- **Serializable JSON document that's easily shareable and reproducible**: The intervention protocol is one JSON document that fully describes an experiment: it can be hashed, diffed, shared, and re-run. It is self-contained and enables exact reproduction.
- **Agnostic to neural network interfaces.**: The intervention protocol defines the experiment, not the implementation. The protocol is passed to a separate parser/compiler compiling the protocol into runnable code of intervention engines (like pytorch-native hooks, NNsight/NNterp, SGLang, Megatron, etc.). The protocol says *what*; engine specific parser/planner derives *how* (forward count, fusion, batching, sweep parallelization) and execute an intervention protocol.

## 1. Document layout

Sections in this order (order enforced; `save` last). A document may also be
written in two halves — `application` then `method` — which carry these same
sections between them (§1.1):

| # | key | required | content |
|---|---|---|---|
| 1 | `version` | ✓ | `"1"` |
| 2 | `type` | –* | `protocol` \| `method` \| `workflow` — what this file is (*required in a method file, §1.1) |
| 3 | `description` | – | free text, the file's intent |
| 4 | `model` | ✓ | the neural network ℒ, and how it is realized numerically (alias: `neural_model`) |
| 5 | `data` | ✓ | input rows: `base` (+ `counterfactual`) |
| 6 | `positions` | – | named token-position specs |
| 7 | `sites` | ✓ | named activation addresses — the complete tap inventory |
| 8 | `featurizers` | – | named feature-space maps |
| 9 | `params` | – | free/constant tensors owned by no featurizer |
| 10 | `reads` | ✓ | value producers |
| 11 | `writes` | – | effect definitions (inert until listed) |
| 12 | `intervened_models` | –* | which writes are in force on which input (*required if `writes` present) |
| 13 | `metrics` | – | closed reductions over read values |
| 14 | `train` | – | the fit, declared |
| 15 | `save` | ✓ | the complete output manifest — non-empty, last |

- **One global namespace**: every name in sections 6–13 must be unique across
  all of them; reserved names: `base`, `counterfactual`, `counterfactual[j]`, `original`.
- All cross-references must resolve; references are by name, never inline
  duplication.
- **Artifact-valued fields**: anywhere a scalar or position is expected,
  `{"artifact": "<path>", "key": "<field>"}` reads one value from a prior
  run's artifact at load. Missing artifact = load error.

### 1.1 The method / application split

A document answers two questions at once: *what is the experiment* — the
hypothesis, what is read, what is written into whom, how it is scored — and
*what was it run on* — which network, over which rows, at which addresses, in
which precision. The first half transfers to another model and another task;
the second half is exactly the part that does not. A document may therefore
say both, in two labelled halves:

```json
{
  "version": "1",
  "description": "the run's intent",
  "application": {
    "model": {"key": "meta-llama/Llama-3.1-8B", "revision": "main", "dtype": "bf16"},
    "data":  {"base": {...}, "counterfactual": {...}},
    "sites": {"target": {"layer": 18}}
  },
  "method": { "...": "reads, writes, intervened_models, metrics, save" }
}
```

**One file is one experiment run.** The split is a shape *inside* the
document, not a second file to keep in step with the first: a run document is
self-contained, hashable and shareable exactly as a flat one is. The `method`
half may instead be a **path** to a reusable method file (relative to the
document, `"type": "method"` at its top level) — the loader inlines it, and
the record of what ran carries the whole thing either way. A method file may
also simply be pasted into the half; the `type` and `version` it brings along
are checked, and a method digests the same inline as in its file.

| | the **method** half | the **application** half |
|---|---|---|
| holds | the experiment: `causal_model`, `positions`, `featurizers`, `reads`, `writes`, `intervened_models`, `metrics`, `train`, `save`, and the site *names* it addresses | the inputs and the addresses: `model` (key, revision, dtype, quantization), `data`, and the site records |
| must hold | `reads` and `save` — what is measured is the method's, never the application's | `model` and `data` — the network and the rows a method leaves open |
| may not hold | `model`, `data` | — |

A method's `sites` entries may be partial or empty — `"target": {}` names an
address the application supplies, `{"component": "block_output"}` fixes the
component and leaves the layer open, `{"component": "lm_head"}` is already
closed. Everything still open is the method's **signature**, and `explain`
prints it.

Composition is a **disjoint-or-equal merge**: every leaf comes from exactly
one half, or from both with the same value (a restatement, cross-checked like
a `save` entry's bindings, §2.12). An application may *complete* a method,
never overrule it — a contradiction is a load error (rule 18), because an
application that could overrule its method would make the method's digest a
claim about nothing. The two `description`s join, method first.

The composition is an ordinary protocol document: it validates, expands,
canonicalizes and digests **exactly as the same experiment written flat**.
Splitting is an authoring choice, not a second dialect, and the point digest —
the provenance unit — is unmoved by it (§7). Dotted paths (`--set`, a workflow
step's `set`) address the *composition*, so they mean the same thing in both
forms. Which method a run used is reported by `explain`, written into the run
record, and stamped into artifacts; it is not part of the canonical bytes.

## 2. Section reference

### 2.1 `model`

| field | meaning |
|---|---|
| `model.key` | model name (HF key or registry name) — the network as a *name* |
| `model.revision` | checkpoint revision |
| `model.dtype` | the compute dtype the weights are realized in: `fp32` (default) \| `bf16` \| `fp16` |
| `model.quantization` | optional — load-time weight quantization (below) |

- `neural_model` is accepted as an alias of `model`; canonical form uses `model`.
- **Precision is part of the experiment, not of the run.** The same protocol at
  `bf16` and at `nf4` produces different numbers, so `dtype` and `quantization`
  are document vocabulary and enter the digest. An authored file may stay
  silent — the canonical form materializes `dtype` (§7), so no *record* is ever
  silent about the precision its numbers came out of. The CLI's `--dtype` is
  shorthand for `--set model.dtype=…` (§9): it changes the document, and the
  digest changes with it.

```json
"model": {
  "key": "meta-llama/Llama-3.1-8B", "revision": "main", "dtype": "bf16",
  "quantization": {"scheme": "nf4", "method": "bitsandbytes", "compute_dtype": "bf16", "double_quant": true}
}
```

| quantization field | meaning |
|---|---|
| `scheme` | ✓ — `int8` (LLM.int8() mixed-precision decomposition) \| `nf4` \| `fp4` (the two 4-bit datatypes) |
| `method` | the quantizer: `bitsandbytes` (default, the only v1 entry) |
| `compute_dtype` | dtype the dequantized matmuls run in; defaults to `model.dtype` |
| `double_quant` | 4-bit only — quantize the quantization constants |
| `int8_threshold` | `int8` only — the outlier threshold (default `6.0`) |

- There is no bare `int4`: 4-bit is `nf4` or `fp4`, and a field whose point is
  to name one realization may not be ambiguous about which.
- Weights quantized **ahead of time** (GPTQ, AWQ) are a property of the
  checkpoint, so `model.key`/`revision` already name them; `quantization`
  describes quantization applied *at load* to an unquantized checkpoint.
- A document with `quantization` requires the `quantized_weights` capability
  (§8), so an engine that cannot realize it refuses instead of quietly running
  something else.

### 2.2 `data`

```json
"data": {
  "base":   {"dataset": "weekdays/train", "field": "input"},
  "counterfactual": {"dataset": "weekdays/train", "field": "counterfactual_inputs[0]"}
}
```

- `dataset`: a **ref — a local path under the data root, no digest**. The parser
  resolves it at load and stamps the content digest into the canonical form
  (sec. 7). A resolver provides three things for a ref: its content digest, its
  columns, and its rows.
- Tables are **serialized ahead of the load, never generated during it**: a ref
  resolves by reading bytes, so no digest depends on importing task code, a
  tokenizer, or the network. Task-generated tables are built by
  `causalab.tasks.serialize` (`scripts/build_task_dataset.py`), which also
  writes a `<ref>.manifest.json` provenance sidecar that resolution ignores; a
  Hub-hosted dataset enters the same way, by being materialized first.
- Roles are the keys: `base` (required) and `counterfactual` (optional). `counterfactual` is
  singular; if its value is an array, references index it as `counterfactual[j]`.
  Rows are paired: one base row + its counterfactual row(s) form one example.
- `field` selects the column; `[j]` indexes list-valued columns.
- **Anything per-row or task-semantic is a column**, computed when the table is
  built — answer forms for `match` (sec. 2.10), values that place a position per
  row (sec. 2.3). Documents reference columns; they never compute.
- Dataset **columns** referenced by metrics and by `column` positions, and the
  prompt **variables** named by `variable` positions and by `scope` /
  `relative_to` anchors, are checked against the resolved tables by
  `validate --data`, not at load. The check covers **every expanded point**, so
  a reference that appears only at one coordinate of a sweep is checked there
  too.

### 2.3 `positions`

Named entries; a read/write `pos` is a name here, or an inline spec.

| form | resolves to |
|---|---|
| `-1` (bare int, sugar) | `{"index": -1}` |
| `"all"` (bare string, sugar) | `{"all": true}` |
| `{"index": n}` | one token per row. `n < 0` counts from the end of the sequence; `n ≥ 0` is rebased past any chat prefix |
| `{"variable": "x"}` | all tokens of prompt variable `x` — a per-row window, ragged across rows |
| `{"column": "c"}` | all tokens of the string in column `c` of the row — the per-row form |
| `{"span": [a, b]}` | fixed window `[a, b)` |
| `{"all": true}` | every content token of the row — ragged across rows |
| + `"scope": {"variable": "x"}` / `{"column": "c"}` | interpret the index/span inside the anchor's span |
| + `"relative_to": {"variable": "x"}` / `{"column": "c"}` | offset from the anchor's span |
| + `"generated": {"max_new_tokens": n}` | resolve the anchor inside the row's greedy continuation instead of its prompt |

- Positions are **never resolved to integers in the document**. Resolution is
  an engine service against a `PositionFrame` (pad side, packing, sequence
  shard map) — sec. 8.

**The continuation frame (`generated`).** A decode produces a second frame, so
addressing it needs no new vocabulary: `generated` is a **frame selector**, not
an anchor, and exactly one anchor accompanies it. `{"generated": {…}, "index":
-1}` is the last generated token, `{…, "all": true}` every generated token,
`{…, "span": [0, 3]}` the first three, `{…, "variable": "x"}` the tokens where
the model said the row's value for `x`.

- The decode is **greedy** — argmax at every step. Sampling is not expressible:
  a document is a value (sec. 7), and a sampled continuation is not a function
  of it.
- `max_new_tokens` is required, ≥ 1, and sweepable. It is a mapping rather than a
  bare int so stopping conditions can join it later. Two positions on one
  (model, input) with different budgets are legal: the run decodes the deepest,
  and each read windows its own.
- The frame **ends at the row's first EOS**, so widths differ and continuation
  reads are ragged. A window reaching past a row's end clips; a row that
  generated nothing contributes no positions. Unlike the prompt frame — where an
  out-of-range index is an authoring error and refused — how far a row generates
  is a *result*, and refusing on it would make a document fail on data.
- **`variable` in the continuation** — "the tokens where the model said the
  row's value for `x`" — differs from its prompt-side twin in two ways, both
  because the continuation is a *result* rather than an input. It takes the
  **first** occurrence instead of demanding exactly one, since a generation may
  repeat itself as a matter of course; and **zero** occurrences yield zero
  positions rather than refusing, because whether the model says the thing is
  usually the experiment. A metric over such a read reports the miss as a null
  value with `matched: false` (sec. 2.10), so it is data, not an exception.
  Character spans come from the decode's own incremental detokenization, so a
  match that starts inside a merged piece still lands on every token that
  produced it.
- **Reads only.** A write may not carry `generated` (rule 16): the continuation
  exists because the prefill already ran, and an intervention reaches it through
  the first token's logits and through what the prefill left in the KV cache —
  nothing re-fires per decode step. `train` and `generated` do not combine
  either: a greedy decode is an argmax chain with no gradient path.
- **`lm_head` at generated position `j` is the distribution *after* token `j`.**
  The one that *produced* token `j` sits at `j − 1`, and for `j = 0` that is the
  last prompt position — an ordinary `{"index": -1}`. Stated here so no document
  has to rediscover it.
- `column`, `scope` and `relative_to` are refused with `generated`: they resolve
  against the prompt, which the continuation does not contain.
- `variable` vs `column`. A **prompt variable** is looked up per role: the
  `<field>_variables` sibling of the role's text column first, then a
  same-named column. A **column** is looked up only as a top-level column of
  the row, so it is a property of the *row*, not of a role's text — the same
  `{"column": "c"}` resolves to the same string whichever role reads it. Use
  `column` when the value is computed by the task (per-row answer symbols,
  chosen entities) and every role must read the same string; use `variable`
  when each role's own text carries its own value, which is what a
  counterfactual pair usually needs.
- **What `validate --data` can and cannot say about a position.** Both
  spellings are checked for **existence** — a `column` against the resolved
  tables' columns, a `variable` against each role's `<field>_variables` sibling
  and the same-named-column fallback. Neither is checked for **width**: what a
  variable or a column resolves to is a char span, hence a token count, and the
  pure verbs hold no tokenizer (`ResolutionEnv` carries datasets and artifacts
  and stays torch- and network-free). So a per-row window that turns out ragged
  across rows is a *run-time* refusal — rule 19 for a write — and no amount of
  pre-flighting moves it earlier. When a document needs one token per row at a
  variable, say so: `{"index": -1, "scope": {"variable": "x"}}` is the last
  token of `x`'s span and is never ragged.
- The value in a column position is a **string**, resolved like a variable's
  value (it must occur exactly once in the row's text). Integer token indices
  are deliberately not a v1 spelling: they would bind a table to one
  tokenizer, and a task that can compute an index can serialize the substring
  instead.
- ⚠️ **There is no chat prefix in v1.** Both the `n ≥ 0` rebase above and the
  "past any chat prefix" clause below are written against a `prefix_lengths`
  the encoder sets to **0 for every row** — `apply_chat_template` is called
  **nowhere** in the package, so the rule is an identity today. (The
  `use_chat_template` field and chat-prefix hook in
  `neural/token_positions.py` belong to the pre-protocol pipeline surface the
  task packages annotate against; no engine implements them. That module says
  so at the top.) A dataset that wants an instruct
  model's chat frame bakes the *rendered* template into its `input` column,
  which works; what it must not do is leave the template's leading BOS in
  place, because the encoder adds special tokens as the tokenizer defines them
  and a second BOS shifts every position by one. That case is refused at
  encode. A first-class `chat` field is a spec change and its own PR.
- **`{"all": true}`** selects the row's real tokens only: padding is excluded,
  and so is any chat prefix — the same frame `{"index": n}` uses for `n ≥ 0`.
  It takes no `scope` or `relative_to` (there is nothing left to narrow), and
  `all` is a reserved name, so `"pos": "all"` is always the sugar and never a
  lookup in this table. Rows of unequal length make it ragged: reads carry
  that natively, a write at `all` needs every row to be the same length in
  v1.

### 2.4 `sites`

```json
"target": {"component": "block_output", "layer": 18}
```

| field | meaning |
|---|---|
| `component` | one of the vocabulary below |
| `layer` | depth index (where the component has one) |
| `head` / `expert` / `stream` | optional sub-axes: attention head, MoE expert, **mixer stream** |

`head` and `stream` are checked against the component, not against the model:
`head` is refused on a component with no head axis, and bounded by *that
component's* head count — which under GQA is narrower for a KV-space component
than for a query-space one. `stream` is one of `full_attention` /
`linear_attention`.

Component vocabulary (per-engine `SiteResolver` maps each to a tap), in
execution order:

`input_ids` · `embeddings` · `block_input` · `attention_input_norm` ·
`delta_qkv` · `delta_gate` · `delta_conv` · `delta_query` · `delta_key` ·
`delta_value` · `delta_beta` · `delta_decay` · `delta_kv_mem` ·
`delta_state_update` · `delta_state` · `delta_kernel_output` · `delta_premix` ·
`attention_query_pre_rope` · `attention_key_pre_rope` ·
`attention_value_states` · `attention_gate` · `attention_query` ·
`attention_key` · `attention_scores` · `attention_z` · `deltanet_qkv` ·
`deltanet_gate` · `deltanet_qkv_conv` · `deltanet_query` · `deltanet_key` ·
`deltanet_value` · `deltanet_beta` · `deltanet_decay` · `deltanet_state` ·
`deltanet_core_out` · `deltanet_gated_out` · `attention_result` ·
`attention_output` · `attention_premix` · `attention_probs` · `block_mid` ·
`mlp_input_norm` · `mlp_input` · `router_logits` · `router_scores` ·
`expert_idx` · `expert_gate_proj` · `expert_up_proj` · `expert_activation` ·
`expert_permutation` · `expert_output` · `routed_output` · `mlp_activation` ·
`shared_expert_gate_proj` · `shared_expert_up_proj` ·
`shared_expert_activation` · `shared_expert_output` · `shared_expert_gate` ·
`mlp_output` · `block_output` · `ln_final` · `lm_head`

- **`sites` is the complete inventory**: every site a read or write references
  must be declared here, including `lm_head` (`{"component": "lm_head"}`).
  There are no implicit site names.
- **The three norm taps** name the two RMSNorms every block carries:
  `attention_input_norm` is `input_layernorm`'s **output** (what the mixer
  consumes), `block_mid` is `post_attention_layernorm`'s **input** (the residual
  stream after the mixer is added), and `mlp_input_norm` is that same module's
  **output**. So a block satisfies
  `block_mid = block_input + attention_output` and
  `block_output = block_mid + mlp_output`.
- **`input_ids` is the model's token input, not an activation.** It is
  layer-less, **read-only**, and *not a feature space*: it carries integer ids
  on a position axis, so no featurizer may attach to it and it has no width.
  Read `embeddings` for the vector the ids look up.
- **The MoE surface** splits four ways. The **router** exposes
  `router_logits` (all experts), `router_scores` (the renormalized top-k) and
  `expert_idx` (which experts, integer ids on the same top-k axis);
  `router_probs` is *derived* — `softmax(router_logits)` — and is not a
  component. `routed_output` is the combined expert output, and the **shared
  expert** exposes its SwiGLU interior plus `shared_expert_gate`, the scalar
  that mixes it in. The *routed* per-expert interior (round 3) has no module
  boundaries at all — the experts module stores its weights as 3-D parameters
  and computes the whole interior inside one dispatched
  `ALL_EXPERTS_FUNCTIONS["grouped_mm"]` call — so its components are reached by
  wrapping that dispatch entry, and they carry a **dispatch pin**: a model
  loaded with any other `experts_implementation` (the `"eager"` per-expert
  loop, `"batched_mm"`) is refused by name, because a different factorization
  computes different intermediates even where the block's output agrees (to
  4.2e-7 on the fixture). `expert_activation` is the activated gate half,
  `act_fn(gate_e)` — the same tensor `mlp_activation` names on the llama
  family — represented **token-major**: `(batch·position, top_k · d_expert)`,
  slot *k* the *k*-th ranked expert, joined to experts through `expert_idx`.
  Its slot axis is a ranking, like `router_scores`, with the same
  basis-fitting refusal — and so are the other interior slots:
  `expert_gate_proj` and `expert_up_proj` are the two halves of the fused
  `[gate_e | up_e]` projection (one capture, two addresses — the
  `attention_gate` precedent), and `expert_output` is the down-projection's
  output **before** the routing weight, pinned by the identity
  `routed_output == Σ_slot expert_output · router_scores` (exactly, 0.0).
- **`expert: e` is the ragged face of the routed interior.** On the four
  interior components it selects the (position, slot) pairs the router sent to
  expert *e* and returns flat rows plus per-example widths (a ragged value);
  an expert no token chose returns width-0 rows — a data fact, not an error.
  A write under `expert: e` lands only on that expert's rows (and therefore
  lands nowhere when no addressed token chose it). `featurizer`/`dims` are
  refused on this face: they are sized against the token-major `top_k · d`
  axis and these rows are `d`-wide. On every other MoE component `expert`
  is still refused — those tensors have no per-expert axis.
  `expert_permutation` (integral, read-only) is the serving kernel's row
  bookkeeping for anyone aligning raw kernel-order tensors; it lives inside
  the fused forward where no module boundary exists, so only the nnsight
  engine's `.source` address table serves it. The other interior components
  are served by both engines — the reference engine through the dispatch
  wrapper above, the nnsight engine through its `.source` addresses — with
  the same token-major presentation and the same pre-routing-weight
  `expert_output`.
- **`expert_idx` is a routing table, not a feature space** — the same rule as
  `input_ids`: integer ids, no featurizer, no width. And `router_scores` has a
  width but its axis is a per-token **ranking**, not a basis: column *k* is the
  *k*-th ranked expert, a different expert for different tokens, so a basis
  fitted across positions is fitted across a basis that is itself shuffled per
  position. A **basis-fitting** featurizer there (`subspace`, `pca`, `sae`) is
  therefore refused; `identity`, `standardize` and `gate` act per column and are
  still accepted, because "how large is the top-ranked score, typically" is a
  meaningful question about a ranking.
- **`expert` is refused on every MoE component.** None of these tensors is
  indexed by expert id: the router's axes are all-experts or top-k, the shared
  expert is not one of the routed experts, and the per-expert interior is
  indexed by routed *slot* (its top-k axis) — `expert_idx` says which expert
  fills each slot.
- **The Gated DeltaNet interior** (`deltanet_*`) is the linear-attention
  mixer's inside — the fused q|k|v projection pre- and post-conv (the conv tap
  is channels-first), the q/k/v splits (q and k in *key-head* space, before
  `repeat_interleave` — the linear-attention analogue of GQA), the per-head
  write strength `deltanet_beta` and decay `deltanet_decay`, the output gate,
  the delta kernel's return pre- and post-gate, and **`deltanet_state`**: the
  recurrent state, once per 64-token prefill chunk. The state's position axis
  is the **kernel's chunk index**, not a token position — read it whole or at
  an integer chunk index; text anchors have nothing to resolve against there.
  Per-token prefill state does not exist: the recurrent kernel runs only in
  single-token decode, by the modeling code's own dispatch, so it is refused
  by name rather than served at a granularity the kernel does not have. In
  the **generated frame** the state *is* per token: each decode step runs the
  recurrent kernel once, and a continuation read of `deltanet_state` gets one
  state per generated position — a separate, decode-verified address, because
  decode dispatches different kernels than prefill (interior components
  without one refuse by name in that frame). These live inside one fused
  forward — the nnsight engine serves them through its `.source` address
  table; the reference engine refuses them by name.
- **`attention_result` is the per-head contribution to the residual stream**,
  and the only component the model never computes: the block projects the whole
  `attention_premix` at once, so what the forward pass forms is the *sum* over
  heads. `sum_h attention_result == attention_output` (minus the o-projection's
  bias, which belongs to no head) is the identity that defines it.
  Naming a `head` is strongly encouraged — the dense form is `heads` times
  `attention_output`, which on a 64-head model at hidden 4096 is 64× the memory
  — but the whole tensor is not refused, only documented. The read is derived
  after the position gather, so the cost is `n_positions · heads · hidden`
  rather than `seq · heads · hidden`.
- **Three components are read-only, and a write to any is refused** rather than
  silently discarded. `input_ids` is the model's input. `router_logits` is
  discarded by the MoE block itself, which routes on the scores and indices it
  computed from them — so a write there could not reach anything. Write
  `router_scores` to reweight the chosen experts, or `expert_idx` to change
  which experts fire. `attention_result` is *derived* — there is no tensor there
  to change — so write `attention_premix` with the same `head` instead; the
  result is a linear function of it.
- **The mixer's interior is four module boundaries, not four chunk ops.**
  `attention_query_pre_rope` and `attention_key_pre_rope` are the queries and
  keys as the mixer computes them, *before* RoPE rotates them — on a family with
  `q_norm`/`k_norm` those norms run before RoPE, so their outputs are exactly
  these tensors. `attention_value_states` is `v_proj`'s output: the actual value
  vectors, in **KV-head space**, and the tap sits before the KV cache is
  updated, so a write there reaches it. `attention_gate` is the second split of
  the gated-attention family's fused `[q | gate]` projection.
- **`attention_value_states` is not `attention_premix`,** and the two are the
  reason the latter was renamed. `attention_premix` is the o-projection's
  *input* — the mixer's output after the gate, in **query-head** space, `heads ·
  head_dim` wide. `attention_value_states` is `v_proj`'s output, in **KV-head**
  space, `kv_heads · head_dim` wide. Under GQA those differ by the group ratio,
  so a `head` valid on one can be out of range on the other; the bound is the
  component's, and naming a head the component does not have is an error.
- **`attention_gate` exists only where the mixer computes one.** Qwen3.5/3.6's
  attention multiplies its output by `sigmoid(gate)` before projecting out, and
  packs the gate into the q-projection. A family without one refuses the
  component by name rather than returning a slice of `q`. All four components
  are refused on a fused-qkv family (GPT-2's `c_attn`), and all four require a
  full-attention layer.
- **Four more components live *inside* the attention function**, where no
  forward hook reaches: `attention_query` and `attention_key` are the post-RoPE
  queries and keys as that function receives them (`attention_key` before the
  GQA `repeat_kv`, so **KV-head space**), `attention_scores` is the softmax's
  input, and `attention_z` is the function's result — the mixer's output
  *before* the gate multiply and the o-projection. Unlike the module-boundary
  four, these do not depend on separate q/k/v projections, so they read on a
  fused-qkv family too.
- **`attention_scores` is the write surface `attention_probs` could not be.**
  They are the same tensor one step apart and have identical axes; what differs
  is what happens next. After the pattern comes the value multiply, which
  assumes rows summing to 1 and gets whatever an edit produced — so the pattern
  accepts only `swap`. After the scores comes the model's own softmax, which
  renormalizes by construction — so **every mechanism is legal**. Attention
  knockout is an `add_scaled` of a large negative mask; head boosting is a
  scale. Note that a *uniform* shift is a no-op, because softmax is invariant to
  a shift along the axis it normalizes: a knockout has to be targeted, which
  means a full-shape operand rather than a scalar. `gaussian` is refused, since
  its noise is drawn per feature axis and this tap has none.
- **Continuation reads are refused where the steps do not stack.** A decode step
  attends over the whole KV cache, so a tensor indexed by the positions being
  attended *to* grows by one per step while the query axis stays 1.
  `attention_probs`, `attention_scores` (two position axes) and `attention_key`
  (one position axis, over the keys) therefore refuse in the `generated` frame;
  `attention_query` and `attention_z` are query-axis-shaped and read normally.
  Writes never need the rule — rule 16 already makes them prefill-only.
- **`attention_probs` is the whole attention pattern**, `(batch, heads, query,
  key)`, and round 1 exposes it whole: `pos: "all"`. Both of its trailing axes
  are positions — its *feature* axis IS a position axis — so addressing one
  query row, attaching a featurizer, or slicing `dims` is **refused** rather
  than approximated. Those three refusals are not written per component: the
  tap declares its axes as `(batch, head, position[query], key_position[key])`,
  which has two position axes and therefore no `(batch, position, feature)`
  form, and each refusal follows from something that form would have provided.
  A write replaces the whole pattern, which is what an
  interchange on attention means, and both inputs must have the same number of
  positions. The edit is handed back to the model's own value multiply — nothing
  recomputes it — so a write here works on every family whose eager attention
  the backend can wrap, and only `swap` is refused arithmetic because nothing
  downstream restores rows summing to 1.
- **The Gated DeltaNet interior begins at its module boundaries** (round 4.1):
  `delta_qkv` is `in_proj_qkv`'s output — the fused `[q | k | v]` projection,
  whose three widths are *unequal* (`key_dim`/`key_dim`/`value_dim`), so it has
  no head axis and reads whole (or via `dims`); `delta_gate` is `in_proj_z`'s
  output, the output gate, value-head space; and `delta_premix` is `out_proj`'s
  **input** — the post-norm, post-gate mixer value, the exact analogue of
  `attention_premix`, which is why the name. All three require a
  `linear_attention` layer: at a full-attention layer they refuse with the
  mirror of the DeltaNet refusal ("a gated-attention mixer computes no
  delta-rule state"), and a family with no linear stream anywhere (llama,
  GPT-2) hits that refusal at every layer. The conv output and the kernel
  boundary are *function* taps (round 4.2) — the `conv1d` module never fires.
- **Seven more DeltaNet boxes live at the kernel boundary** (round 4.2), as
  arguments and returns of two module-global call sites the forward uses:
  `delta_conv` is `causal_conv1d_fn`'s return (channels-first, the fused
  unequal widths again, so no head axis); `delta_query`/`delta_key`/
  `delta_value` are the kernel's first three arguments — post-conv,
  GVA-**tiled** to the value-head count, and **pre**-l2norm (the kernel
  normalizes and scales internally, so these are the tensors a write can
  steer); `delta_beta` (`sigmoid(in_proj_b)`) and `delta_decay` (the
  log-decay `g`, negative reals) are its per-head gates, whose feature axis
  IS the head axis; `delta_kernel_output` is its return — the pre-norm,
  pre-gate `core_attn_out`, pinned by
  `norm(delta_kernel_output, delta_gate) == delta_premix` (exactly). The
  wrappers swap the modeling file's own globals for the dynamic extent of the
  tapped mixer's forward and call through to the originals — so whatever
  hub/`fla` dispatch the environment resolved keeps computing, and identity
  is bit-exact by construction. Both delta-rule kernels and both conv
  entry points are swapped together, so cached decode steps (which natively
  run the recurrent kernel and `causal_conv1d_update`) are tapped identically
  to prefill — `delta_key` therefore reads in the generated frame, unlike
  `attention_key` (the kernel receives one step's k, not the prefix). A
  `kernelize()`d mixer (a hub-kernel class forward) is refused by name, as is
  a family whose modeling file does not export the four globals. The untiled
  q/k and the post-split views are not components (F7: one box, one address —
  they are `delta_conv` rows re-viewed).
- **The DeltaNet per-step interior is read by stepping the library's own
  recurrent kernel** (round 4.3) — intercept, never transcribe, §2.3 of the
  round plan. `delta_state` is the recurrent state `S_t`: one `d_k × d_v`
  matrix per head per step, the second shape with no feature space (after the
  attention pattern) — but unlike the pattern it keeps its one position axis,
  so positions gather on the *steps* axis, `head:` selects a matrix stack, and
  `featurizer`/`dims` refuse off the declared axes. `delta_kv_mem`
  (`(S_{t-1}·exp(g_t) · k̂_t).sum`) and `delta_state_update` (`(v_t −
  kv_mem_t)·β_t`, the diagram's `delta`) are derived from adjacent states and
  pinned by the reconstruction identity `S_t == S_{t-1}·exp(g_t) + k̂_t ⊗
  delta_t` against the kernel's own returned states, exactly. At **prefill** a
  read runs the stepwise loop in the chunked call's *shadow*: the base forward
  is bit-identical, and the cost is O(seq) extra kernel calls at the tapped
  layer only (on a real checkpoint a full-seq all-layers `delta_state` is
  `layers · seq · heads · d_k · d_v` floats — address positions early). At
  **decode** the model runs the recurrent kernel natively, so generated-frame
  reads are plain per-step captures, pinned cross-path against test-side
  stepping. A **write** to `delta_state` substitutes the stepwise loop for the
  chunked call so edits feed forward — the one deliberate path-forcing in the
  vocabulary, costing ~5e-7 on the fixture's logits, pinned per layer as a
  bound. Its tensor operand must cover exactly the write's addressed steps
  (step-for-step, no broadcasting). `delta_kv_mem` is **read-only** (a memory
  readout has no independent existence — write `delta_state` or `delta_value`)
  and `delta_state_update` writes are deferred (D6: they lower exactly onto a
  state edit via the reconstruction identity).
- **`stream` names a mixer stream, and it is a per-layer fact.** It is one of
  `full_attention` / `linear_attention`. A hybrid tower carries a different mixer
  at different depths (Qwen3.6's text tower alternates Gated DeltaNet with gated
  full attention), so a site whose declared `stream` contradicts the layer it
  names is refused at load. `attention_probs` requires
  a full-attention layer: a linear-attention block computes no attention matrix,
  so the refusal there is about the architecture and is permanent.
- Sites are pure data — no behavior, no model handles.

### 2.5 `featurizers`

`featurize(x) → (f, err)`; `inverse(f, err) → x̂`; both defined per kind.

| kind | featurize | param slots | authored fields |
|---|---|---|---|
| `identity` (default) | `(x, 0)` | — | — |
| `subspace` | `(Qᵀx, 0)` | `weight` | `k`, `parametrization` ∈ `cayley` \| `matrix_exp` \| `stiefel`, `seed` |
| `pca` | `(Pᵀx, 0)` | `weight` | `k`, `file_path` |
| `sae` | `(enc(x), x − dec(enc(x)))` | `enc, dec, b_enc, b_dec` | `file_path` |
| `standardize` | `((x−μ)/σ, 0)` | `mu, sigma` | `file_path` |
| `gate` | `(σ(θ)⊙x, (1−σ(θ))⊙x)` | `theta` | — |

- **Widths are derived** from (model, site) — never authored. Only choices
  (`k`, `parametrization`, `dtype`, `init`, `seed`) are authored.
- **Params are auto-declared** per kind, named `<featurizer>.<slot>`.
- **Composition**: a `featurizer` reference may be a list `["rot", "gate"]`,
  applied left-to-right with a per-stage `err` list.
- **Error-term contract**: `err` and unselected dims always come from the
  pre-write value at the address — so a zero write ablates only the feature
  contribution, and a `dims` write is a subspace swap.
- **`seed`** (optional, `subspace` only): the draw its **initial** rotation
  comes from. Absent, it is the document's seed (`train.seed`, or 0 with no
  fit), so an existing document is unchanged and its canonical form does not
  grow a field. Illegal together with `file_path`: a loaded featurizer's
  weights are its bytes and it draws nothing.
  - This is what makes an *untrained* `subspace` a first-class **random rank-k
    basis**: `Q = qr(randn(d, k))` at that seed, orthonormal by construction.
    With `seed: {"sweep": [0, 1, 2]}` and no `train` block, one document is the
    matched-k random-subspace control — three draws at one cell, scored exactly
    as the fit was, in one model load. Without an authorable seed the control
    needed a `train` section it had nothing to train, which is why every study
    that wanted it built the draws by hand.
    See `configs/protocols/random_subspace_control.json`.
- **`file_path`** (optional): load a fitted artifact instead of computing.
  Its `ArtifactIdentity` (sec. 8) is checked; mismatch refuses. A loaded
  featurizer may not appear in `train.params`.
  - ⚠️ **`model_dtype` is part of that identity, and `--dtype` is not a way to
    satisfy it.** The compared value is the document's `model.dtype`, and a
    `model` block with no `dtype` **implies `fp32`** — so an apply document that
    omits it is refused against a fit that declared `bf16`, with
    `[V15] … implies 'fp32' but the bundle was stamped 'bf16'`. `--dtype` is a
    `--set model.dtype=…` shorthand on a **document** run, and a *workflow* run
    does not accept it at all — so a chained fit → apply can only be repaired in
    the file. Write `"dtype": "bf16"` into the apply document's `model` block,
    next to the fit's. The refusal message says so, and names the stamped
    value.
- **`entry`** (optional, only with `file_path`): which entry of that bundle.
  A swept document writes one file across all its points, keyed by
  coordinate (`weight[k=8,seed=0]`, sec. 2.12), so "the fit at k=8, seed=0"
  is `{"entry": {"k": 8, "seed": 0}}` — coordinate *names*, as they appear
  in the key, not full axis ids.
  - Omitted, the entry is implied by the **consuming point's own
    coordinates**: axis identity is name identity (sec. 3), so a document
    swept on `featurizers.rot.k` selects the fit at *its* `k` and the two
    sweeps zip instead of crossing. Coordinates the producer never had are
    ignored; a bundle with a single entry needs nothing.
  - **Authored, it is used exactly as written and is not completed from those
    coordinates.** The two spellings are alternatives, not layers: completing
    one from the other would make a selector's meaning depend on which axes the
    consuming document happens to sweep. So a *partial* `entry` against a
    multi-axis bundle resolves only when it is already unique, and otherwise
    refuses naming the coordinates that would disambiguate — pinning `k`
    elsewhere in the document does **not** narrow an `entry` that omits `k`.
    Name every varying coordinate, or drop `entry` entirely.
  - A selection that matches no entry, or more than one, is a **load
    error** — never first-hit-wins. Inside a workflow it is caught before
    any step runs, since a producing document's entry names follow from its
    own expansion (workflow spec sec. 5.10).
  - All of a bundle's slots for one entry come from the same point: an
    SAE's `enc` and `dec` cannot be crossed between fits.

### 2.6 `params` (optional)

Free tensors owned by no featurizer (steering vectors, a free written value):

| field | meaning |
|---|---|
| `file_path` | constant tensor, loaded |
| `entry` | which entry of that bundle (sec. 2.5), plus the reserved `slot` key |
| `shape`, `init` | trainable free tensor (must then appear in `train.params`) |

- A loaded constant is read from the bundle's `value` tensor by convention.
  A bundle *harvested from a read* is keyed by that read's name instead, so
  `{"entry": {"slot": "acts"}}` names it. `slot` is a params-only key — a
  featurizer's slots are fixed by its kind.

### 2.7 `reads`

```json
"v_cf":  {"site": "target",  "pos": -1, "model": "original", "input": "counterfactual"},
"logits": {"site": "lm_head", "pos": -1, "model": "patched",  "input": "base"}
```

| field | meaning |
|---|---|
| `site`, `pos` | the address |
| `model` | `original` (un-intervened) or a declared intervened_model |
| `input` | `base` \| `counterfactual` \| `counterfactual[j]`. For an intervened model this is redundant with the IM's own `input` and is **cross-checked** — mismatch is a load error |
| `featurizer` | optional; value is read in feature space |
| `dims` | optional static index list into the feature axis; default = all |

- Value = `featurize(activation at (site, pos) in model)[dims]`.
- A read in model `M` sees the activation **with all of `M`'s writes applied**
  (upstream and at the same address). To read an un-written value, read in
  `original` (or an IM without that write).
- Reads never carry `do`.

### 2.8 `writes` and the `do` algebra

```json
"patch": {"site": "target", "pos": -1, "featurizer": "rot", "do": {"swap": "v_cf"}}
```

- A write is an **inert definition**: no `model`, no `input`, no conditions.
  It executes inside every intervened_model that lists it.
- Effect at its address: `write(inverse(scatter(do(f[dims]) into f), err))` —
  untouched dims and `err` from the pre-write value (sec. 2.5).
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
| `{"pytorch_fn": {"qualname": "…"}}` | arbitrary | absolute; **local-only** — refused at load by any non-local engine |

- Per (site, overlapping pos, model): **at most one absolute write**; any
  number of additive writes. Application order: absolute first, then additive
  deltas summed. This replaces any commutativity analysis and makes write sets
  order-free.
- `gaussian.axis` tells a tensor-parallel engine whether the draw is
  replicated or sharded across ranks; `seed` is part of the hash.
- **An operand is read from at or above the address it lands on** (rule 20).
  Deeper is *executable* — the operand's model runs first, which is exactly
  what sec. 2.9's acyclicity licenses — but the network has no edge from the
  deeper address to the shallower one, so a write fed that way is attributable
  to no path, and a number attributable to no path is what this rule exists to
  refuse. Equal depth is the two-pass idiom: harvest a receiver's value in one
  intervened model, inject it at that same address in another. If the value is
  genuinely meant as an externally supplied constant rather than a routed
  activation, say so — harvest it into a bundle in its own run and load it as a
  `params` entry (sec. 2.6), where being a constant is the declared thing.
  Because the order is `(layer, intra-block rank)` and not layer alone, the
  rule is block-order aware: a head's contribution feeding the same layer's MLP
  is upstream, and the reverse is refused.

### 2.9 `intervened_models`

```json
"patched": {"input": "base", "writes": ["swap_sender", "freeze_10", "freeze_11"]},
"final":   {"input": "base", "writes": ["inject"]}
```

| field | meaning |
|---|---|
| `input` | **mandatory** — `base` \| `counterfactual` \| `counterfactual[j]` |
| `writes` | the writes in force; **unordered** (canonical form sorts) |

- `original` is the reserved name for the un-intervened model (on any input);
  it is never declared.
- **Membership rule**: every declared write appears in ≥ 1 intervened_model.
- **Cross-model data flow has exactly one channel**: a read in model A may be
  the operand of a write in force in model B. No direct IM→IM wiring, no
  inheritance. The graph (IM → writes → operand reads → IMs) must be acyclic —
  it is the execution schedule's skeleton.

### 2.10 `metrics`

Closed vocabulary; `of` names a read. The other value fields do **not** all
mean the same thing, and the difference has cost real debugging time:

- `a` / `b` / `token` / `target` / `expected` name **dataset columns** — the
  answer is per row, so the document names the column and the table carries the
  string.
- `class_probs`'s `groups` holds **literal token strings**, `{name: [tokens]}`.
  A class is a property of the answer *space*, one for the whole run, so there
  is no column to read it from.

| kind | fields | what the value fields name | result per example |
|---|---|---|---|
| `logit_diff` | `of, a, b` | columns | `logits[a] − logits[b]` |
| `token_logit` | `of, token` | column | `logits[token]` |
| `cross_entropy` | `of, target` | column | CE against target |
| `kl` | `of, target` | a **read** | KL between two reads' distributions |
| `class_probs` | `of, groups` | **literal token strings** | summed probability per group |
| `top_k` | `of, k, by` | — | the k top-ranked entries of the read (see below) |
| `match` | `of, expected` (+ optional `mode`) | column (of a string, or a **list** of equivalent forms) | match indicator |
| `decode` | `of` | — | the addressed tokens as text |

**Reads a kind may bind to.** Every kind but `kl` and `top_k` names *vocabulary
entries* — an authored string resolved to a token id — so it binds to a *plain*
`lm_head` read and a metric over anything else is a load error. Plain means no
`featurizer` and no `dims`: a featurizer re-expresses the projection in its own
latents and `dims` re-indexes a slice, so under either one the read's entries
are no longer token ids even though the site says `lm_head`. `kl` compares two
reads against each other; `top_k` reports indices along whichever axis its read
has. Those two bind to a read at any component.

**Domains.** Every kind consumes one of two things from its read, and which
one is a property of the kind:

| domain | kinds | consumes |
|---|---|---|
| `distribution` | everything above except `decode` | the read's dense value at the addressed positions — the vocabulary projection for every kind but `top_k` |
| `ids` | `decode` | only the tokens the decode produced |

An `ids` kind therefore obliges **no** vocabulary projection anywhere (§8's
materialization requirement) — a text probe is cheap by construction, not by a
engine's cleverness. It also only means something where tokens were
*produced*: `decode` binds to a read whose position carries `generated` (§2.3),
and a `decode` over a prompt-frame read is a load error.

#### `top_k` — one kind over any read

`top_k` is the reduction for "I only want the largest few entries per row". Its
reason to exist is that the alternative is saving the whole tensor to disk and
argsorting it later: a 4k-wide residual stream, or a 100k-latent SAE/BSF
featurizer output, times every example and every position. Like `reduce: mean`
on a save entry (§2.12), it happens **where the rows are gathered**.

So `top_k` binds to any read — `lm_head`, `block_output`, `mlp_activation`, a
featurizer's output. There is deliberately no `top_dims` / `top_features`
sibling kind: one kind, disambiguated by mandatory fields.

- **`k`** (mandatory, integer) — how many entries per row. `1 ≤ k ≤ width`.
- **`by`** (mandatory, `value` | `abs_value` | `prob`) — the ranking rule. It
  is mandatory because only the author knows what the axis is, and the answers
  differ: a vocabulary projection has no meaningful negative entries, while a
  residual stream and a signed feature code do, so ranking an SAE code by
  signed value and by magnitude return different sets.
  - `value` — the k largest signed entries. Any read.
  - `abs_value` — the k largest by `|x|`; the reported value stays signed. Any
    read.
  - `prob` — softmax the last axis, then take the k largest probabilities.
    **Plain `lm_head` reads only** — a softmax across neurons, SAE latents, a
    featurizer's re-expression of the projection or a `dims` re-index of it
    normalizes over an axis that is not an event space, so its "probabilities"
    would be probabilities of nothing; validation refuses it elsewhere. A
    pre-`by` document that ranked logits meant `prob`.

**Result columns.** Each has one fixed meaning, in every document. A column is
*absent* when it does not apply — never reinterpreted:

| column | meaning | emitted when |
|---|---|---|
| `indices` | index along the read's last axis (a token id on `lm_head`, a neuron on `mlp_activation`, a latent on a featurizer output) | always |
| `tokens` | that index decoded as a token string | the read is a plain `lm_head` tap |
| `values` | the **raw** read value at that index | always |
| `probs` | the softmax probability over the vocabulary | `by: "prob"` |

`values` is always raw — a logit under `by: "prob"`, not the probability — so a
downstream reader never has to know the ranking rule to know what it is
holding. The normalized number lives in its own column.

**Metrics over several positions.** A read may address more than one position —
every generated token of a row, a window of them, the tokens where the model
said something (§2.3). A metric over such a read reduces **per position**, and
its table says which:

- one row per (example, position), carrying the `step` it scored and a
  `matched` flag; `decode` is the exception that reduces the whole window to
  one string, and its `step` is null because no single step owns it;
- an example that addressed **nothing** — it stopped generating, or never said
  what a `variable` anchor looked for — still gets exactly one row, with a null
  value and `matched: false`. "The model never said it" must not be
  indistinguishable from "it said it and scored 0";
- prompt-frame metrics are unchanged: one row per example, no `step` column.

- A metric binds to exactly one read → one (model, input). Same metric in two
  models = two reads + two metrics.
- Metrics are gather-then-reduce over read values and dataset columns —
  nothing else. Cross-read arithmetic (differences of saved metrics) is
  post-hoc analysis. The vocabulary stays closed so engines can lower kinds
  to fused/vocab-parallel implementations.
- **`token_form`** (**required**, `auto` | `bare` | `space_prefixed`) — how
  this metric's string answers become token ids. Required on every kind that
  names token strings (`logit_diff`, `token_logit`, `cross_entropy`,
  `class_probs`, `match`); `kl` and `top_k` never resolve a string and refuse
  the key. That stays true under `top_k`'s any-read semantics: it *reports*
  indices it found and decodes them only when the read taps `lm_head` — it
  never turns an authored string into a token id, so the knob would have
  nothing to apply to.
  - **Why required rather than defaulted.** How a string becomes a token id is
    a fact about the model's tokenizer, and a document that does not say which
    rule it means gets whichever one the library happened to prefer. That guess
    has been measurably wrong four ways in production: a leading space
    (`" ?"`=907 against `"?"`=30), punctuation that merges with the token
    before it, two authored forms resolving to one id and being summed twice,
    and the non-case of digits where `" 7"` really is two tokens and `auto` is
    right. `auto` remains available — as something a document *chooses*.
  - `auto` tries `" " + s` first and falls back to `s`. That is right when the
    answer follows a space in the prompt — weekdays, names, MCQA letters.
  - It is **wrong** when the answer does not follow a space and both forms
    happen to be single tokens. Under gpt2, `"?"` is token 30 and `" ?"` is
    token 5633: a `match` on a punctuation answer scored 5633, the model emits
    30, and the metric read a flat 0.000. Pin `token_form: "bare"` for those.
    **`auto` now refuses** rather than guessing whenever the two forms
    disagree — as a single token each (the case above), or, under
    `mode: "first_token"`, on which piece they credit. It warned before, and a
    warning that produces a wrong number anyway is not a check.
  - The form applies to every token string in the metric, so a `class_probs`
    whose groups mix spaced and bare answers must stay on `auto`.
  - ⚠️ **A leading space in an authored value is normalized away, not
    honored.** The resolver strips it and then `token_form` alone decides the
    form, so `" X"` and `"X"` name the same answer. The consequence is worth
    stating because it is the opposite of what the spelling suggests: the
    common `["X", " X"]` idiom — written to "cover both forms" — is **inert**.
    Both entries resolve identically, and in a `class_probs` group, where the
    kind sums its members' ids, that used to be summed twice and report a
    probability above 1 (a measured **1.9927**). A group whose members collide
    on one id is now refused; list each answer once and say which form with
    `token_form`.
- A column value resolves to **one token**, space-prefixed form first; a
  multi-token value refuses rather than silently scoring its first piece.
- `match` is the exception, and only when told: its `expected` column may hold
  a **list of equivalent surface forms** (synonyms, casings — the argmax
  matching any of them scores 1.0), and `"mode": "first_token"` credits a
  form's first token instead of requiring the form to be one token. `mode` is
  `"exact"` by default and is materialized into the canonical form, so an
  omitted `mode` and an authored `"exact"` digest identically.
  - Which forms are equivalent is **task data**, serialized as a column when
    the table is built (from the causal model's `output_tokens` declaration) —
    never a document-side string transform. Case folding included: a task that
    wants case-insensitivity serializes the casings as forms, because at this
    layer the comparison is between token ids, not strings.
  - `first_token` is what "prefix" means with logits at one position. It
    over-credits an answer space that is not first-token-distinct; whether a
    table's answer space *is* first-token-distinct is a property of the
    dataset, so the mode is opt-in per document and never a default — **and
    the metric refuses** a row set in which two different answers share a
    first token, which is the last place the claim can be checked before a
    number exists. `" 85"` is `[220, "8", "5"]` on Qwen, so an emitted `87`
    would otherwise score 1.000 against an expected `85`.

### 2.11 `train`

| field | meaning |
|---|---|
| `objective` | `[[weight, metric_or_reg], …]`; reg = `{"l1": name}` \| `{"l2": name}` where name is a featurizer (all its params) or a dotted slot |
| `params` | what is optimized: featurizer names (all slots) or dotted slots; the **only** trainability declaration |
| `optimizer` | `{name, lr, …}` — lr/schedule/clip live here |
| `steps` | `{"epochs": n}` or `{"updates": n}` |
| `batch` | `{"pairs": n}` — counts base+counterfactual **pairs**, not rows |
| `anneal` | dotted-path schedules, e.g. `{"gate.theta.temperature": [start, end, frac]}` |
| `precision` | `{feature, loss}` dtypes — the *model's* dtype is `model.dtype` (§2.1), one home per fact |
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
- `file_path` is relative to the run's output directory. **JSON and
  safetensors are the only two formats**: dense numerics → `.safetensors`,
  per-example metric tables → `.json`, an array of row objects. Row labels
  repeat — that is the deliberate trade for a file `jq` and a human can both
  read. One file per metric, so a document saving three metrics writes three
  tables. In swept documents
  the path is unchanged; axis coordinates become columns / keyed entries
  (`weight[k=8,seed=0]`, one record per entry in the header's `entries`
  table — sec. 8).
- **`reduce`** (optional, reads only): save a statistic over the read's
  gathered rows instead of the rows. Closed vocabulary, `mean` in v1:
  `(…, width)` collapses to `(width,)`, the broadcast form a write operand
  takes (sec. 2.8) — mean ablation is a harvest with `reduce: mean` feeding
  a `params` constant. The reduction happens where the rows are gathered, so
  the un-reduced harvest never reaches disk: an ablation grid over an 8B
  model's layers is gigabytes of activations for kilobytes of means. A
  metric already reduces its read, and a featurizer bundle holds fitted
  parameters rather than rows; `reduce` on either is a load error.
- Rules (all load errors): every metric saved (no objective/eval exception —
  the loss trajectory is always in the results) · every trained featurizer
  saved · untrained or `file_path`-loaded featurizers not saveable · writes
  and intervened_models not saveable.

## 3. Sweeps

- **Every axis is an explicit wrapper** on a field of a named table entry:
  `{"sweep": [v1, v2, …]}` or `{"sweep": {"range": [start, stop, step?]}}`.
  Bare arrays are never axes. Works on scalar- and list-typed fields alike.
- **Axis identity = name identity**: sweeping `sites.target.layer` moves every
  read/write/metric referencing `target` together. The same list written on
  two fields is two axes (a cross) — share by referencing one name.
- Axes **propagate through the reference graph**; entities off the axis stay
  singletons shared by all points.
- Multiple axes form the **cross product**; nothing else (no zip, no
  conditionals; dependent axes are a generator's job — emit the JSON).
  - ⚠️ **`data.base.dataset` and `data.counterfactual.dataset` are two names,
    hence two axes.** Sweeping both over the same *n* refs is an *n*×*n* cross
    that pairs every base table with every counterfactual table, and rows are
    paired **across roles by index** (sec. 2.2) — so *n*²−*n* of those points
    silently score mispaired rows rather than failing. One campaign hit the
    4096-point cap this way and read the cap as the symptom. There is no zip to
    reach for: the intended shape is **one table carrying both sides**, base
    reading `input` and the counterfactual role reading
    `counterfactual_inputs[j]` of the *same* ref — which is what every shipped
    preset does, and what leaves one axis to sweep. When the two sides really
    are separate files, that is one document per pair (or one `--set` per
    shard), not one document with two axes. Letting two fields share one axis
    is a spec change and its own PR.
- Coordinates suffix derived names (`rot[k=8]`) and key results.
- Expansion is **deterministic at load**: one document ⇒ a set of point
  protocols. The document digest names the campaign; each point's digest is
  the provenance unit. The planner content-dedups sub-values shared across
  points — shared harvests and forwards fall out automatically (identical
  reads intern to one read).
- **The planner derives the sharing; the engine claims it.** A forward
  group's `digest` is the identity of everything determining its activations,
  and **taps are deliberately not in it** — reading layer 3 or layer 23 of the
  same un-intervened forward is the same forward. So a 32-layer scan's
  counterfactual harvest is *one* group carrying 32 taps:
  `interned_groups(plans)` counts what a campaign owes (65) against the
  `sum(num_forwards)` a per-point loop pays (128). Spending that is execution's
  job, not the plan's — run each distinct digest **once**, capturing the
  **union** of the taps every point asked of it, and let each point gather and
  featurize its own value out of that one capture. The reference engine does
  this and reports what it ran as `RunResult.forwards`. The trade is memory: a
  shared pass holds every tapped address at once where a per-point loop held
  one. An engine that interns nothing is still correct, only slower — it
  leaves `RunResult.forwards` at 0, which reads as "not measured".

## 4. Execution semantics

- **Models → forwards.** For each expanded point, the models are: `original`
  on every input it is read on, plus each intervened_model. Each (model)
  is one forward group over its input rows; fusion, batching, and staging
  across groups are the engine's choice. `num_forwards` is derived, never
  authored.
- **Within a model**: apply each in-force write at its address (absolute
  first, then additive sum); reads see the fully written state.
- **Across models**: operand values flow along the acyclic model graph;
  the engine stages them (fused multi-pass, saved constants, or microbatch
  wiring — its call).
- **Elision**: a model whose reads are all satisfied may stop its forward
  after the deepest tap; a full-depth pass is never owed. A group that
  decodes is the exception — every step needs the head, so nothing is
  elided. When several points **share** the group (sec. 3), the deepest tap is
  the deepest of the union: the one pass has to serve all of them, so read
  `stop_after` off the interned group, never off the point that ran first.
  Interning still wins against elision — 32 passes elided at layers 0..31 cost
  ~16x the single full-depth pass that replaces them.
- **Decoding groups.** A group whose reads address the continuation
  (sec. 2.3) runs one prefill plus `max_new_tokens` greedy steps: `n` tokens
  need `n` steps, because the last generated token must be consumed by a
  forward for its own activations to exist. Writes apply in the prefill only.
  The depth is derived from the group's positions, never authored.
- **Determinism**: `gaussian` draws from its declared seed; sweep expansion
  and canonicalization are pure functions of the document.

## 5. Validation — load-error checklist

A conforming loader rejects the document unless all of these hold:

1. Strict keys: unknown fields anywhere are errors; closed enums reject with
   suggestions. Derived fields (sec. 7) may not be authored.
2. Section order per sec. 1; `save` last.
3. Global namespace: no duplicate names across sections 5–12; no reserved
   names (`base`, `counterfactual`, `counterfactual[j]`, `original`, `all`)
   declared.
4. Every reference resolves: sites (declared inventory only), positions,
   featurizers, params, reads, writes, intervened_models, metrics.
5. Reads: `model` ∈ `original` ∪ IMs; `input` a valid role; if `model` is an
   IM, `input` equals the IM's `input`.
6. Writes carry no `model`/`input`/conditions; operands name reads, params, or
   literal scalars.
7. Every write is in ≥ 1 intervened_model; every IM has a mandatory valid
   `input`; the model graph is acyclic.
8. Per (site, overlapping pos, model): ≤ 1 absolute write.
9. `dims` selections co-occurring at one address in one model are disjoint.
10. `save` non-empty; entry shapes exact; bindings match resolution; every
    metric and every trained featurizer saved; nothing else saveable.
11. Sink rule: every read is saved, a metric input, or an operand.
12. Loaded featurizers (`file_path`) are not trained; trained featurizers are
    declared kinds with trainable slots.
13. `pytorch_fn` present ⇒ refused unless the selected engine is local.
14. Sweep wrappers well-formed; the expanded point count is reported (and may
    be capped without an explicit override flag).
15. Artifact-valued fields resolve (missing artifact = error, never a
    default).
16. Generation is read-only and prefill-only: no write's `pos` carries
    `generated`, and `train` does not co-occur with a `generated` position.
17. The model's realization is coherent: a `quantization` block carries only
    the knobs its own scheme has (`double_quant` is 4-bit vocabulary,
    `int8_threshold` is int8 vocabulary).
18. Composition (§1.1): a split document carries both halves, `application`
    first; the method declares neither `model` nor `data` and does declare
    `reads` and `save`; the application declares `model` and `data`; a method
    *file* agrees with the document on `version`; every leaf is supplied by
    exactly one half, or by both with the same value; and the composition is
    closed — every input and every site address bound. An unfilled hole is
    refused with the list of what is missing.
19. Write widths are uniform: every row a write addresses carries the same
    number of positions. Only an `all` or `variable` write can be ragged, and
    only the tokenizer can say how wide a row is — so, unlike the rest of this
    checklist, rule 19 is checked when the run encodes its inputs, **before any
    forward pass**, not at load. `validate` cannot decide it: the pure verbs
    hold no tokenizer, by design.
20. Operand reachability: every write operand that names a **read** is read at
    an address no deeper than the one the write lands on, in the `(layer,
    intra-block rank)` order of sec. 2.4's vocabulary. Equal is legal — that is
    the harvest/inject idiom. Params and literal-scalar operands have no
    address and are unconstrained.

## 6. Derived — never authored

| property | derivation |
|---|---|
| featurizer widths, param shapes | from (model config, site); parametrization internals are not authored |
| param slots | per featurizer kind (sec. 2.5) |
| `requires` | capability set, sec. 8 |
| `num_forwards`, fusion, staging | from the model graph; a compile property |
| decode depth, and what a continuation read obliges | from the group's `generated` positions, `save` and the metrics over it (sec. 8) |
| dataset content digest | resolved + stamped at load |
| point protocols + digests | deterministic sweep expansion |
| `ArtifactIdentity` | stamped into artifacts, sec. 8 |

## 7. Canonical form and digests

- **An experiment is a value, not a program.** One JSON document fully
  describes an experiment: it can be hashed, diffed, shared, and re-run.
  It never contains tensors, closures, resolved token indices, module
  references, or anything only one engine could interpret.
- **The parser owns execution.** The document says *what*; the parser/planner
  derives *how* (forward count, fusion, batching, sweep parallelization) and
  `explain` reports it.
- **Everything declared must reach a sink; everything derivable is derived.**
  Dead declarations are load errors. Authored files are minimal; the stamped
  canonical form materializes every default (sec. 7).
- **Format**: strict JSON (unknown keys = error). YAML is accepted at the
  authoring surface; the object model is normative. JSON has no comments —
  use `description`.
- **v1 scope**: prefill-only *interventions*. Greedy decode is addressable as a
  position frame (sec. 2.3) and readable; there are no decode-step writes, no
  sampling, and one neural model per document.
- **Canonical-stamp principle**: the authored file may be minimal; the
  canonical form materializes *everything* — every default (constant LR,
  optimizer betas, `model.dtype` and the quantization scheme's own knobs),
  every resolved reference (dataset digests, artifact values), every derived
  width, sugar expanded (int and `"all"` positions, alias `neural_model` →
  `model`), unordered lists sorted (IM write lists), sweeps expanded to
  points. `type` is authoring metadata and is dropped: the canonical form is
  the experiment, not the file.
- **Composition is transparent** (§1.1): a document authored as a method plus
  an application canonicalizes to the same bytes as the same experiment
  authored as one file, so how a point was reached never moves its digest.
  Method provenance (the method's own content hash) rides in the run record
  and the artifact stamp instead.
- `digest = sha256(canonical bytes)` — sorted keys, canonical floats; each
  param replaced by its content hash. Document digest = campaign; point
  digest = provenance unit, stamped on every artifact as `produced_by`.
- Any change to canonical form bumps `version` and ships a loader migration.
  Pin a golden corpus (canonical form + digest per example) in tests.

## 8. Engine contract

An engine implements these services:

| service | contract |
|---|---|
| `SiteResolver` | site record → tap in its execution engine (component vocabulary, sec. 2.4) |
| position resolution | Pos spec + `PositionFrame` (pad side, packing, sequence shard map) → indices; supports flat, per-row, and ragged windows |
| planner | model graph → forward groups; fusion/batching/staging; elision |
| cross-point interning | run each distinct group `digest` once, capturing the union of the taps every point asked of it; each point gathers and featurizes its own value from that capture, keeping its own per-entry provenance (sec. 3). Optional but expected — report what ran as `RunResult.forwards` |
| mechanisms | the closed `do` set, class order per address; refuse `pytorch_fn` if non-local |
| featurizers | kinds table with declared dtypes; error-term contract |
| metrics | lower kinds to native ops; derive minimal logit materialization (`logits_to_keep`, vocab-parallel CE) from `save` + metric needs |
| generation | greedy-decode a group to its derived depth; materialize a distribution only where `save` or a metric needs one (see below); writes stay in the prefill |
| training | own the `train` loop (optimizer, accumulation, anneal, early stop, checkpoints) — the document never changes across engines |
| RNG | realize `gaussian` per declared seed + axis semantics, bit-stable across parallelism layouts |
| stamping | write canonical point protocols + digests; `ArtifactIdentity` into every featurizer bundle's safetensors header |

`ArtifactIdentity` (stamped, checked on any `file_path` load; mismatch
refuses): `produced_by` digest · model key + revision · **model dtype +
quantization** · tokenizer · site record · `k` · parametrization · featurizer
dtype · trained-on data ref + digest · engine · code commit. A rotation
fitted against bf16 weights is not the same artifact as one fitted against
fp32 weights, and the stamp is what says so.

**Per entry, not per file.** A swept document writes one file from many
points, so the file-level stamp carries only what every point agrees on;
whatever differs (`k`, the point digest, a swept site) is stamped per tensor
key in an `entries` table in the same header — `{key: {slot, coords, …identity}}`.
That table is what makes an entry selectable (sec. 2.5) and provable: the
check runs against the record of the entry a document actually selects. A
bundle with no table is a single-point or hand-made artifact and is checked
at file level, as before.

**Capabilities.** `requires` is derived from the document; an engine declares
what it supports; `choose_engine = first b where requires ⊆ b.capabilities`;
refusal messages generate from the missing capability.

Two kinds of entry, one comparison. The **coarse verbs** below are the closed
`CAPABILITIES` vocabulary. **Component entries** are generated, never listed:
every site a read or write references contributes `component:<name>` (a write
also `component:<name>:write`), and each engine declares the component sets it
serves — so a document touching a component outside one engine's site
vocabulary routes past it to an engine that serves it, and the generated
refusal names the entry. The closed vocabulary behind these entries is the
sec. 2.4 `Component` literal itself. Stream- and layer-level constraints (a
full-attention box on a DeltaNet layer, a read-only component) stay
engine-internal policy: they depend on the loaded model or are true of every
engine, so routing on them would be either impossible or misleading.

| capability | required when |
|---|---|
| `grad` | `train` present |
| `paired_forward` | a write's operand read has a different `input` than the write's model |
| `full_logits` | a full `lm_head` read is saved, or a `class_probs` / `top_k` metric reads `lm_head` other than through a `dims` slice — a *featurized* `lm_head` read still obliges the whole projection (the featurizer consumes it) even though its value is latents. A `top_k` over any other component obliges no vocabulary projection (sec. 2.10) and must not be charged for one |
| `generate` | any position carries `generated` (sec. 2.3) |
| `quantized_weights` | `model.quantization` present (sec. 2.1) |
| `writable_attention_probs` | a write targets `attention_probs` |
| `pytorch_fn_local` | any `pytorch_fn` |
| `component:<name>`[`:write`] | generated — a read or write references a site with that component (writes add `:write`) |

Reference matrix:

| capability | nnsight (HF) | Megatron | SGLang/serving |
|---|---|---|---|
| `grad` | ✓ (≤ ~1 node) | ✓ (the point of it) | ✗ |
| `paired_forward` | ✓ fused invokes | ✓ pairs per microbatch | ✗ |
| arbitrary writes | ✓ | ✓ | additive steering only |
| `full_logits` | ✓ | ✗ vocab-parallel only | ✓ |
| `quantized_weights` | ✓ bitsandbytes | ✗ | ✓ its own quantizers |
| `pytorch_fn_local` | ✓ | ✗ | ✗ |

**Materialization (generation).** A continuation read's cost is not the decode,
it is the vocabulary: at batch 32 and 16 steps, every step's distribution over a
128k vocabulary is ~260 MB in fp32, one step is ~16 MB, a site's activations
~8 MB, the token ids ~2 KB. The planner therefore derives, per group, the decode
depth and — per continuation read — whether anything downstream consumes a
distribution: the read is saved, or a metric in the `distribution` domain
reduces it (sec. 2.10). An `ids`-domain metric does **not** count, which is the
point of the domain: a text probe — `decode` over a continuation read, nothing
saved — obliges no vocabulary projection at all, and an engine **must not**
build one where the answer is no.

*How* it complies is its own business: keeping only the addressed steps,
projecting a narrower slice (`logits_to_keep` takes an index tensor), replaying
the sequence teacher-forced, or a vocab-parallel reduction. The reference
engine keeps `ln_final` activations across steps and projects through the head
only at the addressed positions, which needs no second pass — an implementation
note, not a requirement. `explain` prints the obligation so the bill is legible
before a run.

**Execution scale.** Documents and workflows are scheduler-agnostic — they
never name devices, hosts, or job systems. The division of labor:

- A **engine** owns all intra-run execution: device placement, batching, and
  any parallelism across a campaign's points or across its own
  accelerators — declared, like everything else, through its capability set
  and constructor. The reference engine takes `device` and runs points
  serially; sharded and multi-device engines are engine work, not document
  vocabulary. **Precision is not on this list**: `dtype` and `quantization`
  change the numbers, so they are the document's (§2.1), and an engine reads
  them per point rather than being told once.
- **Job dispatch is site tooling outside this repository.** The one seam it
  needs is the CLI's `--points START:STOP` selector: an external scheduler
  expands nothing itself, launches `run` per index range, and recombines by
  digest — every shard stamps its artifacts as members of the same campaign
  (`document_digest` is unaffected by slicing).

## 9. CLI

The four verbs dispatch on the document's type (§1.1): a **workflow** runs its
step graph, a **method file** answers only what a method can answer
(`validate`, `digest`, and `explain`, which prints its signature — `run` is
refused: there are no inputs and no addresses), and a **protocol document**
runs the full pipeline, a split one composing its halves first.

| verb | effect |
|---|---|
| `run <doc>` | validate, expand, plan, execute, stamp; writes `<out>/protocol.json` — the canonical document, its digest, the per-point provenance digests, and the method it was composed from |
| `validate <doc> [--data]` | sec. 5 checks; `--data` also checks column and prompt-variable references, at every point |
| `explain <doc>` | models + forward plan, expanded point count, derived `requires`, resolved bindings, digest, what `save` produces |
| `--engine` (explain) | also route the document and print which engine would serve it, or the sec. 8 refusal. Opt-in: engines are heavy, and without it `explain` stays torch-free |
| `digest <doc>` | the campaign digest |
| `--set path=value` | ad-hoc override — exploration only; promote anything that matters into the file |
| `--device` (run) | reference-engine placement: any torch device string (`cpu` default, `cuda`, `cuda:1`, `mps`). Placement is execution; precision is not (§8) |
| `--engine` (run) | `auto` (default) is every installed engine with the reference **first**, routed by `choose_engine` (sec. 8); name one to pin it. A document never names an engine — it declares what it needs — so the default is routing rather than a choice the document did not make |
| `--dtype` (run) | shorthand for `--set model.dtype=…` — it edits the document, so the run's digest is the overridden document's and the record never lies about what produced the numbers. Refused on a workflow, whose steps each declare their own |
| `--points START:STOP` (run) | execute one half-open point-index shard of the expanded campaign (sec. 8, execution scale); document runs only — digests and stamps are unaffected |
| `--register-from-hf` | resolve an unregistered `model.key` from its HF config before loading, instead of refusing `[V4]`. Opt-in, so without it a digest never depends on the network; `run` always does it. On a workflow it pre-registers **every** inner document's key |

## 10. Worked examples

The same interchange experiment as one run document, split into its two halves
(§1.1). The method fixes the mechanism and the scoring and leaves the inputs
and the layer open; the application closes them. It composes to the one-file
flat document further down, digest for digest.

```json
{
  "version": "1",
  "description": "Llama-3.1-8B in bf16, answer-slot residual at layer 18, over the weekdays training pairs.",
  "application": {
    "model": {"key": "meta-llama/Llama-3.1-8B", "revision": "main", "dtype": "bf16"},
    "data": {
      "base":   {"dataset": "weekdays/train", "field": "input"},
      "counterfactual": {"dataset": "weekdays/train", "field": "counterfactual_inputs[0]"}
    },
    "sites": {"target": {"layer": 18}}
  },
  "method": {
    "description": "Interchange intervention: swap the answer-slot residual from the counterfactual into base; IIA scoring.",
    "causal_model": {"key": "weekdays.causal_model"},
    "sites": {
      "target":  {"component": "block_output"},
      "lm_head": {"component": "lm_head"}
    },
    "reads": {
      "v_cf":   {"site": "target",  "pos": -1, "model": "original", "input": "counterfactual"},
      "logits": {"site": "lm_head", "pos": -1, "model": "patched",  "input": "base"}
    },
    "writes": {"patch": {"site": "target", "pos": -1, "do": {"swap": "v_cf"}}},
    "intervened_models": {"patched": {"input": "base", "writes": ["patch"]}},
    "metrics": {
      "iia":        {"kind": "match",      "of": "logits", "expected": "cf_answer"},
      "logit_diff": {"kind": "logit_diff", "of": "logits", "a": "cf_answer", "b": "base_answer"}
    },
    "save": [
      {"value": "iia",        "model": "patched", "input": "base", "file_path": "iia.parquet"},
      {"value": "logit_diff", "model": "patched", "input": "base", "file_path": "logit_diff.parquet"}
    ]
  }
}
```

`causalab explain` on that document prints the composed plan and the method's
digest; on a method *file* it prints what is still to bind — `model`, `data`,
and `sites.target.layer`. A layer scan is a one-line edit of the application
half: `"sites": {"target": {"layer": {"sweep": {"range": [0, 32]}}}}`, with the
method untouched and still hashing the same.

Path patching (sender → receiver, off-path frozen; shows cross-model flow):

```json
{
  "version": "1",
  "model": {"key": "meta-llama/Llama-3.1-8B", "revision": "main"},
  "data": {
    "base":   {"dataset": "ioi/test", "field": "input"},
    "counterfactual": {"dataset": "ioi/test", "field": "counterfactual_inputs[0]"}
  },
  "sites": {
    "sender":   {"component": "attention_premix",  "layer": 9, "head": 9},
    "receiver": {"component": "block_input",      "layer": 12},
    "a10":      {"component": "attention_output", "layer": 10},
    "a11":      {"component": "attention_output", "layer": 11},
    "lm_head":  {"component": "lm_head"}
  },
  "reads": {
    "v_sender":   {"site": "sender",   "pos": -1, "model": "original", "input": "counterfactual"},
    "v_a10":      {"site": "a10",      "pos": -1, "model": "original", "input": "base"},
    "v_a11":      {"site": "a11",      "pos": -1, "model": "original", "input": "base"},
    "v_receiver": {"site": "receiver", "pos": -1, "model": "patched",  "input": "base"},
    "logits":     {"site": "lm_head",  "pos": -1, "model": "final",    "input": "base"}
  },
  "writes": {
    "swap_sender": {"site": "sender",   "pos": -1, "do": {"swap": "v_sender"}},
    "freeze_10":   {"site": "a10",      "pos": -1, "do": {"swap": "v_a10"}},
    "freeze_11":   {"site": "a11",      "pos": -1, "do": {"swap": "v_a11"}},
    "inject":      {"site": "receiver", "pos": -1, "do": {"swap": "v_receiver"}}
  },
  "intervened_models": {
    "patched": {"input": "base", "writes": ["swap_sender", "freeze_10", "freeze_11"]},
    "final":   {"input": "base", "writes": ["inject"]}
  },
  "metrics": {
    "logit_diff": {"kind": "logit_diff", "of": "logits", "a": "answer", "b": "cf_answer"}
  },
  "save": [
    {"value": "logit_diff", "model": "final", "input": "base", "file_path": "logit_diff.json"}
  ]
}
```

DAS with a k × seed sweep (9 fits from one harvest; shows axes + train +
featurizer save):

```json
{
  "version": "1",
  "model": {"key": "meta-llama/Llama-3.1-8B", "revision": "main"},
  "data": {
    "base":   {"dataset": "weekdays/train", "field": "input"},
    "counterfactual": {"dataset": "weekdays/train", "field": "counterfactual_inputs[0]"}
  },
  "sites": {
    "target":  {"component": "block_output", "layer": 18},
    "lm_head": {"component": "lm_head"}
  },
  "featurizers": {
    "rot": {"kind": "subspace", "k": {"sweep": [8, 16, 32]}, "parametrization": "cayley"}
  },
  "reads": {
    "v_cf":  {"site": "target",  "pos": -1, "model": "original", "input": "counterfactual", "featurizer": "rot"},
    "logits": {"site": "lm_head", "pos": -1, "model": "patched",  "input": "base"}
  },
  "writes": {
    "patch": {"site": "target", "pos": -1, "featurizer": "rot", "do": {"swap": "v_cf"}}
  },
  "intervened_models": {
    "patched": {"input": "base", "writes": ["patch"]}
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
    {"value": "iia", "model": "patched", "input": "base", "file_path": "iia.json"},
    {"value": "ce",  "model": "patched", "input": "base", "file_path": "ce.json"},
    {"value": "rot", "site": "target", "file_path": "rot.safetensors"}
  ]
}
```

## 11. Glossary (Geiger et al., arXiv:2301.04709)

| this spec | causal abstraction |
|---|---|
| `model` | the low-level model ℒ (the high-level model ℋ lives with the task's dataset, not in documents) |
| `intervened_models.<name>` | ℒ_{b∪𝕀} — the intervened model |
| a write's `do` | an interventional 𝕀_X |
| `swap` from a counterfactual read | interchange intervention (`IntInv`; `DistIntInv` when featurized) |
| site + pos + dims | the target variable set **X** |
| featurizer | the translation τ |
| `match` metric | interchange-intervention accuracy (IIA) |
