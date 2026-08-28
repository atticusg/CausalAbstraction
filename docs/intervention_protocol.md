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
| 4 | `data` | ✓ | input rows: `base` (+ `counterfactual`) |
| 5 | `positions` | – | named token-position specs |
| 6 | `sites` | ✓ | named activation addresses — the complete tap inventory |
| 7 | `featurizers` | – | named feature-space maps |
| 8 | `params` | – | free/constant tensors owned by no featurizer |
| 9 | `reads` | ✓ | value producers |
| 10 | `writes` | – | effect definitions (inert until listed) |
| 11 | `intervened_models` | –* | which writes are in force on which input (*required if `writes` present) |
| 12 | `metrics` | – | closed reductions over read values |
| 13 | `train` | – | the fit, declared |
| 14 | `save` | ✓ | the complete output manifest — non-empty, last |

- **One global namespace**: every name in sections 5–12 must be unique across
  all of them; reserved names: `base`, `counterfactual`, `counterfactual[j]`, `original`.
- All cross-references must resolve; references are by name, never inline
  duplication.
- **Artifact-valued fields**: anywhere a scalar or position is expected,
  `{"artifact": "<path>", "key": "<field>"}` reads one value from a prior
  run's artifact at load. Missing artifact = load error.

## 2. Section reference

### 2.1 `model`

| field | meaning |
|---|---|
| `model.key` | model name (HF key or registry name) — the network as a *name* |
| `model.revision` | checkpoint revision |

- `neural_model` is accepted as an alias of `model`; canonical form uses `model`.

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
- Dataset **columns** referenced by metrics and by `column` positions are checked
  against the resolved tables by `validate --data`, not at load.

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
  a backend service against a `PositionFrame` (pad side, packing, sequence
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
  chosen entities): being an explicit reference, it is checked by
  `validate --data`, where a variable that only happens to exist as a column
  is not.
- The value in a column position is a **string**, resolved like a variable's
  value (it must occur exactly once in the row's text). Integer token indices
  are deliberately not a v1 spelling: they would bind a table to one
  tokenizer, and a task that can compute an index can serialize the substring
  instead.
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
| `head` / `expert` / `stream` | optional sub-axes: attention head, MoE expert, residual-stream index |

Component vocabulary (per-backend `SiteResolver` maps each to a tap):

`embeddings` · `block_input` · `block_output` · `attention_output` ·
`attention_value` · `attention_probs` · `mlp_input` · `mlp_output` ·
`mlp_activation` · `router_logits` · `expert_output` · `ln_final` · `lm_head`

- **`sites` is the complete inventory**: every site a read or write references
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
  pre-write value at the address — so a zero write ablates only the feature
  contribution, and a `dims` write is a subspace swap.
- **`file_path`** (optional): load a fitted artifact instead of computing.
  Its `ArtifactIdentity` (sec. 8) is checked; mismatch refuses. A loaded
  featurizer may not appear in `train.params`.
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
| `{"pytorch_fn": {"qualname": "…"}}` | arbitrary | absolute; **local-only** — refused at load by any non-local backend |

- Per (site, overlapping pos, model): **at most one absolute write**; any
  number of additive writes. Application order: absolute first, then additive
  deltas summed. This replaces any commutativity analysis and makes write sets
  order-free.
- `gaussian.axis` tells a tensor-parallel backend whether the draw is
  replicated or sharded across ranks; `seed` is part of the hash.

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

Closed vocabulary; `of` names a read; other value fields name dataset columns.

| kind | fields | result per example |
|---|---|---|
| `logit_diff` | `of, a, b` | `logits[a] − logits[b]` |
| `token_logit` | `of, token` | `logits[token]` |
| `cross_entropy` | `of, target` | CE against target |
| `kl` | `of, target` (a read) | KL between two reads' distributions |
| `class_probs` | `of, groups` | summed probability per group |
| `top_k` | `of, k, by` | the k top-ranked entries of the read (see below) |
| `match` | `of, expected` (+ optional `mode`) | match indicator |
| `decode` | `of` | the addressed tokens as text |

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
backend's cleverness. It also only means something where tokens were
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
  post-hoc analysis. The vocabulary stays closed so backends can lower kinds
  to fused/vocab-parallel implementations.
- **`token_form`** (optional, `auto` | `bare` | `space_prefixed`; default
  `auto`) — how this metric's string answers become token ids. Legal on every
  kind that names token strings (`logit_diff`, `token_logit`, `cross_entropy`,
  `class_probs`, `match`); `kl` and `top_k` never resolve a string and refuse
  the key. That stays true under `top_k`'s any-read semantics: it *reports*
  indices it found and decodes them only when the read taps `lm_head` — it
  never turns an authored string into a token id, so the knob would have
  nothing to apply to.
  - `auto` tries `" " + s` first and falls back to `s`. That is right when the
    answer follows a space in the prompt — weekdays, names, MCQA letters — and
    it is the default so pre-`token_form` documents are unchanged.
  - It is **wrong** when the answer does not follow a space and both forms
    happen to be single tokens. Under gpt2, `"?"` is token 30 and `" ?"` is
    token 5633: a `match` on a punctuation answer scores 5633, the model emits
    30, and the metric reads a flat 0.000 with no error raised anywhere. Pin
    `token_form: "bare"` for those. `auto` warns when the two forms disagree.
  - The form applies to every token string in the metric, so a `class_probs`
    whose groups mix spaced and bare answers must stay on `auto`.
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
    dataset, so the mode is opt-in per document and never a default.

### 2.11 `train`

| field | meaning |
|---|---|
| `objective` | `[[weight, metric_or_reg], …]`; reg = `{"l1": name}` \| `{"l2": name}` where name is a featurizer (all its params) or a dotted slot |
| `params` | what is optimized: featurizer names (all slots) or dotted slots; the **only** trainability declaration |
| `optimizer` | `{name, lr, …}` — lr/schedule/clip live here |
| `steps` | `{"epochs": n}` or `{"updates": n}` |
| `batch` | `{"pairs": n}` — counts base+counterfactual **pairs**, not rows |
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
- **Within a model**: apply each in-force write at its address (absolute
  first, then additive sum); reads see the fully written state.
- **Across models**: operand values flow along the acyclic model graph;
  the backend stages them (fused multi-pass, saved constants, or microbatch
  wiring — its call).
- **Elision**: a model whose reads are all satisfied may stop its forward
  after the deepest tap; a full-depth pass is never owed. A group that
  decodes is the exception — every step needs the head, so nothing is
  elided.
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
13. `pytorch_fn` present ⇒ refused unless the selected backend is local.
14. Sweep wrappers well-formed; the expanded point count is reported (and may
    be capped without an explicit override flag).
15. Artifact-valued fields resolve (missing artifact = error, never a
    default).
16. Generation is read-only and prefill-only: no write's `pos` carries
    `generated`, and `train` does not co-occur with a `generated` position.

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
- **v1 scope**: prefill-only *interventions*. Greedy decode is addressable as a
  position frame (sec. 2.3) and readable; there are no decode-step writes, no
  sampling, and one neural model per document.
- **Canonical-stamp principle**: the authored file may be minimal; the
  canonical form materializes *everything* — every default (constant LR,
  optimizer betas, dtypes), every resolved reference (dataset digests,
  artifact values), every derived width, sugar expanded (int and `"all"`
  positions, alias `neural_model` → `model`), unordered lists sorted (IM
  write lists),
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
| generation | greedy-decode a group to its derived depth; materialize a distribution only where `save` or a metric needs one (see below); writes stay in the prefill |
| training | own the `train` loop (optimizer, accumulation, anneal, early stop, checkpoints) — the document never changes across backends |
| RNG | realize `gaussian` per declared seed + axis semantics, bit-stable across parallelism layouts |
| stamping | write canonical point protocols + digests; `ArtifactIdentity` into every featurizer bundle's safetensors header |

`ArtifactIdentity` (stamped, checked on any `file_path` load; mismatch
refuses): `produced_by` digest · model key + revision · tokenizer · site
record · `k` · parametrization · dtype · trained-on data ref + digest ·
backend · code commit.

**Per entry, not per file.** A swept document writes one file from many
points, so the file-level stamp carries only what every point agrees on;
whatever differs (`k`, the point digest, a swept site) is stamped per tensor
key in an `entries` table in the same header — `{key: {slot, coords, …identity}}`.
That table is what makes an entry selectable (sec. 2.5) and provable: the
check runs against the record of the entry a document actually selects. A
bundle with no table is a single-point or hand-made artifact and is checked
at file level, as before.

**Capabilities.** `requires` is derived from the document; a backend declares
what it supports; `choose_backend = first b where requires ⊆ b.capabilities`;
refusal messages generate from the missing capability.

| capability | required when |
|---|---|
| `grad` | `train` present |
| `paired_forward` | a write's operand read has a different `input` than the write's model |
| `full_logits` | a full `lm_head` read is saved, or a `class_probs` / `top_k` metric reads `lm_head` other than through a `dims` slice — a *featurized* `lm_head` read still obliges the whole projection (the featurizer consumes it) even though its value is latents. A `top_k` over any other component obliges no vocabulary projection (sec. 2.10) and must not be charged for one |
| `generate` | any position carries `generated` (sec. 2.3) |
| `writable_attention_probs` | a write targets `attention_probs` |
| `pytorch_fn_local` | any `pytorch_fn` |

Reference matrix:

| capability | nnsight (HF) | Megatron | SGLang/serving |
|---|---|---|---|
| `grad` | ✓ (≤ ~1 node) | ✓ (the point of it) | ✗ |
| `paired_forward` | ✓ fused invokes | ✓ pairs per microbatch | ✗ |
| arbitrary writes | ✓ | ✓ | additive steering only |
| `full_logits` | ✓ | ✗ vocab-parallel only | ✓ |
| `pytorch_fn_local` | ✓ | ✗ | ✗ |

**Materialization (generation).** A continuation read's cost is not the decode,
it is the vocabulary: at batch 32 and 16 steps, every step's distribution over a
128k vocabulary is ~260 MB in fp32, one step is ~16 MB, a site's activations
~8 MB, the token ids ~2 KB. The planner therefore derives, per group, the decode
depth and — per continuation read — whether anything downstream consumes a
distribution: the read is saved, or a metric in the `distribution` domain
reduces it (sec. 2.10). An `ids`-domain metric does **not** count, which is the
point of the domain: a text probe — `decode` over a continuation read, nothing
saved — obliges no vocabulary projection at all, and a backend **must not**
build one where the answer is no.

*How* it complies is its own business: keeping only the addressed steps,
projecting a narrower slice (`logits_to_keep` takes an index tensor), replaying
the sequence teacher-forced, or a vocab-parallel reduction. The reference
backend keeps `ln_final` activations across steps and projects through the head
only at the addressed positions, which needs no second pass — an implementation
note, not a requirement. `explain` prints the obligation so the bill is legible
before a run.

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
    "counterfactual": {"dataset": "ioi/test", "field": "counterfactual_inputs[0]"}
  },
  "sites": {
    "sender":   {"component": "attention_value",  "layer": 9, "head": 9},
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
