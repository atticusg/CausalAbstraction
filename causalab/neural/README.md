# Neural

Foundation layer of the toolkit: the language-model I/O pipeline, a vocabulary
for naming *where* in a transformer to intervene, a vocabulary for naming *what*
to write there, and the engine that runs it. Everything here answers *where* to
intervene, *how* to read/write the feature space, and *how* to run a model with
interventions applied — nothing about specific hypotheses or featurizer
implementations.

Depended on by `io/`, `methods/`, and `analyses/`; depends on neither `tasks/`
nor `causal/`. **Must not import from `methods/`** (concrete featurizers,
training loops, and composed methods live there) — see `docs/CODEBASE.md §3` for
the layering invariants.

Backbone: **[nnsight](https://nnsight.net) 0.8**. An intervention is a block of
ordinary torch code interleaved with the model's forward — read a module's
`.output`, change it, write it back — so causalab does not build or own an
intervention *object*. See `activations/engine.py` for the execution model.

## Module map

| File | Role | Public API |
|------|------|-----------|
| `pipeline.py` | tokenize / load weights / generate / decode | `Pipeline`, `LMPipeline`, `resolve_device` |
| `units.py` | base location + feature-space spec for an intervention point | `ComponentIndexer`, `AtomicModelUnit`, `InterchangeTarget` |
| `LM_units.py` | LM-component bindings of `AtomicModelUnit` | `ResidualStream`, `AttentionHead`, `AttentionOutput`, `MLP`, `AttentionHeadValue`, `AttentionHeadQuery` |
| `components.py` | component type + layer → a readable/writable `Site` on the envoy tree | `resolve_site`, `Site`, `head_layout`, `component_width`, `device_for_layer` |
| `featurizer.py` | feature spaces: paired featurize/inverse modules | `Featurizer`, `ComposedFeaturizer`, `Identity*Module` |
| `interventions.py` | what to write, in a feature space | `build_intervention`, the seven mode classes |
| `token_positions.py` | declarative token-position specs → resolved token indices | `TokenPosition`, `Template`, `build_token_positions`, `paired_token_position`, `combined_token_position` |
| `__init__.py` | public re-exports | `ResidualStream`, `AttentionHead`, `MLP`, `ComponentIndexer`, `AtomicModelUnit`, `InterchangeTarget`, `Pipeline`, `LMPipeline`, `resolve_device` |

## Core abstractions

- **`Pipeline` / `LMPipeline`** — `LMPipeline` wraps a HuggingFace causal LM.
  It exposes **two handles on one network**: `model`, the raw `nn.Module` (for
  config reads, device moves, and anything installing its own hooks), and
  `nnsight`, the `TransformersModel` envoy the engine traces. `load()` tokenizes
  a batch of traces, applies the chat template, left-pads, and builds
  `offset_mapping`; `generate()` runs plain generation; `format_generation()`
  turns a generate result into the `{sequences, scores, string}` dict scorers
  consume. Pass `load_weights=False` for a meta-device pipeline (full module
  tree and `nnsight.scan()` for shapes, no weights).

  **Do not `copy.deepcopy(pipeline.model)`** — see the class docstring.

- **`AtomicModelUnit` = where + how** — composes a `ComponentIndexer` (the
  *location*: layer, component type, dynamic indices) with a `Featurizer` (the
  *feature space*). `resolve_positions()` gives per-example token positions in
  the padded-batch frame; `head_index()` gives the attention head, if any. Units
  serialize to JSON (+ safetensors for non-trivial featurizers).

- **`InterchangeTarget`** — a nested grouping of `AtomicModelUnit`s. Each group
  shares one counterfactual input, so a target with *N* groups needs *N*
  counterfactuals per example. List-like (`__getitem__`/`__iter__`/`__len__`,
  `flatten()`, `nest_to_match()`) and fans `save`/`load`/featurizer ops out to
  its units.

- **`Site`** (`components.py`) — a hookable value plus how to shape it. A
  component type (`block_output`, `mlp_activation`, `head_value_output`, …) plus
  a layer resolves to a `Site` whose `read()` returns the activation and whose
  `write()` puts one back, both valid only inside a trace. Per-head width comes
  from `config.head_dim` and k/v heads are addressed in KV space, so
  grouped-query attention and decoupled-`head_dim` models (Qwen3) are handled
  without remapping by the caller.

- **`Featurizer`** — paired featurize/inverse modules with `is_trivial()`
  (identity) and `>>` composition (`ComposedFeaturizer`, which tracks per-stage
  reconstruction error). Concrete subclasses (SAE, subspace, …) live in
  `methods/` and are restored via lazy subclass dispatch on load.

- **`build_intervention(featurizer, mode)`** (`interventions.py`) — the seven
  modes as `nn.Module`s: `interchange`, `collect`, `add` (steering), `replace`,
  `noise`, `interpolation`, `mask`. Every mode preserves the base's
  reconstruction error, so intervening in a rank-k subspace leaves the
  orthogonal complement untouched. `mask` owns learnable parameters and `noise`
  owns an RNG, so both must be **built once per run** and reused across batches.

- **`TokenPosition` / `Template`** — `Template` fills `{variable}` placeholders
  and maps character ranges → token indices via `offset_mapping` (results cached
  by tokenizer/text). `TokenPosition` (a `ComponentIndexer`) resolves indices
  dynamically against a pipeline. Build from declarative specs with
  `build_token_positions`; combine with `paired_token_position` (different
  positions for base vs. counterfactual) and `combined_token_position` (union).

## `activations/` — running interventions

| File | Role |
|------|------|
| `engine.py` | the execution core — `build_plans`, `build_interventions`, `collect_unit_activations`, `collect_unit_activations_under`, `generate_with_interventions`, `forward_with_interventions`, and the position primitives `FlatIndex` / `flat_index` / `align_per_example` / `gather_positions` / `scatter_positions` |
| `targets.py` | build `InterchangeTarget`s per component type — `build_residual_stream_targets`, `build_attention_head_targets`, `build_attention_output_targets`, `build_mlp_targets` — plus `detect_component_type_from_targets`, `extract_grid_dimensions_from_targets` |
| `collect.py` | collect activations / features — `collect_features`, `collect_source_representations`, `collect_class_centroids` |
| `interchange_mode.py` | activation swapping base↔counterfactual — `prepare_interchange_batch`, `collect_group_sources`, `batched_interchange_intervention`, `run_interchange_interventions`, `run_two_pass_path_patching` |
| `interpolate.py` | arbitrary featurized interpolation `fn(f_base, f_src, **params)` — `run_interpolation_interventions`, `sweep_interpolation_interventions` |
| `data_utils.py` | shared output post-processing — `convert_to_top_k`, `move_outputs_to_cpu` |

### How the engine runs

A run is **two passes**: read the sources in their own forward, then write them
into the base's run. Cross-model patching (sources from a *different* model) and
steering (no source pass at all) fall out of the same path. The one-trace
alternative — base and sources as batched invokes handing values across with
`tracer.barrier` — works, but needs sites visited in forward order with *two*
barrier rounds each or a later source pushes the model past a site the base has
not written yet.

Two ordering rules follow from the interleaver serving each module location
once, in forward order:

1. Sites are visited in `(layer, component rank)` order.
2. Units sharing a site are applied together in one read/modify/write — reading
   the same location twice raises `OutOfOrderError`.

Both are handled inside the engine; callers pass units in any order and get
results back in the order they asked.

There is **no intervention model to build or delete**. An intervention exists
only for the duration of a trace, so there is no per-batch hook construction,
no teardown, and no device bookkeeping for index tensors.

## Data flow

The full external surface, in the order a consumer (`io/`, `methods/`,
`analyses/`) uses it.

```
LMPipeline(model)                     # 0  load model + tokenizer
  → build_token_positions(...)        # 2  where: token indices
  → ResidualStream/AttentionHead(...) # 2  where: component + layer
  → build_*_targets(...)              # 2  group units into InterchangeTarget(s)
  → unit.set_featurizer(Featurizer)   # 3  how: feature space (identity by default)
  → run_interchange_interventions(    # 4  run the mode
        pipeline, dataset, target)
  → convert_to_top_k / move_*_to_cpu  # 5  post-process outputs
```

### 0. Device & model setup

- `resolve_device(device=None)` — auto-pick `cuda` → `mps` → `cpu`.
- `LMPipeline(model_or_name, max_new_tokens=3, max_length=None, logit_labels=False,
  position_ids=False, use_chat_template=False, chat_answer_directive=None,
  padding_side="left", load_weights=True)` — wrap a HF causal LM (or a name to
  load). Extra kwargs: `device`, `dtype`, `device_map`, `config`, `hf_token`,
  `eager_attn`, `tokenizer` (needed for a model built in-process, which has no
  `config.name_or_path`).

### 1. Run the model (no interventions)

- `pipeline.load(raw_input)` → encodings dict (`input_ids`, `attention_mask`,
  `offset_mapping`, optional `position_ids`); applies the chat template,
  left-pads, and records char→token offsets used by token positions.
- `pipeline.generate(prompt)` — plain generation; `pipeline.dump(output)` decodes
  logits/ids → string(s); `pipeline.compute_outputs(...)` batches base +
  counterfactual inputs.
- Introspection for grid building: `get_num_layers()`, `get_num_attention_heads()`.

### 2. Specify *where* to intervene

- **Token positions** (`token_positions.py`): `Template` fills `{variable}`
  placeholders and maps character ranges → token indices via `offset_mapping`;
  `TokenPosition` resolves those indices dynamically against the pipeline. Build
  with `build_token_positions` / `build_token_position_factories` from
  declarative specs (absolute/scoped/relative indices, a named variable's
  tokens, or a callable); combine with `paired_token_position` (base vs.
  counterfactual) and `combined_token_position` (union).
- **Component units** (`LM_units.py`): `ResidualStream`, `MLP`, `AttentionOutput`,
  `AttentionHead`, `AttentionHeadValue`, `AttentionHeadQuery` — each an
  `AtomicModelUnit` binding `(layer, [head], token positions)` to a component
  type, with the residual/MLP `target_output` and MLP `location`
  (`mlp_input`/`mlp_output`/`mlp_activation`) variants.
- **Grouping** (`targets.py`): `build_residual_stream_targets`,
  `build_attention_head_targets`, `build_attention_output_targets`,
  `build_mlp_targets` produce dicts of `InterchangeTarget`, with grouping modes
  (`one_target_all_units` / `one_target_per_unit` / `one_target_per_layer`).
  Inspect existing targets with `detect_component_type_from_targets` and
  `extract_grid_dimensions_from_targets`.

### 3. Specify *how* — the feature space

- `Featurizer` (identity by default; `is_trivial()` reports it) attaches to a unit
  via `set_featurizer`; chain with `>>` into a `ComposedFeaturizer` that preserves
  per-stage reconstruction error. Concrete featurizers (SAE, subspace, …) come
  from `methods/` and are restored on load by lazy subclass dispatch.

### 4. Run a mode

- **Collect** — `collect_features(dataset, pipeline, model_units, batch_size=32,
  collect_output_logits=False)` → `{unit_id: (n_samples, n_features)}`;
  `collect_class_centroids(...)` for per-class means;
  `collect_source_representations(...)` to gather a source model's activations.
- **Interchange** — `run_interchange_interventions(pipeline,
  counterfactual_dataset, interchange_target, batch_size=32, output_scores=True,
  source_pipeline=None)` swaps base↔counterfactual activations.
  `run_two_pass_path_patching(...)` does receiver-set path patching (collect
  under a sender+restorer interchange, then inject on the clean base).
- **Steering / replacement / noise** — `methods/steer/steer.py`'s
  `run_steering_interventions(...)`, with `type_by_unit` to mix modes in one pass.
- **Interpolation** — `run_interpolation_interventions(...)` applies
  `fn(f_base, f_src, **params)` in feature space; `sweep_interpolation_interventions(...)`
  loops `(featurizer, fn, params)` configs.
- **Cross-model patching** — pass `source_pipeline=` to read sources from a
  different model.

Within an example, the base and counterfactual must select the **same number of
tokens** — interchange pairs the base's *i*-th position with the source's *i*-th,
so unequal counts have no pairing. `prepare_interchange_batch` rejects a mismatch
up front with an actionable message.

Widths may differ *between* examples: positions are indexed as flat
`(example, position)` pairs, so a value that tokenizes to two tokens for one
example and one for another is fine. Selecting *nothing* for an example is
refused — that example would be left un-intervened and then scored beside the
rest, which is a wrong number rather than a crash.

### 5. Post-process outputs

- `convert_to_top_k(...)` extracts top-k logits/indices/tokens from full-vocab
  scores (memory-efficient); `move_outputs_to_cpu(...)` detaches and moves nested
  output tensors to CPU.

## Persistence

`AtomicModelUnit` / `InterchangeTarget` / `Featurizer` round-trip to JSON
(metadata) plus safetensors (non-trivial featurizer weights); `InterchangeTarget`
fans `save`/`load` to its units, and LM units reload via `load_modules()`.
