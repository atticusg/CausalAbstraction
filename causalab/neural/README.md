# Neural

Foundation layer of the toolkit: a thin surface over **pyvene** (intervention
modes, the `IntervenableModel` wrapper, the base `Featurizer` interface) plus the
language-model I/O pipeline. Everything here answers *where* to intervene, *how*
to read/write the feature space, and *how* to run a model with interventions
applied — nothing about specific hypotheses or featurizer implementations.

Depended on by `io/`, `methods/`, and `analyses/`; depends on neither `tasks/`
nor `causal/`. **Must not import from `methods/`** (concrete featurizers,
training loops, and composed methods live there) — see `docs/CODEBASE.md §3` for
the layering invariants.

Backbone: **pyvene** (pinned to a git build). `docs/PATH_PATCHING.md` documents
the targeted 0.1.8 component surface, pyvene↔causalab mappings, and known gaps.

## Module map

| File | Role | Public API |
|------|------|-----------|
| `pipeline.py` | tokenize / load weights / generate / dispatch interventions to pyvene | `Pipeline`, `LMPipeline`, `resolve_device` |
| `units.py` | base location + feature-space spec for an intervention point | `ComponentIndexer`, `AtomicModelUnit`, `InterchangeTarget` |
| `LM_units.py` | LM-component bindings of `AtomicModelUnit` | `ResidualStream`, `AttentionHead`, `AttentionOutput`, `MLP`, `AttentionHeadValue`, `AttentionHeadQuery` |
| `featurizer.py` | feature-space transforms + per-mode intervention builders | `Featurizer`, `ComposedFeaturizer`, `Identity*Module`, `build_feature_*_intervention` (7 modes) |
| `token_positions.py` | declarative token-position specs → resolved token indices | `TokenPosition`, `Template`, `build_token_positions`, `build_token_position_factories`, `paired_token_position`, `combined_token_position` |
| `__init__.py` | public re-exports | `ResidualStream`, `AttentionHead`, `MLP`, `ComponentIndexer`, `AtomicModelUnit`, `InterchangeTarget`, `Pipeline`, `LMPipeline`, `resolve_device` |

## Core abstractions

- **`Pipeline` / `LMPipeline`** — `LMPipeline` wraps a HuggingFace causal LM.
  `load()` tokenizes a batch of traces, applies the chat template, left-pads, and
  builds `offset_mapping` + `position_ids`; `generate()` runs plain generation;
  `intervenable_generate()` runs generation through pyvene with intervention
  configs (and `source_representations` for cross-model patching). Pass
  `load_weights=False` for a config-only pipeline (shapes without weights).

- **`AtomicModelUnit` = where + how** — composes a `ComponentIndexer` (the
  *location*: layer, component type, dynamic indices) with a `Featurizer` (the
  *feature space*). `create_intervention_config()` emits the pyvene config;
  `index_component()` resolves indices with attention-mask padding handling;
  units serialize to JSON (+ safetensors for non-trivial featurizers).

- **`InterchangeTarget`** — a nested grouping of `AtomicModelUnit`s that share one
  counterfactual input. List-like (`__getitem__`/`__iter__`/`__len__`,
  `flatten()`, `nest_to_match()`) and fans `save`/`load`/featurizer ops out to its
  units.

- **`Featurizer`** — paired featurize/inverse modules with `is_trivial()`
  (identity) and `>>` composition (`ComposedFeaturizer`, which tracks per-stage
  reconstruction error). Provides the per-mode intervention builders
  (`build_feature_{interchange,collect,steering,replace,interpolation,noise,mask}_intervention`).
  Concrete subclasses (SAE, subspace, …) live in `methods/` and are restored via
  lazy subclass dispatch on load.

- **`TokenPosition` / `Template`** — `Template` fills `{variable}` placeholders and
  maps character ranges → token indices via `offset_mapping` (results cached by
  tokenizer/text). `TokenPosition` (a `ComponentIndexer`) resolves indices
  dynamically against a pipeline. Build from declarative specs with
  `build_token_positions`; combine with `paired_token_position` (different
  positions for base vs. counterfactual) and `combined_token_position` (union).

## `activations/` — running interventions

The subpackage that executes the modes that `units` + `targets` configure.

| File | Role |
|------|------|
| `intervenable_model.py` | factory + lifecycle for the pyvene `IntervenableModel` — `prepare_intervenable_model`, `prepare_mixed_intervenable_model` (interchange+collect, for path patching), `delete_intervenable_model`, `device_for_layer`. Modes: interchange / collect / mask / add / replace / interpolation / noise |
| `targets.py` | build `InterchangeTarget`s per component type — `build_residual_stream_targets`, `build_attention_head_targets`, `build_attention_output_targets`, `build_mlp_targets` — plus `detect_component_type_from_targets`, `extract_grid_dimensions_from_targets` |
| `collect.py` | collect activations / features — `collect_features`, `collect_source_representations`, `collect_class_centroids` |
| `interchange_mode.py` | activation swapping base↔counterfactual — `prepare_intervenable_inputs`, `run_interchange_interventions`, `run_two_pass_path_patching` |
| `interpolate.py` | arbitrary featurized interpolation `fn(f_base, f_src, **params)` — `run_interpolation_interventions`, `sweep_interpolation_interventions` |
| `data_utils.py` | shared output post-processing — `convert_to_top_k`, `move_outputs_to_cpu` |

Re-exported from `activations/__init__.py`: `collect_source_representations`,
`prepare_intervenable_model`, `delete_intervenable_model`, the four
`build_*_targets`, `detect_component_type_from_targets`,
`extract_grid_dimensions_from_targets`.

## Data flow

The full external surface, in the order a consumer (`io/`, `methods/`,
`analyses/`) uses it. Each stage names the public functions/classes it provides.

Canonical sequence:

```
LMPipeline(model)                     # 0  load model + tokenizer
  → build_token_positions(...)        # 2  where: token indices
  → ResidualStream/AttentionHead(...) # 2  where: component + layer
  → build_*_targets(...)              # 2  group units into InterchangeTarget(s)
  → unit.set_featurizer(Featurizer)   # 3  how: feature space (identity by default)
  → run_interchange_interventions(    # 4+5 build IntervenableModel, run the mode
        pipeline, dataset, target)    #     dispatches to pipeline.intervenable_generate
  → convert_to_top_k / move_*_to_cpu  # 6  post-process outputs
```

### 0. Device & model setup

- `resolve_device(device=None)` — auto-pick `cuda` → `mps` → `cpu`.
- `LMPipeline(model_or_name, max_new_tokens=3, max_length=None, logit_labels=False,
  position_ids=False, use_chat_template=False, chat_answer_directive=None,
  padding_side="left", load_weights=True)` — wrap a HF causal LM (or a name to
  load). `load_weights=False` gives a config-only pipeline (shapes without
  weights, for cheap unit/target construction).

### 1. Run the model (no interventions)

- `pipeline.load(raw_input)` → encodings dict (`input_ids`, `attention_mask`,
  `offset_mapping`, `position_ids`); applies the chat template, left-pads, and
  records char→token offsets used by token positions.
- `pipeline.generate(prompt)` — plain generation; `pipeline.dump(output)` decodes
  logits/ids → string(s); `pipeline.compute_outputs(...)` batches base +
  counterfactual inputs.
- Introspection for grid building: `get_num_layers()`, `get_num_attention_heads()`.

### 2. Specify *where* to intervene

- **Token positions** (`token_positions.py`): `Template` fills `{variable}`
  placeholders and maps character ranges → token indices via `offset_mapping`;
  `TokenPosition` resolves those indices dynamically against the pipeline. Build
  with `build_token_positions` / `build_token_position_factories` from declarative
  specs (absolute/scoped/relative indices, a named variable's tokens, or a
  callable); combine with `paired_token_position` (base vs. counterfactual) and
  `combined_token_position` (union).
- **Component units** (`LM_units.py`): `ResidualStream`, `MLP`, `AttentionOutput`,
  `AttentionHead`, `AttentionHeadValue`, `AttentionHeadQuery` — each an
  `AtomicModelUnit` binding `(layer, [head], token positions)` to a pyvene
  component type, with the residual/MLP `target_output` and MLP `location`
  (`mlp_input`/`mlp_output`/`mlp_activation`) variants.
- **Grouping** (`targets.py`): `build_residual_stream_targets`,
  `build_attention_head_targets`, `build_attention_output_targets`,
  `build_mlp_targets` produce dicts of `InterchangeTarget`, with grouping modes
  (`one_target_all_units` / `one_target_per_unit` / `one_target_per_layer`).
  Inspect existing targets with `detect_component_type_from_targets` and
  `extract_grid_dimensions_from_targets` (recover the layer/head/position axes).

### 3. Specify *how* — the feature space

- `Featurizer` (identity by default; `is_trivial()` reports it) attaches to a unit
  via `set_featurizer`; chain with `>>` into a `ComposedFeaturizer` that preserves
  per-stage reconstruction error. Concrete featurizers (SAE, subspace, …) come
  from `methods/` and are restored on load by lazy subclass dispatch.
- The featurizer supplies the pyvene intervention class for each mode:
  `build_feature_{interchange,collect,steering,replace,interpolation,noise,mask}_intervention`.

### 4. Build the intervenable model

- `prepare_intervenable_model(pipeline, units_or_targets, intervention_type)` →
  pyvene `IntervenableModel`; flat unit lists are auto-wrapped into a single-group
  `InterchangeTarget`, and static indices enable pyvene's `use_fast` path. Modes:
  `interchange` / `collect` / `mask` / `add` / `replace` / `interpolation` /
  `noise`.
- `prepare_mixed_intervenable_model(...)` mixes interchange + collect in one model
  (path-patching PASS 1), enforcing the collect-order contract.
- Device handling: `device_for_layer(pipeline, layer)` resolves the right GPU for a
  layer on sharded models. Always release with `delete_intervenable_model(...)`
  (moves to CPU, deletes, `gc.collect()`, clears the CUDA cache).

### 5. Run a mode (`activations/`)

- **Collect** — `collect_features(dataset, pipeline, model_units, batch_size=32,
  collect_output_logits=False)` → `{unit_id: (n_samples, n_features)}`;
  `collect_class_centroids(...)` for per-class means;
  `collect_source_representations(...)` to gather a source model's activations.
- **Interchange** — `prepare_intervenable_inputs(...)` tokenizes + indexes + checks
  shape/raggedness, then `run_interchange_interventions(pipeline,
  counterfactual_dataset, interchange_target, batch_size=32, output_scores=True,
  source_pipeline=None)` swaps base↔counterfactual activations.
  `run_two_pass_path_patching(...)` does receiver-set path patching (collect under
  a sender+restorer interchange, then inject on the clean base).
- **Interpolation** — `run_interpolation_interventions(...)` applies
  `fn(f_base, f_src, **params)` (canonically `(1-α)·f_base + α·f_src`) in feature
  space; `sweep_interpolation_interventions(...)` loops `(featurizer, fn, params)`
  configs.
- **Cross-model patching** — pass `source_pipeline=` (interchange) or
  `source_representations` (via `intervenable_generate`) to patch activations
  collected from a different model.

### 6. Post-process outputs

- `convert_to_top_k(...)` extracts top-k logits/indices/tokens from full-vocab
  scores (memory-efficient); `move_outputs_to_cpu(...)` detaches and moves nested
  output tensors to CPU.

### Persistence

`AtomicModelUnit` / `InterchangeTarget` / `Featurizer` round-trip to JSON
(metadata) plus safetensors (non-trivial featurizer weights); `InterchangeTarget`
fans `save`/`load` to its units, and LM units reload via `load_modules()`.
