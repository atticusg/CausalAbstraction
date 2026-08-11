# Rebasing causalab on nnterp

Six questions, one doc:

1. **Could the `causalab/neural` backbone move off pyvene?** Part 1 decouples *what
   `causalab/neural` provides to the rest of the toolkit* (the durable
   desiderata) from *how it is implemented on **pyvene*** today, and maps each
   capability to the **nnsight / nnterp** primitive that would serve it instead.
2. **What does nnterp offer beyond that?** Part 2 surveys nnterp functionality
   *above* the low-level intervention plumbing — analysis methods, a model-load
   validation framework, cross-architecture onboarding, prompt/target-token
   utilities, visualization — that causalab could adopt wholesale, independently
   of (or alongside) a backbone swap.
3. **How should the intervention layer actually be built on nnsight, and what's
   left for causalab to own?** Part 3 plans the declarative→imperative rewrite at
   the design level — a single read-modify-write primitive, the collect∘intervene
   fusion pyvene can't express, the proposed abstraction stack, and causalab's
   specific role on top of nnterp.
4. **How do we ship it?** Part 4 breaks the Part 3 design into specific
   sub-issues, maps their dependencies, and groups them into waves that can be
   built in parallel.
5. **How do we consolidate what the rebase built?** Part 5 (added post-rebase;
   umbrella #480) designs the **engine unification**: the three lowerings plus
   the dataset-engine bypass collapse into ONE scheduler + ONE executor (plain
   sequential traces, no `model.session()`) + TWO trace emitters, generation
   staging becomes a terminal stage kind, and the three divergent generation
   output shapes unify into one `GenerationResult`.
6. **How does the declarative surface catch up to the engine?** Part 6
   (umbrella #491, successor to Part 5) designs the **where-unification**:
   the legacy `AtomicModelUnit`/`InterchangeTarget` surface — still what
   analyses, tasks, and the `activations/` wrappers speak — retires in favor
   of `Site`/`FeaturizedSite`/`Edit`/`Plan`, with its serialization, position
   binding, grouping, and grid-builder affordances re-homed on the engine
   vocabulary.

## Contents

- [Conventions and provenance](#conventions-and-provenance)
- [Part 1: neural functionality mapping (pyvene vs nnterp)](#part-1-neural-functionality-mapping-pyvene-vs-nnterp)
  - [1. Model and pipeline I/O](#1-model-and-pipeline-io)
  - [2. Intervention sites: components](#2-intervention-sites-components)
  - [3. Intervention sites: token positions and grouping](#3-intervention-sites-token-positions-and-grouping)
  - [4. Intervention types](#4-intervention-types)
  - [5. Run interventions at scale](#5-run-interventions-at-scale)
  - [6. Cross-model patching](#6-cross-model-patching)
  - [7. Featurizer](#7-featurizer)
  - [8. Output post-processing](#8-output-post-processing)
  - [9. Persistence](#9-persistence)
- [Part 2: nnterp capabilities beyond neural](#part-2-nnterp-capabilities-beyond-neural)
  - [Cross-architecture standardization and onboarding](#cross-architecture-standardization-and-onboarding)
  - [Model-load validation framework](#model-load-validation-framework)
  - [Analysis methods (logit lens, patchscope)](#analysis-methods-logit-lens-patchscope)
  - [Target-token prompt utilities](#target-token-prompt-utilities)
  - [Activation collection and grafting helpers](#activation-collection-and-grafting-helpers)
  - [Visualization](#visualization)
  - [Remote execution (NDIF)](#remote-execution-ndif)
- [Part 3: implementing the intervention layer on nnsight](#part-3-implementing-the-intervention-layer-on-nnsight)
  - [The nnsight execution model, briefly](#the-nnsight-execution-model-briefly)
  - [One primitive: the site edit (read-modify-write)](#one-primitive-the-site-edit-read-modify-write)
  - [Fusing collect and intervene in one forward pass](#fusing-collect-and-intervene-in-one-forward-pass)
  - [Proposed abstraction stack](#proposed-abstraction-stack)
  - [What causalab still owns](#what-causalab-still-owns)
  - [Open design questions](#open-design-questions)
- [Part 4: work breakdown into parallel waves](#part-4-work-breakdown-into-parallel-waves)
  - [Sub-issues](#sub-issues)
  - [Waves](#waves)
  - [Critical path and parallelism](#critical-path-and-parallelism)
  - [Orchestration: how we build it](#orchestration-how-we-build-it)
- [Part 5: engine unification (one scheduler, one executor, two emitters — no session)](#part-5-engine-unification-one-scheduler-one-executor-two-emitters--no-session)
  - [Why: three lowerings plus a bypass](#why-three-lowerings-plus-a-bypass)
  - [Target architecture](#target-architecture)
  - [No-session policy](#no-session-policy)
  - [Design](#design)
  - [Refusal relocation map](#refusal-relocation-map)
  - [Unification sub-issues (EU0–EU7)](#unification-sub-issues-eu0eu7)
  - [Legacy boundary (successor: #491)](#legacy-boundary-successor-491)
- [Part 6: where-unification (one declarative surface over the engine)](#part-6-where-unification-one-declarative-surface-over-the-engine)
  - [Why: two parallel "where" surfaces](#why-two-parallel-where-surfaces)
  - [What the legacy surface carries](#what-the-legacy-surface-carries)
  - [Target vocabulary: two frozen specs over the engine](#target-vocabulary-two-frozen-specs-over-the-engine)
  - [Serialization: named specs round-trip; callables do not](#serialization-named-specs-round-trip-callables-do-not)
  - [Position binding](#position-binding)
  - [Grouping: Plan-input naming is the contract](#grouping-plan-input-naming-is-the-contract)
  - [Grid builders (the targets.py successor)](#grid-builders-the-targetspy-successor)
  - [Migration policy](#migration-policy)
  - [Where-unification sub-issues (WU1–WU6)](#where-unification-sub-issues-wu1wu6)
- [Cross-cutting migration notes](#cross-cutting-migration-notes)
- [Sources](#sources)

---

## Conventions and provenance

**Provenance / confidence**

- **pyvene column** — code-verified against the causalab source
  (`activations/intervenable_model.py`, `featurizer.py`, `units.py`,
  `LM_units.py`, `pipeline.py`) and `docs/PATH_PATCHING.md`; primitive names and
  component strings are quoted from those files.
- **nnterp rows / Part 2** — **code-verified** against the local nnterp checkout
  `~/nnterp` @ `7dbe4da` (~v1.3.0). Confirmed against source: the
  `StandardizedTransformer` constructor; the settable
  `layers_/attentions_/mlps_{input,output}[i]` accessors
  (`LayerAccessor.__setitem__`, `rename_utils.py:358`);
  `num_layers/num_heads/hidden_size/vocab_size`; `steer(...)`
  (`standardized_transformer.py:282`); `attention_probabilities[i]` (gated by
  `enable_attention_probs`); the `logit_lens`/`patchscope_lens` helpers; the
  rename scheme; and load-time validation + `python -m nnterp run_tests`. The
  **absence** of per-head value/query accessors and of any
  ablation/gradient/patching/cross-model helper was also confirmed in source.
- **Raw-nnsight rows** (the fallbacks listed below the nnterp option) — derived
  from [`docs/NNsight_overview.md`](NNsight_overview.md), a reference compiled
  from official sources and *not* code-verified locally; validate against the
  installed `nnsight` before relying on them.

**Legend (Gaps / notes column, Part 1)**

- **✓** direct equivalent — a primitive maps cleanly.
- **◑** achievable but **hand-rolled** — no off-the-shelf primitive; causalab
  would write the trace logic itself.
- **✗** no direct primitive — a real gap; pyvene offers something nnsight/nnterp
  does not expose.
- **▣** **backbone-agnostic** — already causalab-side code that calls neither
  backbone's intervention machinery; carries over unchanged.

---

## Part 1: neural functionality mapping (pyvene vs nnterp)

The "Functionality" column is the durable contract — it should survive any
backbone change. The two primitive columns are implementations of it. Section
order follows the data-flow walkthrough in
[`causalab/neural/README.md`](../causalab/neural/README.md).

The headline architectural delta: **pyvene is declarative** (build an
`IntervenableConfig`, then run it), while **nnsight is imperative** (write a
`with model.trace():` body that manipulates real tensors in forward-pass order).
Most ◑ rows are this delta — a config that causalab *constructs* becomes trace
code that causalab *emits*.

### 1. Model and pipeline I/O

| Functionality | pyvene primitive | nnterp / nnsight primitive | Gaps / notes |
|---|---|---|---|
| Load model + tokenizer | HF model handed to `pv.IntervenableModel(model=...)`; load is causalab's `LMPipeline` | `nnterp.StandardizedTransformer(name)` / `nnsight.LanguageModel(name, device_map="auto", dispatch=True)` — wraps the original HF model in place | ✓ both wrap HF without reimplementing |
| Config-only load (`load_weights=False`) | causalab-side (build units/targets without weights) | `model.scan(input)` resolves shapes via `FakeTensorMode` (no compute); remote loads as meta tensors | ◑ different mechanism (shape pass vs. weightless construct) |
| Tokenize traces: chat template, left-pad, `offset_mapping` | causalab-side (HF tokenizer) | causalab-side (HF tokenizer) | ▣ reusable as-is |
| `position_ids` for absolute-pos models | causalab `ensure_position_ids` / `left_pad_position_ids` | causalab-side; passed into `model.trace(...)` inputs | ▣ reusable |
| Plain generation | causalab `pipeline.generate` (HF `generate`) | `model.generate(input, max_new_tokens=N)` | ✓ |
| Decode logits/ids → strings | causalab `pipeline.dump` (tokenizer) | causalab-side | ▣ reusable |
| Model introspection (#layers, #heads) | causalab `get_num_layers` / `get_num_attention_heads` (from config) | nnterp `model.num_layers` / `num_heads` / `hidden_size` / `vocab_size` | ✓ nnterp exposes directly |
| Device resolution | causalab `resolve_device` (cuda→mps→cpu) | `device_map="auto"` (HF/accelerate) | ✓ |

### 2. Intervention sites: components

pyvene addresses sites by **component-location string**; nnterp addresses them by
**uniform accessor** (`layers_output[i]`, etc.) and intervenes by **assignment**.

| Functionality | pyvene primitive | nnterp / nnsight primitive | Gaps / notes |
|---|---|---|---|
| Residual stream @ (layer, pos), block **input** | `"block_input"` (`LM_units.py:49`) | nnterp `layers_input[i]` | ✓ |
| Residual stream, block **output** | `"block_output"` | nnterp `layers_output[i]` (the tuple's `[0]` in raw nnsight) | ✓ |
| Embeddings (layer −1 special case) | `block_input` at layer 0 | nnterp `token_embeddings` / `embed_tokens` | ✓ |
| MLP output | `"mlp_output"` (`LM_units.py:133`) | nnterp `mlps_output[i]` | ✓ |
| MLP input | `"mlp_input"` | nnterp `mlps_input[i]` | ✓ |
| MLP internal activation | `"mlp_activation"` | raw nnsight submodule path or `module.source.<op>.output`; **no named nnterp accessor** | ◑ manual, architecture-specific |
| Attention sublayer output (all heads) | `"attention_output"` (`LM_units.py:250`) | nnterp `attentions_output[i]` | ✓ |
| Attention **head value** vector @ (layer, head, pos) | `"head_value_output"` — GQA-aware (`head // (n_head/n_kv_head)`); decoupled `head_dim` unsupported (`PATH_PATCHING.md §0.5`) | raw nnsight: reshape `self_attn` value tensor `(b, heads, seq, d_head)` + manual GQA remap; pre-projection vectors via `.source` | ✗ **confirmed**: nnterp 1.3.0 has no per-head value accessor (only whole-sublayer `attentions_output[i]`); pyvene's `head_value_output` is a real convenience nnsight/nnterp lack |
| Attention **head query** vector | `"head_query_output"` (`LM_units.py:571`) | raw nnsight reshape of the query projection; `.source` for pre-attention vectors | ◑/✗ manual; no named primitive |
| Attention-weighted head output (path-patching *sender*) | `"head_attention_value_output"` (`LM_units.py:329`) | raw nnsight per-head slice on attn output | ◑ manual |
| Attention probabilities / per-head pattern | **shipped (CAP4, #457)**: `causalab/neural/attention_probs.py` — `AttentionProbabilitiesSite` + `knockout`/`renormalize` modes; gated by `LMPipeline(..., enable_attention_probs=True)` | nnterp `attention_probabilities[i]` (`enable_attention_probs=True`); raw nnsight via `.source` | ✓ nnterp / ◑ raw — "not universal" across families |

### 3. Intervention sites: token positions and grouping

| Functionality | pyvene primitive | nnterp / nnsight primitive | Gaps / notes |
|---|---|---|---|
| Token-position resolution (templates, char→token via `offset_mapping`, dynamic indexers) | causalab-side (`token_positions.py`); indices feed pyvene `unit_locations` | causalab-side; indices become in-trace tensor slices `[:, idx, :]` | ▣ resolution reusable; only the *consumption* differs |
| Static vs. dynamic indices | `unit_locations` + `use_fast=True` static-index fast path (`intervenable_model.py:59-66`) | plain tensor indexing per trace — no static/dynamic distinction | ◑ `use_fast` optimization has no analog (and is unneeded) |
| Padding-frame shift (unpadded→padded) | causalab `_shift_row` via `attention_mask` (`units.py:39-57`) | causalab-side; consumes left-padded tensors | ▣ reusable |
| Group units sharing one counterfactual (`InterchangeTarget`) | config groups keyed by `group_key` (`intervenable_model.py`) | each source = a `tracer.invoke`; "grouping" = which invoke a write reads from | ◑ no target object; expressed as trace structure |
| Subspace / feature selection | `subspaces` + precomputed `subspace_partition` (`pipeline.py:617`, `featurizer.py:516`) | index the feature dim in-trace after featurization | ◑ manual indexing |
| Inspect targets (component type, grid axes) | causalab `detect_component_type_from_targets`, `extract_grid_dimensions_from_targets` | causalab-side | ▣ reusable |

### 4. Intervention types

Each pyvene type is a dynamically-built subclass of a pyvene intervention base,
produced by a `build_feature_*_intervention` factory whose closure captures the
featurizer (`featurizer.py`). In nnsight the "type" is **just trace code**, so the
subclass machinery disappears.

| Functionality | pyvene primitive | nnterp / nnsight primitive | Gaps / notes |
|---|---|---|---|
| **Interchange** (swap base↔source) | `pv.TrainableIntervention` + `pv.DistributedRepresentationIntervention`; `_do_intervention_by_swap(..., "interchange", ...)` (`featurizer.py:485-524`) | two `tracer.invoke`s + `tracer.barrier(2)`: capture from clean, assign into corrupt (`NNsight_overview.md §3.3`) | ◑ canonical recipe; nnterp keeps stable names |
| **Collect** (read activations) | `pv.CollectIntervention` (`featurizer.py:589-617`) | `nnsight.save(...)` on the site read (hand-rolled per-site saves) | ✓ — but **do not use `tracer.cache`**: broken under nnterp renaming (nnterp's own suite skips it — "Cache is not supported yet due to a nnsight renaming issue", `nnterp/tests/test_nnsight_utils.py`) |
| **Steering** (add a direction) | `pv.TrainableIntervention` add; `get_steering_intervention()` (`featurizer.py:646-674`) | nnterp `model.steer(layers, steering_vector, factor, positions, token_positions, batch_index)` → `layers_output += factor·v` (`standardized_transformer.py:282`) | ✓ off-the-shelf nnterp helper |
| **Replace** (overwrite with source) | `get_replace_intervention()` (`featurizer.py:808-837`) | bare `=` assignment to the accessor | ✓ |
| **Interpolation** (`fn(f_base,f_src,**p)`) | `get_interpolation_intervention()` (`featurizer.py:546-581`) | capture `f_base` & `f_src`, apply the fn, write back | ◑ hand-rolled |
| **Noise** (seeded Gaussian) | `get_noise_intervention(seed)` (`featurizer.py:722-776`) | `h + torch.randn(..., generator=seeded)` written back | ◑ hand-rolled; seed via `torch.Generator` |
| **Mask** (learnable binary DBM gate) | `pv.TrainableIntervention`, trained via gradient (`featurizer.py:858-929`) | edits are differentiable — apply a learnable mask param in-trace, train via `.backward()` | ◑ feasible (nnsight supports grads), no helper |
| Per-position application across a span | pyvene `keep_last_dim=True` keeps the position axis (`featurizer.py:497`) | natural — slicing keeps the seq axis | ✓ implicit in nnsight |

### 5. Run interventions at scale

| Functionality | pyvene primitive | nnterp / nnsight primitive | Gaps / notes |
|---|---|---|---|
| Build the intervenable model | `pv.IntervenableConfig(configs)` → `pv.IntervenableModel(config, model, use_fast)` (`intervenable_model.py:83-86`) | no persistent object — `with model.trace():` per batch, or `model.edit()` for persistent edits | ◑ paradigm shift: config object → trace context |
| Batched / multi-input execution | `intervenable_model.generate(base, sources, unit_locations, subspaces, ...)` (`pipeline.py:647-652`) | `tracer.invoke(input)` per input inside `model.trace()` / `model.generate()` | ◑ |
| Collect features at scale | causalab batch loop + `CollectIntervention` (`collect_features`) | saved lists batched over invokes (+ `tracer.stop()` early-exit after the last touched layer); nnterp `collect_token_activations_batched` | ✓ hand-rolled saves — **`tracer.cache` is unusable under nnterp renaming** (see §4 Collect row) |
| Generation with interventions across tokens | pyvene `generate(..., intervene_on_prompt=True)` | `model.generate(..., max_new_tokens)` + `tracer.iter[:N]` | ✓ ⚠ unbounded `iter[:]` deadlocks — bound it |
| Two-pass path patching (receiver sets) | `prepare_mixed_intervenable_model` (interchange+collect) + `sorted_keys` collect-order contract (`intervenable_model.py:105-160`, `PATH_PATCHING.md §8.3`) | `model.session()` / staged invokes + `tracer.barrier`; ordering is explicit in code | ◑ hand-built — but the `sorted_keys` ordering contract **dissolves** (you control order) |
| Device sharding across GPUs | per-key `intervention.to(device)` from `hf_device_map` (`intervenable_model.py:87-98`) | `device_map="auto"` handled by HF/accelerate for reads/writes | ✓ simpler for reads — but source/steering tensors must move to the site's device (nnterp's `steer` shows the pattern), and **trainable featurizer modules at per-layer devices + a cross-device optimizer stay causalab-owned** (ED3) |
| Lifecycle / teardown | `set_device("cpu")` + `del` + `gc.collect()` + `empty_cache()` (`intervenable_model.py:195-211`) | trace context auto-frees; unsaved values dropped on `with`-exit | ✓ simpler (no manual teardown) |
| Static-index speedup | `use_fast=True` | n/a (plain indexing) | — no analog needed |
| Deferred-execution ordering constraint | n/a (declarative; pyvene orders internally) | must touch modules in **forward-pass order** within an invoke or the worker deadlocks (`OutOfOrderError`) | ⚠ new constraint to respect when emitting traces |

### 6. Cross-model patching

| Functionality | pyvene primitive | nnterp / nnsight primitive | Gaps / notes |
|---|---|---|---|
| Patch activations collected from a **different** model | `source_representations` (pre-collected tensors injected with `sources=None`) + collect-order contract (`pipeline.py:596,609-611`) | load two model objects, capture in the source's trace, pass the tensor into the target's trace assignment | ◑ feasible but fully manual — **no single-call API** (confirmed: nnterp ships no cross-model transfer helper); mixing remote/local complicates. Causalab closes the gap at the Plan layer (PL4): `Plan.models` binds an input to the model it runs on; the staged lowering runs each trace on its input's model, values crossing as saved tensors (local execution only) |

### 7. Featurizer

| Functionality | pyvene primitive | nnterp / nnsight primitive | Gaps / notes |
|---|---|---|---|
| Identity (trivial passthrough) | `Identity*Module` / `is_trivial()` | simply don't transform | ✓ |
| Learned featurize/inverse (SAE, subspace, rotation) | featurizer modules wrapped into a pyvene intervention subclass via closure capture (`featurizer.py:485-524`) | call the featurizer module **inside the trace** as a first-class submodule, read/edit features, write reconstruction back (`NNsight_overview.md §3.9`) | ✓ simpler — featurizer is just a module call, no subclass |
| Per-mode intervention-class generation | required: pyvene instantiates `intervention_type(**kwargs)` with no custom args, so featurizer must be closed over | not required — the mode is trace code | ✓ eliminates the closure/subclass machinery |
| Composition (`>>`) with per-stage error | causalab `ComposedFeaturizer` | causalab-side (chained module calls) | ▣ reusable |
| Subspace / feature-index selection | `subspaces` / `subspace_partition` | in-trace feature-dim indexing | ◑ manual |
| Save / load (lazy subclass dispatch) | causalab-side JSON + safetensors | causalab-side | ▣ reusable |

### 8. Output post-processing

| Functionality | pyvene primitive | nnterp / nnsight primitive | Gaps / notes |
|---|---|---|---|
| Top-k extraction from full-vocab scores | causalab `convert_to_top_k` | causalab-side | ▣ carried over, then superseded (EU5b #487): top-k compression is `pipeline.compress_scores_top_k` on the flat `GenerationResult`; the legacy helper is deleted with its module |
| Detach + move outputs to CPU | causalab `move_outputs_to_cpu` | causalab-side (and a remote best-practice: `.cpu()` before `.save()`) | ▣ carried over, then absorbed by the producers (EU5b #487): every engine emitter returns detached CPU tensors, so the post-pass helper is deleted with its module |

### 9. Persistence

| Functionality | pyvene primitive | nnterp / nnsight primitive | Gaps / notes |
|---|---|---|---|
| Serialize units / targets / featurizers | causalab JSON (metadata) + safetensors (weights); pyvene `IntervenableModel` is **ephemeral**, never serialized | causalab-side; nnsight model also ephemeral (`model.edit()` for persistent edits) | ▣ reusable |

---

## Part 2: nnterp capabilities beyond neural

Functionality nnterp ships *above* the read/write plumbing mapped in Part 1 —
things that would land in causalab's `analyses/`, `methods/`, testing, and
model-onboarding layers rather than in `neural/`. All entries are code-verified
against `~/nnterp` 1.3.0.

**Coupling caveat.** Most of these take an nnsight/nnterp model object, so
adopting them *as code* follows the Part-1 backbone decision. Three, though, are
**portable patterns** worth borrowing even while on pyvene: the `RenameConfig`
declarative onboarding, the load-time "edits actually affect output" validation
gate, and the `get_first_tokens` canonical-token logic.

### Cross-architecture standardization and onboarding

`causalab/neural` currently carries per-model knowledge (pyvene component maps per
model type, the GQA head→KV remap, the decoupled-`head_dim` limit — `PATH_PATCHING.md
§0.5`). nnterp factors that into a declarative rename layer, so a new architecture
is a *config*, not new code.

| Capability | nnterp primitive (file) | Value to causalab | Notes |
|---|---|---|---|
| One module tree across families | `StandardizedTransformer` + rename scheme (`rename_utils.py:189`, `get_rename_dict`) | a single accessor set (`layers_output[i]`, …) across 16+ families removes per-model branching in `neural/` | covers GPT-2/J, Llama, Mistral, Mixtral, Gemma 2/3, Qwen 2/3, Phi-3, OPT, Bloom |
| Onboard a new architecture by config | `RenameConfig` (`rename_utils.py:47`) | add a model with a dataclass (module-name aliases + `*_config_key` overrides), not a new component map | no persistent registry file — configs are per-use |
| Standardized access outside a trace | `ModuleAccessor` (`nnsight_utils.py:198`) | reach standardized submodules as plain `nn.Module` for non-trace code | |

### Model-load validation framework

The strongest fit with causalab's verification culture (golden tiers,
"does this edit do anything"): nnterp validates a model **at load** and ships a
pytest-based runner to validate arbitrary checkpoints.

| Capability | nnterp primitive (file) | Value to causalab | Notes |
|---|---|---|---|
| Load-time renaming + IO-shape check | `check_model_renaming` / `check_io` (init, `standardized_transformer.py`) | a model-onboarding gate: modules exist; layer outputs are `(b, seq, hidden)`; embeddings→layers→logits shapes line up | gated by `check_renaming`; `allow_dispatch` falls back `scan`→`trace` |
| "Edits actually affect output" causal check | `attention_probabilities.check_source()` | verifies an intervention is *real* (probs sum to 1; modifying them changes logits) — exactly causalab's core concern | only when `enable_attention_probs=True`; architecture-specific |
| CLI test runner over your checkpoints | `python -m nnterp run_tests --model-names … --class-names …` (`__main__.py`) | run nnterp's ~5 test modules against causalab's target checkpoints as a CI/onboarding gate | delegates to pytest; ships `nnterp/tests/` |

**Shipped (F2, #393):** `causalab/neural/validate.py` implements the gate.
`validate_model_load(name)` is the per-model onboarding check (the load-time
renaming + IO-shape row); `run_nnterp_tests(...)` wraps the CLI runner (run from a
scratch cwd so pytest ignores causalab's own config); `python -m
causalab.neural.validate` drives both. The `attention_probabilities.check_source()`
"edits affect output" row — deferred here (architecture-specific, gated on
`enable_attention_probs`) — shipped with CAP4 (#457): loading through
`LMPipeline(..., enable_attention_probs=True)` makes nnterp run the gate at
load (under the default `check_renaming=True`), so an enabled pattern accessor
is always a *validated* editable target.

**Shipped (CAP5, #458):** `causalab/neural/preflight.py` adds the zero-compute
`scan()` preflight over the Plan IR. `preflight_plan(model, plan)` checks
position specs statically against each input's padded frame (fake tensors
cannot value-check data-dependent indices) and dry-runs the lowered taps as
sequential per-invoke `model.scan()` passes (layer/head bounds, featurizer
shapes, write widths — the site layer's explicit setitem-broadcast guard makes
the width-mismatch class visible under fake mode). The verdict is three-way:
`clean` / `failed` (the real run would fail the same way) / `unsupported`
(scan cannot run on this model or plan — a missing verdict, **not** a plan
failure; nnterp's `allow_dispatch` would silently fall back to a real trace
here, which a preflight must never do). Two sibling capabilities are
`unsupported` by construction: generation plans (CAP2 `Plan.generate` — scan
cannot express the KV-cached decode loop) and backbones carrying installed
persistent edits (CAP4 `causalab/neural/persistent.py` — an in-scan failure
corrupts the installed mediators, so no scan is attempted on an edited
model). Wired as the opt-in `run_plan(..., preflight=True)` gate and as the
`scan=` column of the `validate` CLI.

**Wheel gap / fork.** The `run_tests` CLI is *unusable* from the nnterp 1.3.0 PyPI
wheel: `package-data` globs `tests/*` but not `data/*`, so `nnterp/data/` is
dropped and `nnterp/tests/utils.py` crashes on import
(`ModuleNotFoundError: nnterp.data`). This is a build-config bug, not a missing
source file, so a plain `git+…` dependency reproduces it (uv rebuilds the same
wheel). F2 therefore pins `nnterp` to a fork carrying the one-line packaging fix
(`pyproject.toml [tool.uv.sources]`); migrating back to upstream once the fix lands
is tracked on the umbrella (#391).

### Analysis methods (logit lens, patchscope)

Portable, self-contained functions that would drop into `analyses/`: they wrap a
trace and return per-layer tensors, no persistent state.

| Capability | nnterp primitive (file) | Value to causalab | Notes |
|---|---|---|---|
| Logit lens (per-layer next-token probs) | `logit_lens(model, prompts) → (prompts, layers, vocab)` (`interventions.py:28`) | a ready logit-lens analysis | `return_inv_logits` option |
| Patchscope lens / generate | `patchscope_lens(...) → (prompts, layers, vocab)`; `patchscope_generate(...) → {layer: tokens}` (`interventions.py:233`, `:303`) | patchscope decoding of intermediate states — a method causalab lacks | follows the patchscopes paper |
| Attention-input patching lens | `patch_object_attn_lens(...)` (`interventions.py:358`) | "attend-to-source-state" lens over consecutive layers | |
| Hidden → vocab projection | `project_on_vocab(model, h)` (`nnsight_utils.py:157`); `get_topk_closest_tokens(h, k)` | the core logit-lens primitive + nearest-token readout for any hidden state | |

### Target-token prompt utilities

nnterp's answer to "track the probability of specific answer tokens across prompts
and layers." This **overlaps** causalab's `output_tokens` / `checker` scoring
(see memory) — the piece most worth borrowing is the canonical-first-token logic.

| Capability | nnterp primitive (file) | Value to causalab | Notes |
|---|---|---|---|
| Semantic target-token tracking | `Prompt` + `Prompt.from_strings` + `Prompt.get_target_probs` (`prompt_utils.py:98`) | aggregate probability mass over answer-token groups across layers — complements `output_tokens` scoring | `has_no_collisions` guards overlapping target ids |
| Canonical first-token ids (space-prefix safe) | `get_first_tokens(words, tokenizer)` (`prompt_utils.py:18`) | robust word→token-id mapping (handles leading-space tokenization) — directly relevant to causalab's digit/answer trailing-space gotchas | uses a `"🍐word"` fallback trick |
| Patchscope target prompts | `TargetPrompt` / `TargetPromptBatch` / `repeat_prompt` / `it_repeat_prompt` (`interventions.py:70`) | ready patchscope prompt scaffolds (incl. chat-templated) | |
| Batched prompt scoring | `run_prompts(model, prompts, batch_size, get_probs_func)` (`prompt_utils.py:172`) | batched per-target probability collection | |

### Activation collection and grafting helpers

| Capability | nnterp primitive (file) | Value to causalab | Notes |
|---|---|---|---|
| Batched / remote-session activation collection | `get_token_activations`, `collect_token_activations_batched`, `collect_last_token_activations_session` (`nnsight_utils.py:254`, `:376`, `:315`) → `(layers, prompts, hidden)` | parallels causalab `collect_features`, with remote-session amortization built in | |
| Layer skipping / activation grafting | `skip_layer(i, skip_with=)`, `skip_layers(a, b, skip_with=)` (`standardized_transformer.py:239`, `:249`) | route input→output or graft a saved activation across a layer span | deliberately not wrapped in causalab (#489: adopted then deleted, zero consumers); would return only as a Plan-expressible op under a new-capability issue |
| Standardized read/write helpers | `get_layer_output` / `set_layer_output` / `get_attention_output` / `get_mlp_output` / `get_next_token_probs` (`nnsight_utils.py`) | uniform activation accessors usable outside `StandardizedTransformer` | |

### Visualization

| Capability | nnterp primitive (file) | Value to causalab | Notes |
|---|---|---|---|
| Top-k tokens-per-layer heatmap | `plot_topk_tokens(probs, tokenizer, k, …) → plotly.Figure` (`display.py:13`) | ready logit-lens / patchscope visualization (HTML or image) | **Plotly** dependency; causalab reports are currently PNG/HTML |
| Prompt inspection table | `prompts_to_df(prompts, tokenizer)` (`display.py:119`) | a pandas view of prompts + target tokens for notebooks | |

### Remote execution (NDIF)

`remote=True` / `model.session(remote=True)` runs the same intervention code on
NDIF-hosted **public** models up to 400B+ with zero local GPU — an escape hatch
for models too large to run locally. **Data-egress caveat:** it ships the
intervention graph + inputs to shared infra and serves only public models — keep
private models on local GPUs (see `NNsight_overview.md §4.2`).

---

## Part 3: implementing the intervention layer on nnsight

Part 1 establishes the rewrite is **declarative → imperative**; this part proposes
*how*, at the design level. It deliberately does **not** preserve causalab's
pyvene-shaped abstractions (`IntervenableConfig`, per-mode intervention
subclasses, the collect-order contract). Instead it asks: given nnsight's
execution model, what primitive *best serves the desiderata*? The mechanics cited
are verified (`NNsight_overview.md`, `~/nnterp`); the abstractions are a proposal.

### The nnsight execution model, briefly

A `with model.trace(input):` block is **deferred**: its body runs interleaved with
the forward pass, in **forward-pass order**, over **real tensors** at each hook
point. Reads are plain tensor access; writes are plain assignment
(`model.layers_output[i][:, pos] = ...`, settable in nnterp via
`LayerAccessor.__setitem__`). Values needed after the pass are `nnsight.save`-d.
Multiple inputs are separate `tracer.invoke(...)` blocks; values move *between*
invokes via `tracer.barrier(n)` or a `model.session()`. Edits are differentiable,
so a trainable parameter used inside a trace receives gradients. The unit of work
is a **trace program**, not a config object.

### One primitive: the site edit (read-modify-write)

Every causalab intervention type collapses to one shape — a function applied at a
**site** (a component accessor + position slice + optional featurizer), operating
in feature space:

```
f   = featurize(read(site))      # read may be omitted (pure write)
f'  = g(f, *aux)                 # aux = values from other sites / inputs / params
write(site, inverse(f'))         # write may be omitted (pure collect → save f)
```

The seven pyvene modes are just choices of *which parts run* and *what `g` is* —
no per-mode subclass, no closure factory:

| Mode | reads | feature-space `g` | writes |
|---|---|---|---|
| collect | site | identity → `save` | — |
| replace | (shape only) | constant source vector | site |
| steer | site | `f + factor·v` | site |
| interchange | site + source-site | `f_base`, then `f_base[sub] = f_src[sub]` | site |
| interpolate | site + source-site | `fn(f_base, f_src, **params)` | site |
| noise | site | `f + scale·randn(generator=seed)` | site |
| mask | site + source-site | `(1−gate(θ))·f_base + gate(θ)·f_src`, `θ` learnable (the DBM gate blends toward a source, not a pure `f`-scaling — pinned by ED2 to the pyvene `FeatureMaskIntervention` semantics) | site |

`featurize`/`inverse` is causalab's existing `Featurizer` (identity when trivial).
So the whole "intervention type" axis becomes **one `Edit` = (site, `g`,
read-sources)**, with the modes as thin constructors. The featurizer is applied by
the compiler *around* `g`, not baked into a backbone class.

### Fusing collect and intervene in one forward pass

This is the capability the current causalab primitives lack. In pyvene, *collect*
and *interchange* are different intervention **types**; reading an activation that
is itself produced *under* an intervention needs the mixed model plus the
`sorted_keys` collect-order contract, and is realized only as the bespoke two-pass
path-patching path. On nnsight it is the default — within a single trace you may
write at one site and read+save at another:

```python
with model.session() as s:
    with s.invoke(source):                        # capture clean activation
        src = nnsight.save(model.layers_output[5][:, pos])
    with s.invoke(base):                          # ONE forward: patch @5 AND read @10
        model.layers_output[5][:, pos] = src      # intervene
        mid = nnsight.save(model.layers_output[10])   # collect *post-intervention*
        out = nnsight.save(model.logits)
```

`mid` is the layer-10 activation *as produced under the layer-5 patch* — collected
in the same pass, no special runner. Generalized, this yields path patching,
receiver-set collection, "collect a feature then steer with it online", and
multi-site mediation as **ordinary plans**, not special cases.

The abstraction that captures it: a **Plan** = a set of site-ops (each `collect` or
`edit`) over a set of inputs, which the compiler lowers to the **minimum number of
forward passes** — same-pass when data dependencies respect forward order,
cross-pass (via `session`/`barrier`) only when a write depends on a *later-layer*
read from another input (the clean→corrupt pattern).

### Proposed abstraction stack

What replaces `causalab/neural` is **not** a backbone wrapper but a **compiler from
causal-experiment specs to nnsight trace programs**:

1. **Accessor layer — nnterp.** `StandardizedTransformer` + uniform accessors +
   load-time validation + cross-architecture renaming. causalab stops owning
   per-model component maps and the GQA/`head_dim` special-casing.
2. **Site** *(causalab, ▣ mostly reused)* — `(component, layer, position-resolver,
   featurizer)`. Position-resolver = the existing `Template`/`TokenPosition`;
   featurizer = the existing learned spaces. Resolves to an nnterp accessor + a
   slice + a `(featurize, inverse)` pair. A `HeadView` adapter supplies the
   per-head value/query reshape nnterp lacks (the ✗ gap).
3. **Edit** *(causalab, new but tiny)* — a site + a feature-space `g` + its
   read-sources. The seven modes are constructors over it.
4. **Plan / trace-compiler** *(causalab, new — the real work)* — lowers a
   declarative spec `(inputs × sites × edits × what-to-save)` to nnsight trace
   program(s): orders ops by forward position, fuses collect∘intervene, minimizes
   passes, batches the counterfactual dataset, and handles paired base/CF indices,
   ragged spans, and padding frames.
5. **Methodology / scoring** *(causalab, unchanged)* — IIA, DAS/alignment, base
   accuracy, `output_tokens` scoring — consumes the saved logits/activations.

This **deletes, not ports**: `IntervenableConfig`/`IntervenableModel`, `use_fast`,
the `sorted_keys` contract, per-mode subclass closures, manual teardown, per-key
`hf_device_map` moves, and the per-model component strings.

### What causalab still owns

If nnterp is the substrate, causalab's value is the **causal-abstraction layer** —
everything nnterp has no concept of:

- **Causal models & counterfactual datasets** — `CausalModel`, counterfactual
  generation, expected counterfactual labels, **interchange-intervention accuracy
  / IIA**. nnterp has no notion of a hypothesized causal variable or its alignment
  to a representation.
- **Featurizers as trained, serializable, composable feature spaces** (SAE, DAS
  rotation, subspace) attached to *arbitrary* sites — nnterp has no featurizer.
- **Alignment & adjudication methodology** — DAS search, fixed-subspace IIA, the
  subspace-mediation pipeline.
- **The experiment harness** — declarative configs, the runner, batching, caching,
  golden tiers, reporting / artifact viewer. causalab is an *experiment platform*;
  nnterp is a *library*.
- **The trace compiler itself** — per-example paired interchange at scale with
  causal scoring (incl. the fused collect∘intervene). nnterp *collects
  activations*; it does not run paired base/CF interventions at scale and score
  them causally.
- **Token-position semantics tied to causal variables** — `Template` binds
  positions to causal-model variables and counterfactual pairs, richer than
  nnterp's target-token prompts.

In one line: **causalab = the causal-abstraction layer (models, counterfactuals,
featurizers, alignment, scoring, harness) + a trace compiler targeting nnterp;
nnterp = standardized, validated model access + raw read/write.** `causalab/neural`
shrinks from a pyvene wrapper into a Site / Edit / Plan compiler.

### Open design questions

- **Per-head value/query (the ✗ gap) — RESOLVED by the F4 spike (#395,
  `causalab/neural/head_view.py`).** The gap is closable; the `HeadView` contract is
  fixed and pinned against a raw-hook oracle (CPU coupled/GQA/decoupled-`head_dim`
  Llama + a Qwen3-4B golden). Findings that revise the original plan:
  - The receivers are **not** slices of `attentions_output`; they are the attention
    *projections*: **value** = `v_proj` output reshaped `(b, seq, n_kv, d_head)`
    (KV-head space), **query** = `q_proj` output `(b, seq, n_heads, d_head)`
    (query-head space, pre-RoPE), **sender** = `o_proj` *input* `(b, seq, n_heads,
    d_head)`. All `d_head`-wide at the **true `head_dim`** (`config.head_dim or
    hidden//n_heads`) — the decoupled case (Qwen3 128≠80) pyvene mis-slices.
  - **`.source` is NOT needed for separate-projection families** (Llama, Qwen3, all
    causalab GQA targets): plain projection I/O + reshape suffices. **ST4 (#399)
    closed the fused-QKV case (GPT-2 `c_attn`) *without* `.source` after all**: the
    source view ops (`value_states_view_*`) come out of `.split()` — multi-view
    outputs autograd refuses to mutate in place — so they are read-only, and their
    names track the transformers source line by line. Fused receivers instead
    slice the `c_attn` module output's equal-width `[q|k|v]` columns (write-capable,
    stable), with the sender on `c_proj`'s input. One consequence: fused taps
    execute in a different order (query/value collapse onto the first-firing
    `c_attn`), so forward-rank resolution is model-aware
    (`HeadView.kind_rank` / `forward_rank_on`).
  - The **per-head write path works** (reshaped-slice edit reproduces a hand-rolled
    `v_proj` hook), de-risking ST4/ED1. ST4 shipped it as `HeadView.write` and
    wrapped single heads into the Site protocol as `HeadSite` (#399).
- **Trainable interventions (mask, DAS) — RESOLVED by ED3 (#402; the F6 spike
  never ran standalone, so ED3 pinned the contract directly).** The grad
  contract, pinned by grad-parity tests against a grad-enabled raw-hook oracle
  (`tests/neural/test_trainable.py` — same modules, same loss, same
  `param.grad`):
  - **Saved-logits backward is the contract**: save the logits on-device with
    the autograd graph intact (`model.logits.save()`, no `.cpu()`/detach),
    compute the loss and run `backward()` *outside* the trace. Loss-inside-trace
    (`with loss.backward():`) is not used anywhere.
  - Gradients flow to external `nn.Module` params applied in-trace through
    `LayerAccessor.__setitem__` — both the tuple-rewrap path (`block_output`)
    and plain writes (`mlp_output`) — and through *both* featurize paths (the
    base read and a raw source featurized live in the trace), at batch scale
    with left-padding, and through a `HeadView` projection-site write.
  - Model params freeze **once at load** (`LMPipeline._apply_model_conventions`)
    — no per-intervenable `disable_model_gradients`; featurizer/gate placement
    is one explicit pass (`trainable.place_edit_parameters`, replacing pyvene's
    `get_device` monkeypatch). The multi-device sharded row (#422) is pinned
    by `TestMultiDeviceShardedGolden` (`tests/neural/test_trainable.py`): tiny
    Llama under an explicit two-device `hf_device_map` — placement follows
    each site's layer device (site-backed read-sources go to *their* site's
    device), loss + `param.grad` match the single-device run exactly (incl. a
    raw source collected on one device, consumed at a site on another), and
    one AdamW steps params scattered across devices with its state resident
    per-param. Needs ≥2 CUDA devices, so the single-GPU nightly *skips* it —
    it's an on-demand pin (`pytest -m golden -k MultiDeviceSharded` on a
    2-GPU allocation).
- **Batching model — RESOLVED by PL1's probes + PL3 (#405; the F5 spike never
  ran standalone, so the layout landed where it was consumed).** The measured
  facts (pinned in `plan.py`'s docstring and `tests/neural/test_dataset.py`):
  multi-invoke traces fuse into ONE left-padded forward with row-scoped reads,
  so the **fused single trace** is the layout for prefill/collect work
  (`run_plan`, one invoke per already-batched input, frame-aligned). For
  *generation* the layout is **split-forward, pyvene-parity**: one
  early-stopped collect pass per counterfactual group, then one
  `model.generate` trace over the base batch with all edits applied to the
  prefill (measured: prefill edits persist through cached decode steps and
  match a prefill-hooked HF-generate oracle exactly). Ragged spans batch as a
  flat `(row_ids, col_ids)` gather/scatter (`site.RaggedIndex` — no
  length-bucketing); padding frames dissolve into the run encoding itself
  (`resolve_positions_batched`: one `pipeline.load(traces,
  return_offsets_mapping=True)` per batch side, positions born in the padded
  frame, the unpadded→padded shift now legacy-only). The third lowering —
  **staged traces** (PL2; *as landed*: plain sequential traces — the
  `model.session()` wrapper was removed by EU1 #482, and since EU2 #483
  `StagingRequired` is a strictness-only fact, not compiler control flow;
  see Part 5) — applies exactly when a plan cannot fit one fused trace:
  mixed padded frames, data flow against forward order (the two-pass
  path-patching shape), or inputs bound to different models;
  `run_plan(lowering="auto")` executes the staged schedule directly.
  `tracer.cache` stays off-limits (hand-rolled saves throughout).
  `tracer.barrier` is functionally pinned at batch scale (the plan /
  walking-skeleton / path-patching suites run whole left-padded batches
  through fused traces with barriers, incl. GPU goldens) but its overhead was
  never quantified, and the fused-vs-split memory/throughput trade on a large
  real model remains unmeasured — the pyvene A/B is moot post-SH2 (pyvene is
  deleted); both can be revisited post-cutover if generation cost ever
  matters (CAP1 territory).
- **Generation-step & gradient headroom (PL1 acceptance criteria).** The Plan IR
  must be *able to address* a generation step (decode-time edits via `tracer.iter`)
  and *request* gradients, even though both are implemented post-cutover (CAP1).
  Retrofitting either into a prefill-only, forward-only IR would be a second
  migration — assert the headroom in the IR design now, cheaply.
- **Forward-order constraint** *(confirmed concretely by the F4 spike)*. nnsight
  requires reads of different modules in one trace to be requested in execution
  order — reading `v_proj` before `q_proj` in one block raises `MissedProviderError`.
  The compiler must topologically order site-ops by layer (and by intra-block
  position: q→k→v→o) and split or reject plans needing a backward dependency within
  one invoke. `HeadView.collect` already does this ordering for its own reads.
- **Determinism.** Seeded noise and golden-test stability must reproduce on the new
  backend (carry seeds through to `torch.Generator` inside the trace).
- **Dependency floor.** `nnsight>=0.6` plus a `transformers` version chosen to keep
  the `causalab/neural` hook-oracle harness green while pyvene remains installed
  (pyvene is deleted at cutover, never run as a parallel backend — see migration notes).

---

## Part 4: work breakdown into parallel waves

Decomposes Part 3 into concrete sub-issues (a "rebase on nnterp" umbrella). Each
row is independently ownable; **Depends on** is the full dependency graph; **Wave**
= dependency depth, so every issue in a wave can be built in parallel once the
previous wave lands. IDs group by layer: `F*` foundations, `ST*` site, `ED*` edit,
`PL*` plan/compiler, `MX*` methodology, `SH*` ship. The `▣` desiderata from Part 1
(token-position resolution, featurizer composition, post-processing, persistence)
carry over unchanged and are folded into the issues that reuse them.

### Sub-issues

| ID | Sub-issue | Scope | Depends on | Wave |
|----|-----------|-------|-----------|------|
| F1 | Backbone deps & load spike | add `nnsight≥0.6` + `nnterp` (pyvene stays installed, deleted at SH2); land whatever `transformers` keeps the `causalab/neural` hook-oracle harness green; confirm `StandardizedTransformer` loads causalab's target checkpoints (incl. GQA) | — | 1 |
| F2 | nnterp validation gate | adopt load-time checks + `python -m nnterp run_tests` for model onboarding / CI | F1 | 2 |
| F3 | nnterp model adapter | `Pipeline` replacement: load / tokenize (reuse chat-template, left-pad, `offset_mapping`, `position_ids`) / generate / dump / introspection over `StandardizedTransformer` | F1 | 2 |
| F4 | Per-head reshape spike | de-risk the ✗ gap: per-head value/query via raw nnterp reshape + `.source` on a GQA model; fix the `HeadView` contract | F1 | 2 |
| F5 | Invoke-batching & padding spike | de-risk the batching model **before ED1/PL1 fix interfaces**: how nnsight pads/aligns multi-invoke traces of different lengths; per-invoke read shapes vs. one concatenated batch; memory + throughput of a fused base+sources trace vs. pyvene's split forwards on a real model; barrier overhead at batch scale; per-invoke position-frame mapping; confirm `tracer.cache` is unusable under renaming. Output: the layout decision recorded in this doc | F3 | 3 |
| F6 | Grad-flow spike | de-risk trainable edits **before ED1**: gradients through `LayerAccessor.__setitem__` (tuple rewrap), through a featurizer module applied in-trace at a site, loss-inside-trace vs. saved-logits backward, at batch scale on a tiny model; featurizer-param device placement on sharded models. Output: a pinned mini-test + the grad contract ED1/ED3 build on | F3 | 3 |
| ST1 | Site core | `(component, layer, position, featurizer)` → nnterp accessor + slice for residual (in/out), MLP (in/out/act), attention output | F3 | 3 |
| ST2 | Position-resolver bridge | reuse `Template`/`TokenPosition` → in-trace slices; padding-frame shift `▣` | ST1 | 4 |
| ST3 | Featurizer-in-trace | apply `(featurize, inverse)` around reads/writes; identity, composition, subspace indexing `▣` | ST1 | 4 |
| ST4 | HeadView adapter | per-head value/query reshape + GQA remap + `.source` (closes the ✗ gap) | ST1, F4 | 4 |
| ED1 | Edit core | the read-modify-write primitive (read-only = collect, write-only = replace, RMW general form) | ST1, ST3 | 5 |
| ED2 | Mode constructors | the 7 modes over `Edit` (seeded noise; source-site reads for interchange/interpolate) | ED1 | 6 |
| ED3 | Trainable edits | mask / DAS: learnable params in-trace (grad contract pinned by F6); **the differentiable loss slice** (label-concat forward + CE, today `LM_loss_and_metric_fn`) lands here — MX1/MX2 consume it, never re-implement it; outer optimization loop; featurizer-param device placement + optimizer on sharded models | ED2, F6 | 7 |
| PL1 | Plan IR + single-trace compiler | spec → one trace, **including the canonical cross-invoke interchange (source + base invokes + `barrier` within one trace)**; order ops by forward position; the collect∘intervene fusion. **IR headroom acceptance criteria:** an Edit can address a generation step and a Plan can request gradients (implementation lands post-cutover — CAP1) | ED1, ST2, F5 | 6 |
| WS1 | Walking skeleton (e2e gate) | minimal end-to-end interchange IIA on the new stack: residual-stream Site/Edit/Plan on one preset task — fixed spans, one CF group, no heads — scored with the existing metric off Plan-saved outputs and matched to a captured golden (smoke + golden tiers). Throwaway glue allowed; not a reroute. **Gates Wave 8: PL3–PL5 open only on a green skeleton** | ED2, PL1 | 7 |
| PL2 | Multi-trace / staged compiler | `session` staging across traces; cross-input generality beyond the canonical pair; pass minimization; `OutOfOrderError` guards with actionable errors | PL1 | 7 |
| PL3 | Batched dataset execution | paired base/CF, ragged spans, padding frames (layout per the F5 decision); collect / interchange / interpolation / steer / ablation / causal-tracing at scale; **reroutes the public wrappers it enables** (`collect_features`, `run_interchange_interventions`, `run_steering_interventions`, `run_ablation`, `run_causal_trace`) as each lands, gated by the oracle + captured goldens | PL2, WS1 | 8 |
| PL4 | Cross-model patching | two model objects; capture-source → inject-target as a Plan | PL2, WS1 | 8 |
| PL5 | Path patching / receiver sets | two-pass path patching as Plans (replaces mixed model + `sorted_keys`); **reroutes `run_path_patching*`** | PL2, ST4, WS1 | 8 |
| MX1 | Scoring adapter | route the **inference-time** metrics (IIA / base accuracy / `output_tokens` scoring) off Plan-saved outputs; consumes ED3's differentiable loss slice — does not own it | PL3, ED3 | 9 |
| MX2 | DAS / alignment | alignment search + fixed-subspace IIA on the new backend; **reroutes `train_interventions` + the DAS/DBM/boundless grids** | ED3, MX1 | 10 |
| SH1 | Parity & golden harness | per-mode numerical parity vs the backbone-agnostic hook-oracle + captured goldens (not live pyvene) on existing tiers; determinism/seeding; **all parity + goldens run in eager attention** (the current default) — the SDPA flip is SH3, post-cutover (opens W7, grows through PL/MX) | ED2 | 7 |
| SH2 | Cutover (deletion) | public wrappers are already rerouted incrementally by PL3/PL5/MX2 — SH2 **deletes**: pyvene plumbing (`IntervenableConfig`/`Model`, `use_fast`, `sorted_keys`, subclass closures, component maps) **and the pyvene dependency**; sweeps remaining analyses/imports/stragglers; full-tier green run — no feature flag | all | 11 |
| SH3 | SDPA flip + golden repin | flip the attention default eager→SDPA (eager stays opt-in for `attention_probabilities` work); re-enable `use_cache` in generation; deliberate same-GPU golden repin; record the speedup | SH2 | 12 |
| CAP1 | Capability exploitation tracker | post-cutover backlog of what the migration was *for*: decode-step interventions (`tracer.iter`), attribution-patching pre-scans for locate/DAS grids, attention-probability editing (knockout), `scan()` preflight for position specs / batch plans, `tracer.stop()` collection early-exit, `model.edit()` persistent steering across evals (landed as CAP7, #460 — `causalab/neural/persistent.py`), adopt nnterp `logit_lens`/patchscope + `get_first_tokens` into `methods/` — each graduates to its own issue when picked up | SH2 | 12+ |

### Waves

- **Wave 1 — spike the blocker:** F1. Single gate; nothing starts until backbone deps load.
- **Wave 2 — substrate (×3 parallel):** F2, F3, F4 — validation gate, model adapter, head-reshape de-risk, independent.
- **Wave 3 — site hub + de-risk spikes (×3 parallel):** ST1, F5, F6. The two spikes answer the batching-layout and grad-flow questions *before* ED1/PL1 fix the interfaces those answers shape.
- **Wave 4 — site adapters (×3 parallel):** ST2, ST3, ST4.
- **Wave 5 — edit hub:** ED1.
- **Wave 6 — modes ‖ compiler (×2):** ED2, PL1.
- **Wave 7 — depth + the e2e gate (×4):** ED3, PL2, SH1, **WS1**. WS1 is the milestone: the first end-to-end IIA number on the new stack, matched to a captured golden. PL3–PL5 open only on a green skeleton — waves 8+ *generalize a working path* instead of extending an unexercised one.
- **Wave 8 — scale & advanced modes (×3 parallel):** PL3, PL4, PL5 — each reroutes the public wrappers it enables as it lands.
- **Wave 9 — scoring:** MX1.
- **Wave 10 — alignment:** MX2 (reroutes DAS training/grids).
- **Wave 11 — cutover:** SH2 (deletion + stragglers, not a big-bang reroute).
- **Wave 12 — post-cutover:** SH3 (SDPA flip + golden repin); CAP1 (capability backlog, unscheduled).

### Critical path and parallelism

Critical path (≈11 hops, the serial spine):
`F1 → F3 → ST1 → ED1 → PL1 → WS1 → PL3 → MX1 → MX2 → SH2 (→ SH3)`.

This is the layered core (adapter → site → edit → compiler → **e2e gate** → scale →
scoring → cutover); the **singleton waves (1, 5, 9, 10, 11) are its bottlenecks** —
front-load and de-risk them. Parallel width peaks at **4** (Wave 7), so a
2–3-stream effort saturates the plan. Schedule off-critical-path work
opportunistically: **F2** (validation gate) and **SH1** (parity harness) run
alongside almost everything once their single dependency lands; **F4** must land
before ST4/PL5, and **F5/F6** before ED1/PL1 (the spikes exist to be consumed by
interface design, not to run alongside it). WS1 adds one hop to the spine and buys
the plan its earliest integration evidence — interface mistakes in ST/ED/PL surface
at Wave 7 instead of Wave 9+.

### Orchestration: how we build it

Ship as **stacked PRs**, not a fan-out job. A long-lived integration branch
(`nnterp-rebase`) carries the whole effort: a **draft umbrella PR**
(`main ← nnterp-rebase`) is the visible rollup, and every sub-issue is its own PR
**based on the integration branch** (`Closes #<sub>`, body `Part of #<umbrella>`)
that lands *into* `nnterp-rebase` a wave at a time — `nnterp-rebase → main` merges
only once the cutover (SH2) is green. Each sub-PR is implemented **one PR at a
time**, with its base set to the integration branch rather than `main`. This is
deliberate: the dependency graph is a ~10-hop serial spine
with parallel width ≤ 3 (see [Critical path](#critical-path-and-parallelism)), so
there is little to fan out, and each sub-issue is a design-and-implement task that
wants the reference's interactive plan → review loop rather than a headless agent.
Fan-out (a Claude Code **workflow**) is reserved for the bounded, read-only sweeps
where it pays: the initial `pyvene` / `IntervenableModel` call-site inventory that
sizes the sub-issues, and SH1's per-mode numerical-parity sweep across configs.

Three process rules that keep the waves honest:

- **Within-wave concurrency is allowed.** "One PR at a time" applies within a
  dependency chain, not across a wave: when a wave has width (Waves 2, 3, 4, 7, 8),
  2–3 sub-PRs may proceed concurrently in separate worktrees. Otherwise the plan's
  advertised parallelism is theoretical and the spine pays ~11 serial review
  round-trips.
- **Incremental reroute, single deletion.** Each PL/MX issue reroutes the public
  wrappers it enables *on the integration branch* the moment its oracle + captured
  goldens are green (PL3 → collect/interchange/steer/ablation/causal-tracing,
  PL5 → path patching, MX2 → DAS training/grids). "Replace, not dual-backbone" is
  about never shipping a backend flag to `main` — it does not forbid incremental
  reroute inside the branch. This gives the new stack CI-shaped production traffic
  from Wave 8 onward and shrinks SH2 from the biggest PR of the effort to a
  deletion sweep.
- **Wave-open acceptance criteria.** When a wave opens, expand each of its
  sub-issue bodies from the one-line scope into concrete acceptance criteria — the
  contract guarantees (device placement, padding-frame invariants, tuple handling,
  error behavior) and the oracle/golden tests that pin them — *before*
  implementation starts. A one-line scope is a pointer, not a definition of done.

---

## Part 5: engine unification (one scheduler, one executor, two emitters — no session)

*Added post-rebase (2026-07-06). Umbrella **#480**; design-of-record for the
EU0–EU7 sub-issues. **Annotated as-landed (EU6, #488)**: the bold
**shipped** blocks below record the state on the epic branch with EU0–EU7
all landed (EU5b #487 — consumer migration — merged via PR #500 during this
annotation's review round; its decisions are folded in below); every
file:line anchor is exact in any tree containing
this annotation (EU6's own module-docstring cross-refs shift
`plan.py`/`staged.py`/`dataset.py` line numbers, so anchors resolve at the
EU6 PR's head — not on the bare pre-EU6 epic tip).
The unannotated prose is the original design, kept as the record of intent.*

### Why: three lowerings plus a bypass

The rebase grew the engine in waves, leaving four parallel trace-emission paths:

1. **Single-trace** (`plan.py::_run_single_trace`) — one fused multi-invoke
   trace + `tracer.barrier`; sole home of gradient plans (CAP3).
2. **Staged** (`staged.py::run_plan_staged`) — sequential traces inside
   `model.session()`; the real scheduler (`_schedule` + `_rendezvous_conflict`).
3. **Generation** (`plan.py::_run_generate_trace`) — ONE `model.generate` trace
   on `tracer.iter`; refuses cross-input reads and multiple inputs.
4. **The bypass** — `dataset.run_intervened_generation` hand-codes the
   split-forward layout (per-group `collect_ordered` pass →
   `_generate_with_edits`) because it predates CAP2: a **second generate-trace
   emitter** duplicating the position_ids fix, prefill-persistence semantics,
   and generate defaults.

Verified duplication: the single-trace guards (`_cross_input_phases`,
`_check_barrier_schedulable`, the frame checks, the model check) are
predicate-form duplicates of the staged scheduler's four per-edge fusability
rules (`staged.py` `_schedule`); the staged executor re-implements the
single-input fast path; `staged.py`'s own docstring concedes the single fused
trace is its degenerate schedule ("auto only pays a second pass when one is
semantically required"). Every new seam currently has to be maintained in two
places — once as a refusal, once as a scheduling rule.

**Shipped: the four paths are now one.** EU1 (#482) removed the session,
EU2 (#483) unified plain plans, EU3 (#484) made generation a terminal stage
kind — `run_plan_staged` survives as the ONE executor entry while
`_run_single_trace`, `_run_generate_trace`, and the four single-trace guards
are deleted; the bypass (4) rerouted onto the engine and
`_generate_with_edits` is deleted (EU4, #485). The names in the list above
survive only in this motivating record.

### Target architecture

```
public wrappers (routes unchanged; output shape unified → GenerationResult)
        │  build Plans (+ batch loop / unit→site adaptation in dataset.py)
        ▼
   Plan IR  (unchanged: Plan/CollectOp/EditOp/GenerateSpec/GradientRequest)
        ▼
   SCHEDULER (one; generalized staged._schedule)
        │  → StagedProgram: stages → trace groups → invokes;
        │    + generate_key (terminal model.generate invoke)
        │    + staged_why (per-edge staging reason → strict-mode messages)
        ▼
   EXECUTOR (one; plain sequential traces — NO model.session())
        │  constants cross stage boundaries as saved tensors
        │  (device/dtype coerced at the consuming site)
        ▼
   EMITTERS (two)
     ├─ plain-trace body   (fused multi-invoke + barrier | single-invoke
     │                      fast path | _stop_carrier early stop)
     └─ generate-trace body (bounded tracer.iter discipline; the ONLY
                             generate emitter; NEVER inside a session)
```

- **Single-trace = the degenerate schedule** (one stage, one group).
  `StagingRequired` becomes a schedule *fact* (`num_traces > 1`), raised only
  under `run_plan(..., lowering="single")` strictness — the message assembled
  from `staged_why`, preserving each old refusal's key phrase.
  `lowering="staged"` becomes a deprecated alias of auto (its only delta was
  the session).
  **Shipped (EU2, #483 / PR #495).** `lowering` knob as landed: `"auto"`
  (default) executes the schedule; `"single"` is a strictness assertion —
  `_refuse_not_single` (`causalab/neural/staged.py:520`) raises iff
  `num_traces > 1`; `"staged"` is a documentation-only deprecated alias of
  auto (no `DeprecationWarning` is emitted). Two strictness deltas vs. the
  retired path: `"single"` now *refuses* no-edge multi-input plans the
  retired lowering fused — grouping is connected-components, so auto runs
  them as per-input traces (value-identical; only fire-counts moved) — and
  since EU3 the flag applies to generation plans, which the old
  short-circuit bypassed (a stage-less generation plan passes; any
  force-staged read refuses via the `"generate-with-variable-intervention"` arm).
- **Generation = a terminal stage kind.** Cross-input reads whose consumer is
  the generate stage are force-staged edges (`_generate_forcing`): collect
  stages run first, constants materialize, ONE final generate trace runs last.
  This *derives* the split-forward layout `run_intervened_generation`
  hand-codes today. The cross-stage constant machinery already exists (produce
  taps `.save()` into slots; lazy `aux_get` + `FeaturizedSite._coerce` deliver
  with device/dtype coercion — the same path that carries cross-model values).
  **Shipped (EU3, #484 / PR #496).** `StagedProgram.generate_key`
  (`causalab/neural/staged.py:187`) — always a plain input name, never
  inside `stages`, emitted LAST; `_generate_forcing` (`staged.py:433`)
  seeds reason `"generate-with-variable-intervention"`, checked before every fusability rule,
  so even a frame-aligned forward-rank edge stages; the ONE generate
  emitter is `_emit_generate_trace` (`causalab/neural/plan.py:1215`);
  `num_traces` counts the terminal generate trace. Read semantics as landed
  (documented on `GenerateSpec`, `plan.py:300`): a **cross-input** read
  captures in the source input's own plain forward, in *that input's*
  padded frame — frames need NOT align across the forced boundary; a
  **same-input read at/before** the written site stays in-trace, in the
  op's own step frame; a **same-input read after** the written site reads
  the `(input, "clean")` prefill pass — the clean full-prompt-frame value
  regardless of the consuming op's `step`. The ≠1-input refusal narrowed to
  "a generation plan's *ops* must address ONE input" (`_generate_invoke`,
  `staged.py:412`); source inputs are read-only collect stages.
- **Gradients gate on schedule shape** (exactly one trace, one input, no
  generate stage) instead of on the entry function — keyed on **trace count**
  (the pinned refusal test is a no-edge two-input plan). The measured
  multi-invoke refusal (row-scoped invoke reads are dead-end autograd
  branches) is preserved verbatim.
  **Shipped (EU2, #483)** as `_refuse_gradient_shape`
  (`causalab/neural/plan.py:590`): pass iff the program is exactly one
  trace of one invoke (no generate arm needed — gradients × generate is
  refused at construction). One pinned addition beyond the design:
  `run_plan_staged` (`staged.py:772`) keeps an inline gradient refusal
  that `run_plan` routing never reaches (`TestScheduleUnit` pins it
  model-free).
- **`run_intervened_generation` reroutes** onto `run_plan` (cross-model via
  `Plan.models`; `modes.interchange`/`interpolate` gain `source_input`);
  `_generate_with_edits` is deleted. Seeded-noise draw order stays
  bit-identical (tap order is a monotone image of today's sorted writes) — no
  golden repins on the reroute.
  **Shipped (EU4, #485 / PR #497)**, stronger than designed on draw order:
  instead of the monotone-image argument, `dataset.py` declares `EditOp`s
  in the legacy sorted `(layer, rank, within-group)` order, making the
  engine's tap sort the **identity** on the declaration sequence — orders
  coincide for every expressible shape (A/B bit-identical at the sharpest
  divergent shape; zero repins). The one acknowledged numerical residual
  class (EU2): a **shared** `SeededNoise` stream across edits at ≥2 layers
  on ≥2 *disconnected* plan inputs draws in per-input-trace order rather
  than the retired fused trace's layer-major order — unshipped, unpinned,
  documented on `_grouped` (`staged.py:381`); it cannot arise in a
  generation plan (ops address ONE input). Two previously-silent unshipped
  inputs now refuse loudly on the Plan arm only (the op-less
  `_plain_generate` fallback, `dataset.py:650`, stays legacy-silent):
  reserved HF kwargs (`use_cache`/`return_dict_in_generate`) raise at
  `GenerateSpec` construction (`plan.py:351`), and prompt-shaped
  `position_ids` × `max_new_tokens > 1` raises in `_check_generate_inputs`
  (`plan.py:1167`).
- **One `GenerationResult`** (in `pipeline.py`: `sequences` right-padded
  `(n, max_new_tokens)` CPU, `strings: list[str]`, optional per-step `scores`
  or `scores_top_k`, `to_raw_results()` keeping the stored-artifact schema)
  replaces the three divergent shapes across `pipeline.generate` /
  `collect_dataset_features` / `run_intervened_generation`.
  **Shipped (EU5a, #486 / PR #498).** Public in
  `causalab/neural/pipeline.py:122`; the two score forms are exclusive at
  construction; `to_raw_results()` (`pipeline.py:166`) emits ONE synthetic
  batch — byte-compatible with the legacy `batch_size >= n_examples`
  output (batch-nesting instance boundaries were `batch_size`-dependent,
  never schema); `compress_scores_top_k` (`pipeline.py:192`) is
  structure-identical to the legacy `convert_to_top_k` (pinned; that
  helper and its module were deleted once path patching migrated to the
  flat surface, EU5b #487);
  `right_pad_sequences` (`pipeline.py:94`) carries the width contract —
  `sequences` width is the PIPELINE's `max_new_tokens`, not a `gen_kwargs`
  override's (a deliberate legacy quirk, now named in one place).
  Cross-batch ragged score steps (early EOS finishing one internal batch
  before another) refuse loudly (`ValueError`,
  `causalab/neural/dataset.py:633`; the message names the escapes).
  Consumer migration landed as **EU5b (#487 / PR #500)**. Ragged-scores
  decision as landed: the engine's loud refusal ("cannot flatten per-step
  scores…") propagates out of `run_centroid_layer_scan` **unmodified** —
  no caller-side handling and no `min_new_tokens` injection, which would
  silently change the estimand on exactly the affected edge (scoring
  real-but-post-EOS steps legacy never scored); legacy instead silently
  *dropped* short internal batches, biasing cell averages toward late-EOS
  examples. The escape hatches stay explicit caller choices, named in the
  error message (`batch_size >= len(dataset)` or
  `min_new_tokens=max_new_tokens`); a dataset that uniformly generates too
  few steps scores `nan` (the flat analogue of legacy's all-batches-skipped
  outcome). Pinned in `tests/methods/interchange/test_layer_scan.py`
  (propagation + zero-step nan) and directly at the engine's flatten tail
  in `tests/neural/test_dataset.py` (message + both escape hatches). The
  EU5a review's multi-batch stored-artifact nesting pin also landed
  (`TestSaveInterventionResults`, `tests/io/test_artifacts.py`).
- **Deliberate non-Plan bypasses shrink to two**, both documented:
  `trainable.traced_label_loss` (the measured grad contract) and
  `methods/pullback/optimization.py:720` (optimization inner loop).
  The layer-skip ablation helper is **deleted** (#489: zero consumers; the
  capability lives upstream in nnterp's `skip_layers` and would return as a
  Plan-expressible op under a new-capability issue).
  **Shipped — this is the bypass registry.** The only two deliberate
  non-Plan emissions of engine-covered work (interventions / gradient
  collection), anchors verified: `causalab/neural/trainable.py:377`
  (`traced_label_loss` — the saved-logits-backward grad contract every
  trainable loss builds on) and
  `causalab/methods/pullback/optimization.py:720` (the pullback
  optimization inner loop's per-timestep edit + logits trace). Not counted,
  for the next auditor's grep: the engine's own emitters
  (`plan.py:1084` gradient trace, `plan.py:1324` generate emitter, and
  `_run_trace_group`'s plain-trace bodies — `staged.py:700` single-invoke
  fast path, `staged.py:715` fused multi-invoke), the
  site-layer read primitive the compiler is built from (`collect_ordered`,
  `causalab/neural/site.py:581` — collect-only, no interventions), and the
  sanctioned op-less baseline arm of the EU4 reroute (`_plain_generate`,
  `dataset.py:650` — no ops, kept legacy-silent by design).

### No-session policy

`model.session()` is removed from the execution path entirely (the
nullcontext branch cross-model plans already take becomes the only path):

- Locally measured **time-free and benefit-free**: traces inside a session
  still run as separate sequential forwards (no fusion, no shared KV cache),
  and between locally-executed traces saved values are concrete tensors —
  which is all the stage boundaries need. Numerically proven by the
  cross-model suite, which runs sessionless today.
- **Session-around-generate is unmeasured risk** on the most fragile trace
  body in the package: the `tracer.iter` step-counter hooks register at loop
  *entry* (the bounded iterator must be the FIRST statement), and a skipped
  iteration silently abandons the rest of the body.
- **Re-introduction gate**: a future session PR must first pin step-counter
  integrity + body-abandonment behavior under a session, demonstrate a
  measured GPU wall-clock win on a real workload, and must NEVER wrap a
  generate trace. The genuine session use case is remote (NDIF) round-trip
  amortization, which is out of scope (§6 boundary).

**Shipped (EU1, #482 / PR #493).** The policy of record lives in
`causalab/neural/staged.py`'s module docstring — the "**No session.**"
section (`staged.py:74`) — with the three-part gate exactly as above; any
future session PR starts there, never by wrapping a generate trace. The
removal measured value-free on H100 (bitwise-identical logits, −0.11% wall
clock). One *measured* hazard is recorded in the policy as motivation,
found while benchmarking (nnsight 0.7): a session context **defers its
block body** — a fresh local assigned inside `with model.session():` never
binds in the enclosing frame (`UnboundLocalError`); the removed executor
survived only because it mutated a pre-existing `logits` dict. It sits
beside gate item (a) — whose pins are step-counter integrity and
body-abandonment under a session — as the recorded evidence of why those
pins are needed.

### Design

- `StagedProgram` gains `generate_key: InvokeKey | None` (never inside
  `stages`) and `staged_why: Mapping[_Edge, str]`
  (`"generate-with-variable-intervention" | "intervene-backwards" | "cross-model" | "variable-token-positions" |
  "chain-across-invokes" | "separate-concurrent-interventions"`), plus a `num_traces` property.
- `lower_plan(model, plan)` generalizes `_lower`: always
  `_build_taps(staged=True)` (the superset build — clean-invoke reroute,
  produce-tap saves), then `_generate_forcing`, then `_schedule` recording
  `staged_why`. It must thread **grad_leaves** (4th `_build_taps` element).
  `lower_staged` survives as the structural test surface.
- One executor: `_run_trace_group` merges the duplicated single-key fast path
  with the fused `_emit_invokes` path; the terminal generate stage emits via
  `_emit_generate_trace` (the renamed `_run_generate_trace`, minus tap
  building and the cross-input/multi-input refusals, keeping the input-shape
  refusals and the emission discipline verbatim).
- Dies from `plan.py`: `_run_single_trace`, `_check_barrier_schedulable`,
  `_frame_lengths`, `_cross_input_phases`. Grouping stays
  connected-components-only: a no-edge multi-input plan runs per-input traces
  (value-identical; the additive grouper extension is the fallback).

**Shipped (EU2 #483 + EU3 #484), anchors as landed:** `StagedProgram`
(`causalab/neural/staged.py:170`; `staged_why` :186, `generate_key` :187,
`num_traces` :190; the reason vocabulary `_STAGED_WHY` :159 — EU3 appended
`"generate-with-variable-intervention"`). `lower_plan` (`staged.py:448`) returns a
`LoweredPlan` (`staged.py:202`: program + taps + **live** collects dict +
edges + grad leaves), so `run_plan` (`plan.py:951`) lowers exactly once and
reuses the result for strictness, the gradient gate, and execution;
`lower_staged` (`staged.py:511`) survives as the structural test surface.
The one executor is `_execute_program` (`staged.py:730`) walking stages →
groups → `_run_trace_group` (`staged.py:671`; single-invoke fast path |
fused `_emit_invokes` with barrier | `_stop_carrier` early stop), with
`run_plan_staged` (`staged.py:772`) as the public executor entry. All four
named `plan.py` helpers died as designed (`_refuse_headroom` too), and
`_build_taps` (`plan.py:639`) additionally lost its `staged=`/`ops=`
dual-mode surface in EU3 — the retired generation path was its last
`staged=False` caller, so the superset build is the only build; the one
behavioral edge (read taps inside the generate trace now `.save()`
unconditionally) is pinned by a decode-step property test.

### Refusal relocation map

Refusals **relocate, never disappear**. Construction-time raises
(`Plan`/ops `__post_init__`, edit/modes/site/featurized_site, persistent.py
install gates) and trace-time raises (generate-emitter input-shape checks,
abandoned-body, gradient post-checks, write-broadcast) are **unchanged**.

| Today (exception as control flow) | After (scheduler fact; exception only under `lowering="single"`) — **shipped, final anchors** |
|---|---|
| `_cross_input_phases` — chained cross-input flow | layering rule (`consumes_in_trace`) — `staged.py:315` |
| `_check_barrier_schedulable` — produce after consume | `_rendezvous_conflict` force-stage + rerun — `staged.py:336` |
| `_frame_lengths` — mixed/unknowable padded frames | `frames_align` per-edge test — `staged.py:294` |
| single-trace model check — inputs on different models | same-model per-edge test — `staged.py:311` |
| `_build_taps(staged=False)` src>dst — same-input backward read | the `(input,"clean")` reroute becomes the only behavior — `plan.py:733` |
| generation cross-input reads (NotImplementedError) | **relaxed**: force-staged edges into the generate stage — `_generate_forcing`, `staged.py:433` |
| generation ≠1-input (NotImplementedError) | **narrowed**: *ops* address ONE input; other inputs are source-only stages — `_generate_invoke`, `staged.py:412` |
| per-step StagingRequired→ValueError wrapper | **dies** (unreachable; clean-pass semantics documented on GenerateSpec — `plan.py:300`) |
| `lowering="staged"` × generation (two ValueErrors) | **die** ("staged" = auto alias; a generation plan under it routes through the one scheduler like auto) |
| gradient refusals (staged path, multi-invoke, featurized-wrt, grad-mode-off) | one schedule-shape gate, messages verbatim ("single-input plans only" + measured rationale) — `_refuse_gradient_shape`, `plan.py:590` |

**Shipped as mapped (EU2 #483, EU3 #484); anchors above are exact in any
tree containing this annotation (see the Part 5 intro note).** Strict-mode
messages are assembled by `_refuse_not_single`
(`staged.py:520`), each retired refusal's key phrase preserved ("two
passes", "backward in time", "padded lengths", "pre-tokenized", the
chained-flow sentence, and EU3's "generate trace accepts only constants").
Surviving generate-path refusals, exact locations: the scheduler-side
`NotImplementedError` ("…ops must address ONE input…"); the model-free
input-shape `ValueError`s, fired by the executor *before any stage runs*
(`_check_generate_inputs`, `plan.py:1167` — "pre-tokenized" input,
`position_ids` × multi-step, `min_new_tokens` floor); and the post-trace
`RuntimeError` ("the generate trace body was abandoned…"). One kept refusal
beyond the map: `run_plan_staged`'s inline gradient refusal
(`staged.py:772`; pinned by `TestScheduleUnit`, unreachable via `run_plan`
routing).

### Unification sub-issues (EU0–EU7)

Stacked PRs into `nnterp-rebase`; {EU0, EU1, EU7} independent; EU2→EU3→EU4
serial spine; {EU5b, EU6} fan out after EU5a. *As landed*: the sub-issue PRs
merged into the integration branch `epic/480-engine-unification` (epic PR
#492), one merge commit per PR (the repo allows only merge commits) — the
handoff record for each lives in its PR's comments.

| ID | Sub-issue | Scope | Depends on | Landed |
|----|-----------|-------|-----------|--------|
| EU0 (#481) | Drive-by fixes | preflight `_build_taps` 4-tuple unpack (live test failure); attribution `_forward_inputs` import (live ImportError) | — | on the branch base (`e227a57a`, pre-integration) |
| EU1 (#482) | Session removal | plain sequential traces only; no-session policy + reintroduction gate in the `staged.py` docstring | — | PR #493 |
| EU2 (#483) | One scheduler/executor (plain plans) | `lower_plan` + `staged_why`, `_run_trace_group`, strictness routing, delete the four `plan.py` guards, gradient shape gate | EU1 | PR #495 |
| EU3 (#484) | Terminal generate stage | `generate_key`, `_generate_forcing`, `_emit_generate_trace`, relax generation refusals into scheduling | EU2 | PR #496 |
| EU4 (#485) | Reroute `run_intervened_generation` | Plan-building body; `Plan.models` cross-model; delete `_generate_with_edits`; legacy output kept | EU3 | PR #497 |
| EU5a (#486) | `GenerationResult` + producers | dataclass in `pipeline.py`; `compress_scores_top_k` | EU4 | PR #498 |
| EU5b (#487) | Consumer migration | interchange_mode / interpolate / steer / layer_scan / metric / io via `to_raw_results()` | EU5a | PR #500 |
| EU6 (#488) | Docs as-landed | annotate this Part with shipped state | EU5a | this annotation |
| EU7 (#489) | Delete the layer-skip ablation | remove the layer-skip module + its tests; scrub doc refs | — | PR #494 |

Verification rails (every PR): the raw-hook oracle suites, the prefill-hooked
HF-generate oracle, the step-counting generate oracle, and the leafy-forward
gradient oracle run **unmodified** before any test is rewritten; the
`lower_staged` structural pins survive EU2 unchanged and *extend* in EU3; the
WS1 walking skeleton gates the spine; golden repins are legal only for
shape-pinned goldens in EU5a/EU5b (values engineered identical throughout).

**Shipped: the rails held with ZERO golden repins through EU5a** — even the
anticipated shape-pinned rewrite (`TestChatCoherentGenerateContract`)
collapsed to a docstring truth-update, because the wrappers keep the legacy
dict via `to_raw_results()` and that test's dataset fits one batch. One
recurring gate lesson from the run: quoted suite totals went stale between
sub-issues three times — same-box A/B deltas (re-measure the base tip on the
same machine, same command) are the only trustworthy full-suite gate.

### Legacy boundary (successor: #491)

**Resolved (WU6, #508):** everything this section names is gone — the #491
epic landed the spec vocabulary (Part 6) and the WU6 sweep deleted the files,
the adapters, and the `UnitEdit` class. The paragraph below is kept as the
record of what the boundary was.

What the unification deliberately did NOT retire: the declarative
where-surface that predates the Plan IR. The deprecation surface, named —
`causalab/neural/units.py`, `causalab/neural/LM_units.py`, and
`causalab/neural/activations/targets.py` (the legacy grid builders);
`UnitEdit` (`causalab/neural/dataset.py`); and the two adapters that are the
only places the legacy vocabulary still touches the engine, `unit_site()`
(`dataset.py:91`) and `_unit_edit_to_edit` (`dataset.py:125`), both
docstring-marked as **the legacy boundary**. Below that boundary everything
speaks Plan IR; the EU series added no new or changed engine signature
taking an `AtomicModelUnit`/`InterchangeTarget`/`UnitEdit`. The
**where-surface merge (#491)** — retire `AtomicModelUnit`/`InterchangeTarget`
in favor of `Site`/`FeaturizedSite`/`Edit`/`Plan` — is the successor
umbrella that retires this surface, starting by deleting exactly those two
adapters.

---

## Part 6: where-unification (one declarative surface over the engine)

*Added 2026-07-07. Umbrella **#491**; design-of-record for the WU sub-issues.
Successor to Part 5 the way Part 5 succeeded Part 4: the engine below the
legacy boundary is finished (EU0–EU7 landed on the #480 epic branch), so the
declarative surface above it can now migrate onto the finished engine.
**Annotated as-landed (WU6, #508)**: WU1–WU6 all landed on the epic branch
(PRs #511, #512, #513, #515, #514, and the WU6 sweep PR); the **shipped**
notes below record where the landed state differs from the design. The
unannotated prose is the original design, kept as the record of intent.*

**Shipped deltas (as-landed record, WU6 #508):**

- **Golden key paths embed opaque `spec.key`s with dots inside.** The three
  WU5-repinned goldens (`weekdays_subspace` / `weekdays_path_steering` /
  `weekdays_pullback`) pin featurizer-bundle key paths of the form
  `…models.<cell>.featurizers.residual_stream.L1.block_output.last_token/…`
  — the segment after `featurizers.` is a `spec.key` and contains dots, so a
  golden key path is **no longer an unambiguous dot-split**; tooling over
  golden keys must treat the tail as opaque. Any future spec-key format
  change re-repins those three goldens (value-identical, key-rename-only).
- **`EditSpec` persistence was not shipped.** Bundles are SiteSpec-only
  (`save_site_specs`/`load_site_specs`); the "EditSpec round-trips iff its
  params are data" boundary is documented on the class. Settled: no wave
  WU2–WU6 needed it.
- **`flatten_groups`/`nest_like` were named by the design but shipped by no
  wave.** WU2's single-group guarantee makes `grid[key][0]` the direct
  successor of flattening; callers use plain comprehensions.
- **`ComponentIndexer` relocated, not retired.** It is `TokenPosition`'s
  base class and carries the #430 `is_original` signature detection; the
  sweep moved it verbatim from the deleted `units.py` into
  `causalab/neural/token_positions.py` (pins:
  `tests/neural/test_component_indexer.py`).
- **The legacy loader branch STAYS (census exception).** A WU6 census
  (2026-07-07) found live version-`"2.0"` `units_metadata.json` bundles
  outside this repo's regenerable outputs — a few thousand in an external
  experiment store plus ~112 in local session/artifact trees — so
  `load_site_specs`' `units_metadata.json` branch survives per the
  acceptance's exception clause, with its component table
  (`specs.py::_HEAD_COMPONENTS`) and a frozen-format fixture writer in
  `tests/neural/test_specs.py` pinning the compatibility contract.

### Why: two parallel "where" surfaces

Every layer below `causalab/neural/dataset.py` speaks Plan IR; everything
above it still speaks the pyvene-era declarative surface —
`AtomicModelUnit`/`ComponentIndexer`/`InterchangeTarget`
(`causalab/neural/units.py` + `causalab/neural/LM_units.py`), `UnitEdit`
(`causalab/neural/dataset.py`), and the `causalab/neural/activations/targets.py`
grid builders. The two adapters at the boundary (`unit_site()`,
`_unit_edit_to_edit`) keep the surfaces consistent, but the split has real
carrying costs, each verified in-tree:

- **Two vocabularies for one concept.** A unit names its location with a
  pyvene component string (`component_type="head_attention_value_output"`)
  that `unit_site()` must map onto the engine's `Site`/`HeadSite`; every new
  consumer chooses a side, and wrapper signatures (~20 public entry points)
  freeze the legacy side in place.
- **The unit id string is a de-facto wire format.** Grid detection
  substring-matches ids (`detect_component_type_from_targets`), plot axes are
  parsed out of ids (`causalab/io/plots/unit_id.py`), and the
  `LM_units.*.load_modules` loaders re-parse `Layer-N`/`Token-id` fields with
  hardcoded offsets (their own TODOs call the scheme brittle — and they have
  **zero** production callers).
- **Mutation-based feature-space management.** `set_featurizer` /
  `set_feature_indices` mutate units in place (train → attach → run flows in
  `causalab/methods/trained_subspace/train.py`,
  `causalab/analyses/subspace/loading.py`, the manifold pipelines), so which
  feature space a unit currently carries is a question about execution
  history, not about a value.
- **Duplicated position machinery.** `AtomicModelUnit.index_component` +
  `_apply_padding_shift` implement the unpadded→padded frame shift that the
  engine already made obsolete (`resolve_positions_batched` births positions
  in the padded frame — the shift is documented as legacy-only in the
  cross-cutting notes below).
- **A container with delegation magic.** `InterchangeTarget` is a
  list-of-lists that also fans out `save`/`load`/`set_featurizer`/
  `set_feature_indices` to its units — grouping semantics and persistence
  concerns in one class.

### What the legacy surface carries

Retiring the surface is not deleting an alias — four capabilities live only
on the legacy side today, plus the callers. Each needs the designed home
below before the deletion sweep:

| Capability | Legacy carrier | Live evidence |
|---|---|---|
| Serialization | `InterchangeTarget.save`/`load` (`units_metadata.json` + `featurizers.safetensors` via `causalab/io/nested_artifacts.py`), `Featurizer.to_dict`/`from_dict` | trained-subspace bundles: written by `causalab/io/artifacts.py` `save_training_artifacts` (`trained_target.save(...)`), read by `causalab/analyses/activation_manifold/loading.py` `load_featurizer` → `interchange_target.load(...)` |
| Position binding | `ComponentIndexer` bundled into the unit (`position_resolver`), `with_position_resolver` derived views | `resolve_unit_positions` (dataset.py); span helpers in `causalab/methods/ablation/_spans.py`, `causalab/methods/ablation/reference_vectors.py` |
| Grouping | `InterchangeTarget` outer index ↔ `example["counterfactual_inputs"][g]`; `flatten`/`nest_to_match` | `run_intervened_generation` groups; `nest_to_match` re-shapes per-unit results across the toolkit |
| Grid building | `causalab/neural/activations/targets.py` builders + grid-dim extraction; the three grouping modes (`one_target_all_units`/`per_unit`/`per_layer`) | layer/head scans in `causalab/methods/interchange/layer_scan.py`, `causalab/analyses/subspace/grid.py`, `causalab/analyses/ablation/main.py`, `causalab/runner/helpers.py` |
| Callers | unit-/target-typed signatures | ~45 files: methods/ (ablation, causal_tracing, steer, interchange, path_patching, trained_subspace, spline, pca, logit_lens, pullback), analyses/ (subspace family, activation_manifold, ablation, causal_sufficiency, exploration, path_steering), io/plots, runner, activations wrappers |

Out of scope on purpose: task packages' `token_positions.py` modules author
`TokenPosition` specs — that layer **stays** (only its base class relocates,
see Position binding); `causalab/neural/attention_probs.py` and
`causalab/methods/path_patching/targets.py` `ReceiverSpec` are already
engine-native.

### Target vocabulary: two frozen specs over the engine

Two new frozen dataclasses at the dataset altitude (the layer that owns
batching and position resolution), composing engine values instead of
paralleling them:

```
authoring   TokenPosition specs (causalab/neural/token_positions.py — unchanged)
                │
specs       SiteSpec  = FeaturizedSite + position spec + key (+ width)
            EditSpec  = SiteSpec + named mode + declarative params
                │           groups: Sequence[Sequence[EditSpec]]
                │           outer index g ↔ counterfactual_inputs[g] ↔ plan input cf_{g}
                ▼
dataset.py  resolve positions per batch side → mode constructors → CollectOp/EditOp → Plan
```

- **`SiteSpec(fsite: FeaturizedSite, positions, key: str, width: int | None)`**
  — the `AtomicModelUnit` successor. It *contains* the engine's
  `FeaturizedSite` (which already carries site + featurizer + `feature_ids`)
  rather than re-declaring location fields, so there is nothing to map: the
  `unit_site()` component-string table (`_HEAD_COMPONENTS`) has **no
  successor** — a spec is born holding a real `Site`/`HeadSite`. `positions`
  is a declarative position spec (any `PositionResolver`, e.g. a
  `TokenPosition`, or literal rows) or `None`; `key` is the explicit result
  key (collect outputs, saved bundles — replaces `unit.id`, and is **opaque**:
  nothing may parse it); `width` records the raw feature width the builders
  read from model config (the `shape` field's one real use — DAS/DBM rotation
  sizing, `causalab/methods/causal_tracing/vectors.py` `_n_features`).
  Frozen: featurizer attachment becomes functional —
  `with_featurizer(...)`/`with_feature_ids(...)`/`with_positions(...)` return
  new specs (`with_positions` replaces `with_position_resolver`, same
  shallow-view semantics). Train→attach→run flows return updated specs
  instead of mutating shared ones.
- **`EditSpec(site: SiteSpec, mode, vector, scale, seed, interpolate_fn,
  interpolate_params)`** — the `UnitEdit` successor, same five-mode
  vocabulary (`interchange`/`interpolate`/`replace`/`add`/`noise`) and the
  same construction-time validation. `_unit_edit_to_edit` becomes
  `_edit_spec_to_edit` with an identical body: still THE single spec→engine
  conversion point in `causalab/neural/dataset.py`, still routing through the
  `causalab/neural/modes.py` constructors — no longer a *legacy* boundary,
  just the spec lowering.
- **Groups are plain nested sequences.** `InterchangeTarget` has no successor
  class: `Sequence[Sequence[SiteSpec]]` (and `[[EditSpec]]`) with module
  functions `flatten_groups(groups)` / `nest_like(groups, flat)` replacing
  `flatten`/`nest_to_match`, and mapping helpers replacing the mutating
  fan-outs (`set_featurizer` → map `with_featurizer` over a group). The
  delegation magic dies; persistence moves to the serialization functions
  below.

### Serialization: named specs round-trip; callables do not

The engine stays deliberately non-serializable (`Edit.g` is an arbitrary
callable). What round-trips is the **spec layer**, and only its named subset:

- **New bundle format** — `save_site_specs(specs, dir)` /
  `load_site_specs(dir, token_positions=None)`: one `sites.json` (per spec:
  `key`, structured site record — component/layer, plus head + kind for
  `HeadSite` — `feature_ids`, position spec **name**, `width`, format
  version) plus `featurizers.safetensors`/`featurizers.meta.json` written via
  the existing `causalab/io/nested_artifacts.py` `save_nested` path,
  `Featurizer.to_dict`/`from_dict` reused unchanged. JSON + safetensors only —
  the new path adds **no** `torch.save`/pickle reader.
- **Loads are constructive.** `InterchangeTarget.load` mutated units the
  caller had to pre-build to matching ids; `load_site_specs` *returns* specs.
  Site, featurizer, `feature_ids`, `key`, `width` restore fully from bytes; a
  position restores by name and binds when the caller passes its
  `token_positions` mapping (the same rebinding contract
  `LM_units.*.load_modules` had), else stays `None` with the name kept in the
  record.
- **`EditSpec` round-trips iff its params do**: the named mode plus tensor /
  scalar params (`vector` rides the same safetensors payload; `seed` is an
  int). `interpolate_fn` — a callable — does not serialize; neither does any
  hand-built `Edit`. That boundary is the design: *named modes are specs;
  arbitrary `g` is code.*
- **Artifact-loader shim.** `load_site_specs` gets a legacy branch reading
  `units_metadata.json` bundles (version `"2.0"`), translating
  `component_type` through the retiring `unit_site()` table — the table's one
  surviving home. This keeps every existing trained-subspace bundle loadable
  through the migration. The **dead** legacy readers ship no successor:
  `LM_units.*.load_modules` and `Featurizer.save_modules`/`load_modules`
  (the per-file `torch.save` format; zero production callers) are deleted in
  the sweep outright — one less pickle reader.

### Position binding

The authoring layer stays; the binding moves from *inside the where-object*
to *a field on the spec*:

- `causalab/neural/token_positions.py` (`Template`, `TokenPosition`,
  `build_token_positions`, `paired_token_position`, `combined_token_position`)
  and the task packages' `create_token_positions` are **unchanged**.
  `ComponentIndexer` — `TokenPosition`'s base, today defined in
  `causalab/neural/units.py` — relocates to
  `causalab/neural/token_positions.py` in the sweep (it carries the
  `is_original` signature detection, #430); `SiteSpec.positions` types
  against the `PositionResolver` protocol (`causalab/neural/positions.py`),
  so any resolver or literal rows bind without inheriting from anything.
- Resolution stays where the engine put it: `resolve_spec_positions(spec,
  traces, encoding, is_original=...)` (the renamed `resolve_unit_positions`)
  → `resolve_positions_batched`, positions born in the batch's padded frame,
  per side (`is_original` routes paired base/counterfactual resolvers).
- `index_component` + `_apply_padding_shift` (and `AttentionHead`'s
  two-axis `[head_axis, position_axis]` return) die with the units: the
  engine resolves sequence-axis positions only — the head axis lives on
  `HeadSite`, where it belongs.

### Grouping: Plan-input naming is the contract

`InterchangeTarget`'s outer index already *is* the engine contract since EU4;
retiring the class makes it explicit rather than implicit:

- Group `g` of `groups` reads its sources from
  `example["counterfactual_inputs"][g]`, which
  `run_intervened_generation` binds to plan input `cf_{g}` (cross-input
  `ReadSource`s force-staged into one collect stage per group — the EU3/EU4
  machinery). The key scheme moves from an inline f-string to an exported
  `cf_input_key(g)` next to the run entry, documented as the grouping
  contract: `len(groups)` must match `len(counterfactual_inputs)` on every
  example, exactly as `InterchangeTarget` demanded.
- Path patching's two-group shape (`[[sender], [restorer]]` with
  `counterfactual_inputs=[cf, base]`) and the multi-site-one-source shape
  (`[[unit_a, unit_b]]`) carry over verbatim as nested spec lists.

### Grid builders (the targets.py successor)

`causalab/neural/activations/targets.py` is rebuilt on specs (new module,
old one deleted in the sweep):

- `build_residual_stream_sites` / `build_mlp_sites` /
  `build_attention_output_sites` / `build_attention_head_sites`
  → `dict[key_tuple, list[list[SiteSpec]]]` with the **same key tuples**
  (`(layer, pos.id)` / `(layer, head)`) and the same three grouping modes —
  scans, grids, and `causalab/io/plots/score_heatmap.py` keep their axis
  semantics. Config-derived widths (hidden_size / intermediate size /
  head_dim probing) move into the builders unchanged.
- Grid detection becomes structural: `grid_component(...)` reads
  `site.component` / `HeadSite.kind` off the first spec —
  `detect_component_type_from_targets`'s substring matching on ids dies, and
  `extract_grid_dimensions_from_targets` ports as-is (it already reads only
  the dict keys). `causalab/io/plots/unit_id.py`'s id parsing is replaced by
  the structured records (keys + site fields); after the sweep **nothing in
  the tree parses a where-identifier out of a string**.

### Migration policy

- **Stacked waves on an integration branch**, the #480 pattern: sub-issue PRs
  merge into `epic/491-where-unification` (epic PR #509); the epic PR lands
  on `nnterp-rebase` only after the deletion sweep, so the merged result
  never ships a deprecated path. The prerequisite — the #480 epic PR (#492),
  whose finished engine the specs compose — merged 2026-07-07.
- **Shims are inter-wave scaffolding, not a compatibility layer.** During
  waves (c)–(d) the wrappers accept both forms (a boundary coercion:
  unit/target input → specs via `unit_site()` + spec construction, with a
  `DeprecationWarning`); the sweep deletes the coercions together with the
  surface. Nothing outside the epic branch ever depends on a shim.
- **Key policy.** Collect results and bundle records key on `spec.key`
  (unique, explicit, opaque). Stored artifacts keyed by legacy `unit.id`
  strings regenerate under the new keys; consumers carry keys through instead
  of re-deriving them. Golden repins are legal only where a golden pins the
  key strings themselves (value-identical, key-rename-only) — no numerical
  change anywhere in this Part.
- **Equivalence before migration.** Wave (a) lands spec↔unit equivalence
  tests — `spec.fsite == unit_site(unit)` and resolved-position equality
  across the component × position matrix — so waves (c)–(d) migrate callers
  against a pinned translation, and the oracle suites
  (`tests/neural/activations/test_*_hook_oracle.py`, the parity suite) run
  unmodified until the wave that migrates their subject.

### Where-unification sub-issues (WU1–WU6)

Filed per the wave-open convention: one-line scopes now, expanded to
acceptance criteria before implementation. Stacked PRs into
`epic/491-where-unification`; WU1 is the spine head; {WU2, WU3} fan out after
WU1; {WU4, WU5} fan out after both; WU6 last.

| ID | Sub-issue | Scope | Depends on |
|----|-----------|-------|-----------|
| WU1 (#503) | Spec vocabulary + serialization | `SiteSpec`/`EditSpec` (frozen, functional updates); `save_site_specs`/`load_site_specs` (JSON + safetensors); legacy `units_metadata.json` loader branch; spec↔unit equivalence tests | #502 (design signoff; #492 merged 2026-07-07) |
| WU2 (#504) | Grid builders on specs | `targets.py` successor: `build_*_sites`, structural grid detection, same key tuples and grouping modes | WU1 |
| WU3 (#505) | Wrapper signatures → specs | `dataset.py` entries + `activations/` wrappers + method run-entries take specs; boundary coercions with `DeprecationWarning`; collect keys = `spec.key`; `cf_input_key` exported | WU1 |
| WU4 (#506) | Caller migration: methods/ | ablation, causal_tracing, steer, interchange (layer_scan/attribution/single_pair), path_patching senders, trained_subspace, spline, pca, logit_lens, pullback | WU2, WU3 |
| WU5 (#507) | Caller migration: analyses/ + io + runner | subspace family, activation_manifold, exploration, path_steering, ablation/causal_sufficiency mains, score_heatmap + unit_id plots, runner helpers, `save_training_artifacts` → new bundle | WU2, WU3 |
| WU6 (#508) | Deletion sweep | `units.py`, `LM_units.py`, `UnitEdit`, `unit_site`, `_unit_edit_to_edit`, `targets.py`, boundary coercions, id-parsing; `ComponentIndexer` → `token_positions.py`; legacy loader branch per artifact census | WU4, WU5 |

Verification rails (every PR): the raw-hook oracle suites and the
chat-coherent parity suite run unmodified until the wave that migrates their
subject; WU1's equivalence tests gate every signature/caller PR; golden
repins only for pinned key strings (value-identical). The legacy loader
branch in WU6 deletes only after a shared-filesystem census finds no live
`units_metadata.json` bundles outside regenerable outputs — if any remain,
the branch stays and WU6 records the exception as a follow-up issue.

---

## Cross-cutting migration notes

- **Replace, not dual-backbone.** causalab does not ship a runtime backend flag or
  maintain pyvene and nnsight/nnterp in parallel. `causalab/neural` is rewritten on
  nnsight/nnterp as a straight replacement, gated by the backbone-agnostic
  hook-oracle harness (`tests/neural/activations/hook_oracle.py`, coverage map
  `docs/PYVENE_HOOK_COVERAGE.md`) — the oracle computes ground truth from raw
  `register_forward_hook` and never imports the backbone, so the same tests re-run
  unchanged and confirm the new backbone reproduces the same activations/logits.
  pyvene stays installed only while unmigrated modules still import it and is
  deleted at cutover (SH2); `main` keeps pyvene until the branch merges, so no user
  is broken.
- **Declarative → imperative.** The biggest change is not any single row but the
  control model: causalab currently *builds a config* and hands it to pyvene;
  on nnsight it would *emit a trace body*. The `units`/`targets`/`Featurizer`
  abstractions (the Functionality column) can stay; the `activations/` runners and
  the pyvene-subclass featurizer machinery would be rewritten as trace emitters.
- **Per-head value/query is the one real capability gap (✗).** pyvene exposes
  `head_value_output` / `head_query_output` as first-class components (GQA-aware);
  nnsight/nnterp require manual reshaping of the attention tensors and `.source`
  AST-rewrites for pre-attention vectors. *Confirmed against `~/nnterp` 1.3.0
  source*: `standardized_transformer.py` / `rename_utils.py` expose only
  whole-sublayer `attentions_output[i]` plus optional `attention_probabilities[i]`
  — no per-head value/query accessor. Path patching (which depends on these
  receivers) is where a migration would cost the most.
- **Several pyvene-specific contracts simply disappear** under nnsight: the
  `sorted_keys` collect-order contract, the `use_fast` static-index path, the
  closure/subclass-per-mode featurizer pattern, and manual GPU teardown all become
  non-issues (explicit ordering, plain indexing, direct module calls,
  auto-freed traces).
- **nnterp buys cross-architecture uniformity + validation**, the closest fit to
  causalab's per-component addressing and its verification culture (load-time tests
  that edits actually affect output). But it has **no grad/attribution/patching
  helpers** (drop to raw nnsight) and **no per-head value/query accessors**.
- **Version/dependency friction.** nnterp requires `nnsight>=0.6` (confirmed in
  `~/nnterp/pyproject.toml`); causalab's floor is `nnsight>=0.5.9`, so adopting
  nnterp bumps the floor. The checkout leaves `transformers` **unpinned**, but its
  load-time validation is `transformers`-version-sensitive — land whatever version
  keeps the `causalab/neural` hook-oracle harness green. pyvene 0.1.8 stays
  installed until cutover but is never run as a parallel backend, so its
  `transformers` needs constrain only the transition window, then drop out at SH2.
- **Attention-implementation policy (parity vs. speed).** `LMPipeline` currently
  forces **eager** attention (a pyvene head-hook requirement); SDPA changes logits
  numerically. All migration parity and goldens therefore run in **eager** —
  apples-to-apples against the captured goldens — and the eager→SDPA default flip
  (+ re-enabling `use_cache` in generation, + a deliberate same-GPU golden repin)
  is its own post-cutover step, **SH3**. Without this policy the predictable
  failure mode is golden drift mid-migration that gets "fixed" by loosening
  tolerances. The flip is a real payoff: every non-attention-prob analysis gets
  SDPA/flash throughput that pyvene structurally forbade.
- **`tracer.cache` is off-limits under nnterp renaming.** nnterp's own test suite
  skips it ("Cache is not supported yet due to a nnsight renaming issue"). All
  collection is hand-rolled per-site `nnsight.save(...)` (+ `tracer.stop()`
  early-exit) or nnterp's collection helpers — never `tracer.cache`.
- **Loss-path ownership.** The differentiable scoring slice (label-concat forward
  + logits slice + CE — today `LM_loss_and_metric_fn`) lands once, with **ED3**,
  because training (Wave 7) needs it before the scoring adapter (MX1, Wave 9)
  exists. MX1/MX2 consume it. Two implementations of answer-scoring semantics
  drifting apart is the failure mode this rule prevents.
- **Remote (NDIF) is an escape hatch, not a default.** `remote=True` runs on
  hosted GPUs but sends the intervention graph + inputs to shared infra and serves
  only public models — keep private models local (see
  `NNsight_overview.md §4.2`).

## Upstream candidates (fold into nnterp)

Some functionality causalab builds on top of nnterp is **generic, cross-architecture
model access** — nnterp's own mission, not causalab-specific logic. Those pieces should
eventually be contributed upstream so causalab's surface shrinks and the wider ecosystem
benefits; the causal-abstraction layer (which head/site maps to which causal variable)
always stays in causalab. There is already an upstream relationship to lean on — the
packaging fix (ndif-team/nnterp#49) and the fork/migration tracked in #415.

- **Per-head value / query / attention-output accessors (`HeadView`, `causalab/neural/head_view.py`).**
  Today nnterp exposes attention only at whole-sublayer granularity
  (`attentions_output[i]`, optional `attention_probabilities[i]`); the per-head value
  (KV-head space), query (query-head space), and attention-output-sender vectors are
  reshaped in causalab. This is the archetypal nnterp accessor — architecture-spanning,
  reshape-only, honours the true `head_dim`, and has a clean per-family precedent in
  nnterp's `attention_probabilities` + `AttnProbFunction` callback (and its
  `check_source`-gated `.source` path for the fused-QKV case). It would slot in as e.g.
  `head_values[i]` / `head_queries[i]` / `head_attention_values[i]`. The contract is
  already pinned by `tests/neural/test_head_view.py` (coupled / GQA / decoupled-`head_dim`
  + a real Qwen3 golden), so upstreaming is a mechanical lift-and-shift with those tests
  as the executable spec. Until then `HeadView` is the causalab-side shim.

## Sources

- pyvene column: causalab source (`causalab/neural/`) + `docs/PATH_PATCHING.md`
  (code-verified, cited inline).
- nnterp rows & Part 2: **code-verified** against the local checkout `~/nnterp` @
  `7dbe4da` (~v1.3.0) — `standardized_transformer.py`, `rename_utils.py`,
  `interventions.py`, `prompt_utils.py`, `nnsight_utils.py`, `display.py`,
  `__main__.py`.
- raw-nnsight rows: [`docs/NNsight_overview.md`](NNsight_overview.md) (reference
  compiled from official sources; **not** code-verified — validate before
  implementation).
- Functionality column: the data-flow walkthrough in
  [`causalab/neural/README.md`](../causalab/neural/README.md).
