# Test migration map — protocol refactor

Auditable old-test → new-test mapping for the protocol refactor (base `a50637c` →
the deletion commit `c05b308`). Every test file that existed at the base appears
exactly once below, in one of four categories:

- **re-driven oracle** — oracle/behavioral tests whose assertions were kept
  verbatim and re-driven through protocol documents (same tolerances, same
  ground truth: the raw-hook oracle, moved intact to
  `tests/neural/pytorch_hooks/hook_oracle_lib.py`).
- **replaced interface** — tests bound to a deleted interface (hydra configs,
  runner, methods/analyses wrappers, the Plan API) replaced by
  protocol-equivalent tests of the same behavior.
- **kept frozen** — trees that keep passing unchanged (`tests/causal`,
  `tests/io`, `tests/tasks`, `tests/_helpers`), including files with the minimal
  rewires the deletion forces (import moves, the `PipelineShim` fixture swap).
- **retired** — no protocol expression in v1; each row carries the honest
  rationale.

**Baseline** (at `a50637c`): 3205 tests, 12 pre-existing failures, all in
`tests/runner/test_run_exp_dispatch.py`. **Outcome** (post-cut, `c05b308`):
1333 passed, 0 failed — the kept frozen trees plus the new-layer tests in
`tests/protocol/` and `tests/neural/pytorch_hooks/`.

Path conventions: the *old file* column is relative to the table's area; *new
file* paths are relative to `tests/`. Parametrized data dirs (`configs/`,
`goldens/`) and `__init__.py` package markers are grouped as single rows.

## tests/neural

| old file | classification | disposition | note |
|---|---|---|---|
| `__init__.py`, `activations/__init__.py`, `parity/__init__.py` (3) | kept frozen | kept / retired | `tests/neural/__init__.py` kept; the `activations/` and `parity/` markers left with their packages |
| `conftest.py` | replaced interface | `neural/pytorch_hooks/conftest.py` | session `LMPipeline` fixture → `ModelBundle` fixtures over the reference backend's loader |
| `activations/conftest.py` | replaced interface | `neural/pytorch_hooks/conftest.py` | the llama/gpt2/gqa tiny-random oracle-pipeline fixtures rebuilt as `bundle` + `OracleShim` |
| `activations/hook_oracle.py` | kept frozen | `neural/pytorch_hooks/hook_oracle_lib.py` (moved, R096) | the raw-hook oracle primitives moved intact; only deleted-type annotations trimmed |
| `activations/test_collect.py` | replaced interface | `neural/pytorch_hooks/test_read_oracle.py`, `test_run_corpus.py::test_01_harvest_runs` | `collect_features` shape/ordering/routing contract is now the document `reads` table + save path |
| `activations/test_collect_hook_oracle.py` | re-driven oracle | `neural/pytorch_hooks/test_read_oracle.py` | routing (`::test_reads_match_oracle_captures`), per-component (`::test_every_component_matches_oracle`), head slice (`::test_head_value_read_matches_oracle`); verbatim tolerances atol=1e-5 rtol=1e-4 |
| `activations/test_cross_model_hook_oracle.py` | retired | retired | cross-model/two-pipeline patching is out of v1 — one model per document; see gaps |
| `activations/test_feature_space_hook_oracle.py` | re-driven oracle | `neural/pytorch_hooks/test_write_oracle.py::test_feature_space_swap_keeps_the_complement` | plus `::test_dims_swap_is_a_subspace_swap`; the `base_err` complement-preservation assertions verbatim |
| `activations/test_interchange_mode.py` | replaced interface | `neural/pytorch_hooks/test_write_oracle.py::test_interchange_matches_oracle`, `test_run_corpus.py::test_02_interchange_runs_and_scores` | the `run_interchange_interventions` wrapper died; same-model interchange as a document (cross-model: see gaps) |
| `activations/test_interpolate.py` | replaced interface | `neural/pytorch_hooks/test_parity_goldens.py` (lerp cases) | WU3 wrapper-signature test; the wrapper died, `lerp` is the protocol spelling |
| `activations/test_interpolation_hook_oracle.py` | re-driven oracle | `neural/pytorch_hooks/test_parity_goldens.py::test_protocol_stack_replays_captured_golden` (interpolate→lerp cases) | oracle-captured lerp values replayed verbatim; the arbitrary-`fn` generality retired — v1 spells only `lerp` |
| `activations/test_noise_hook_oracle.py` | re-driven oracle | `neural/pytorch_hooks/test_write_oracle.py::test_gaussian_contract`, `::test_gaussian_draw_realization` | seeded-draw contract plus the exact RNG realization asserted byte-for-byte |
| `activations/test_site_grids.py` | replaced interface | `protocol/test_sweep.py`, `protocol/test_corpus.py` (corpus 07 pins) | grid builders became explicit `{sweep: …}` axes; the 64-point locate grid expansion is pinned |
| `parity/cases.py` | retired | retired | capture-harness registry; the replay inlines the frozen capture recipe constants in `test_parity_goldens.py` |
| `parity/conftest.py` | retired | retired | harness-only session fixtures, superseded by `neural/pytorch_hooks/conftest.py` |
| `parity/goldens/` (`gpt2.json`, `gqa.json`, `llama.json`) | kept frozen | kept in place | the frozen pre-migration numerical pins — replayed by `neural/pytorch_hooks/test_parity_goldens.py` |
| `parity/pins.py` | retired | retired | pin extraction/loading helper; loading is inlined in `test_parity_goldens.py` |
| `parity/test_captured_goldens.py` | re-driven oracle | `neural/pytorch_hooks/test_parity_goldens.py::test_protocol_stack_replays_captured_golden` | every portable pinned case verbatim (tol 1e-4, per-key overrides, exact shapes); 2 gqa `head_value` pins explicitly skipped (`::test_every_pinned_case_is_ported_or_skipped` audits the skip table) — see gaps |
| `parity/test_chat_coherent_parity.py` | retired | retired | the GPU-nightly coherent-backbone tier retired with the analyses; needs a protocol-native golden-tier decision — see gaps |
| `parity/test_mode_parity.py` | re-driven oracle | `neural/pytorch_hooks/test_read_oracle.py`, `test_write_oracle.py` | the live oracle-equivalence sweep, re-expressed per contract (read routing, write semantics) through documents on the same oracle |
| `parity/update_goldens.py` | retired | retired | pin updater for the old harness; the goldens are deliberately frozen — recapturing from the new stack would defeat the anchor |
| `test_attention_probs.py` | retired | retired | CAP4 attention-prob writes: the `attention_probs` component survives in §2.4 but the reference backend refuses it (no `writable_attention_probs` capability); only routing is pinned (`protocol/test_backend_routing.py::test_requires_writable_attention_probs`) |
| `test_component_indexer.py` | kept frozen | kept | unchanged; `ComponentIndexer` backs the kept `token_positions` task semantics |
| `test_dataset.py` | replaced interface | `neural/pytorch_hooks/test_read_oracle.py`, `test_write_oracle.py`, `test_run_corpus.py` | PL3 batched engine deleted; execution semantics re-pinned against the same raw-hook oracle, end-to-end via the CLI |
| `test_edit.py` | replaced interface | `neural/pytorch_hooks/test_write_oracle.py` | ED1 `Edit` API died; write semantics (incl. read-sees-write, `::test_reads_see_the_fully_written_state`) pinned on documents |
| `test_featurized_site.py` | replaced interface | `neural/pytorch_hooks/test_write_oracle.py` (feature-space + error-term tests) | ST3 wrapper died; featurizer/error-term threading pinned vs the oracle |
| `test_featurizer.py` | replaced interface | `neural/pytorch_hooks/test_parity_goldens.py` (sub3/mask cases), `test_run_corpus.py::test_fit_then_apply_roundtrip` | base `Featurizer`/identity/serialization → featurizer stages (`LoadedLinear` = the old `SubspaceFeaturizerModule` math) + artifact round-trip |
| `test_head_view.py` | replaced interface | `neural/pytorch_hooks/test_read_oracle.py::test_head_value_read_matches_oracle`, `test_parity_goldens.py` (gqa family) | head slicing / GQA addressing is backend-internal now; per-head o_proj-input reads pinned vs the oracle |
| `test_modes.py` | replaced interface | `neural/pytorch_hooks/test_parity_goldens.py` (mode→`do` mapping), `test_write_oracle.py` | ED2 constructors died; each old mode's numbers replayed through its documented `do` spelling |
| `test_persistent.py` | retired | retired | CAP7 persistent `model.edit()` lifecycle has no v1 spelling — documents execute stateless intervened models |
| `test_pipeline.py` | replaced interface | `neural/pytorch_hooks/test_positions_frame.py`, `tests/_helpers/pipeline_shim.py` | `LMPipeline` deleted; encode/left-pad batching re-pinned on `PositionFrame`; generation is out of v1 (metrics lower from saved logits) |
| `test_plan.py` | replaced interface | `protocol/test_validation_rules.py`, `protocol/test_corpus.py` | the Plan IR became the document; the refusal-ordering contract became the load-error checklist (one failing doc per rule) |
| `test_positions.py` | replaced interface | `neural/pytorch_hooks/test_positions_frame.py` | ST2 left-pad shift math and the #176 stale-position refusal (`::test_out_of_bounds_refuses`) re-pinned on `PositionFrame` |
| `test_preflight.py` | replaced interface | `protocol/test_validation_rules.py`, `neural/pytorch_hooks/test_positions_frame.py` | CAP5 `scan()`'s model-free gates became load-time validation rules; position bounds fail legibly at resolve time |
| `test_site.py` | replaced interface | `neural/pytorch_hooks/test_read_oracle.py::test_every_component_matches_oracle`, `test_write_oracle.py` | ST1 Site API died; the component vocabulary is spec §2.4, read/write parity pinned per component vs the oracle |
| `test_specs.py` | replaced interface | `protocol/test_schema.py` | WU1 spec vocabulary/serialization → strict document parse (sugar, aliases, wrapper shapes) |
| `test_staged.py` | replaced interface | `neural/pytorch_hooks/test_write_oracle.py::test_two_pass_path_patching_matches_oracle`, `protocol/test_corpus.py` | PL2 multi-trace scheduling is now the derived execution shape, pinned per corpus document |
| `test_token_positions.py` | replaced interface | kept task suites via `tests/_helpers/pipeline_shim.py` | `neural/token_positions.py` is KEPT; its `LMPipeline`-bound suite is replaced by the frozen task trees driving the same factories through `PipelineShim` |
| `test_trainable.py` | replaced interface | `neural/pytorch_hooks/test_train.py` | ED3 DAS/DBM primitives → the train loop: seeded determinism, params-moved-only, gate anneal, objective floor |
| `test_validate.py` | retired | retired | the nnterp load-validation gate left with nnterp; loading is `pytorch_hooks.load_model` against the registry, no CLI gate in v1 |
| `test_walking_skeleton.py` | replaced interface | `neural/pytorch_hooks/test_end_to_end_iia.py` | the task-driven IIA pin, restored over a serialized task table (corpus 10) once the dataset seam existed; values captured fresh (see gaps) |
| `walking_skeleton_pins.json` | retired | retired | its successor pins live in the test module above, not a sidecar |

## tests/methods

| old file | classification | disposition | note |
|---|---|---|---|
| `__init__.py`, `interchange/__init__.py`, `path_patching/__init__.py`, `spline/__init__.py` (4) | retired | retired | package markers, deleted with the tree |
| `interchange/test_attribution.py` | retired | retired | gradient×Δ attribution pre-scan has no v1 document spelling; gradients survive only for `train` (spec §8) |
| `interchange/test_layer_scan.py` | replaced interface | `protocol/test_sweep.py`, `protocol/test_corpus.py` (corpus 07) | the layer scan is a sweep axis; deterministic expansion + the locate-grid shape pinned |
| `interchange/test_single_pair.py` | replaced interface | `neural/pytorch_hooks/test_positions_frame.py::test_out_of_bounds_refuses` | the #176 contract carried verbatim in spirit: a stale position refuses, never addresses the wrong token |
| `path_patching/conftest.py` | retired | retired | Plan-era fixtures (single-step pipelines, pyvene hook-cleanup workarounds) |
| `path_patching/test_head_receivers_hook_oracle.py` | retired | retired | per-head value (v_proj-output, KV-head space) and query receivers have no §2.4 component — `attention_premix` is the o_proj input; flagged at the gate (the parity skip table names it); see gaps |
| `path_patching/test_outputs.py` | replaced interface | `neural/pytorch_hooks/test_metrics.py` | the Plan-logits→`GenerationResult` adapter died; last-position slice/argmax/decode live in metric lowering |
| `path_patching/test_plans.py` | replaced interface | `protocol/test_corpus.py` (corpus 03), `tests/protocols/03_path_patching_im.json` | the pass structure and receiver wiring are explicit authored document content, pinned by digest + execution shape |
| `path_patching/test_receiver_set.py` | retired | retired | multi-receiver documents are expressible (several inject edits) but the set-degeneracy invariant has no dedicated new test; per-receiver two-pass semantics are oracle-pinned |
| `path_patching/test_run.py` | replaced interface | `neural/pytorch_hooks/test_run_corpus.py::test_03_path_patching_runs`, `test_write_oracle.py::test_two_pass_path_patching_matches_oracle` | regime wiring end to end through the real CLI; the scored path oracle-pinned |
| `path_patching/test_targets.py` | retired | retired | the restorer-set *derivation* died with the method; in the protocol the restorer set is explicit authored edits (corpus 03's frozen off-path attention) |
| `path_patching/test_two_pass_hook_oracle.py` | re-driven oracle | `neural/pytorch_hooks/test_write_oracle.py::test_two_pass_path_patching_matches_oracle` | v* collected under the sender swap, injected clean — vs the same two-pass hook oracle, wrapper tolerances atol=1e-4 rtol=1e-3 |
| `spline/test_cubic.py` | retired | retired | pins the spline library of the deleted manifold analysis chain; no protocol expression in v1 — flagged at the gate |
| `spline/test_cubic_vs_tps_periodic.py` | retired | retired | on-demand benchmark comparison of the deleted spline backends |
| `spline/test_featurizer.py` | retired | retired | rank-polymorphism shim of the deleted spline manifold featurizer |
| `spline/test_manifold.py` | retired | retired | `SplineManifold`/`ThinPlateSpline` deleted with the manifold chain |
| `spline/test_pca.py` | retired | retired | PCA parameterization of the deleted spline manifolds |
| `spline/test_periodic.py` | retired | retired | periodic modes of the deleted spline manifolds |
| `test_ablation.py` | replaced interface | `neural/pytorch_hooks/test_run_corpus.py::test_06_hydra_effect_runs`, `test_parity_goldens.py` (replace cases) | resample-ablation runs as corpus 06; zero/mean ablation is a constant `swap` (the pinned replace mode) |
| `test_attention_pattern_analysis.py` | retired | retired | attention-pattern extraction/figures died with the analyses; `attention_probs` reads survive in the vocabulary but the reference backend refuses them (routing pinned in `protocol/test_backend_routing.py`) |
| `test_causal_tracing.py` | replaced interface | `neural/pytorch_hooks/test_run_corpus.py::test_06_hydra_effect_runs`, `test_write_oracle.py` (gaussian tests) | the ROME-style corrupt+restore method → the corpus 06 clean/ablated document pair; seeded-noise semantics oracle-pinned |
| `test_causal_tracing_hook_oracle.py` | re-driven oracle | `neural/pytorch_hooks/test_write_oracle.py::test_gaussian_contract`, `::test_gaussian_draw_realization`, `::test_reads_see_the_fully_written_state` | the RNG-independent draw contract + the mixed-pass ordering (a read sees the fully edited state, §2.7) |
| `test_comparison_fns.py` | replaced interface | `neural/pytorch_hooks/test_metrics.py` | distribution comparisons re-pinned as exact formulas on hand-built logits (`::test_kl_of_identical_distributions_is_zero` etc.) |
| `test_composition.py` | replaced interface | `neural/pytorch_hooks/test_parity_goldens.py` (mask = `["rot","gate"]` case) | the `>>` operator died; composition is the ordered featurizer stack, numbers replayed verbatim |
| `test_distances.py` | retired | retired | cyclic costs / Hellinger / Wasserstein backed the deleted manifold chain; no protocol expression in v1 |
| `test_dual_manifold.py` | retired | retired | dual-manifold viewer integration of the deleted chain |
| `test_edit_training.py` | replaced interface | `neural/pytorch_hooks/test_train.py::test_dbm_fit_trains_theta_and_anneals_temperature` | the ED3 outer loop's temperature schedule pinned through the real fit |
| `test_filter.py` | retired | retired | correct-only dataset filtering was runner data-prep; the protocol consumes prepared splits (content digests pin what enters a run) |
| `test_first_tokens.py` | replaced interface | `neural/pytorch_hooks/test_metrics.py::test_space_prefixed_first_resolution` | the two-form token expansion became the strict space-prefixed-first `column_token_id` contract with a multi-piece refusal |
| `test_interchange.py` | replaced interface | `neural/pytorch_hooks/test_run_corpus.py::test_02_interchange_runs_and_scores`, `test_metrics.py` | `causal_score_intervention_outputs` → metric lowering over saved logits + IIA scoring in the corpus run |
| `test_interpolate.py` | replaced interface | `neural/pytorch_hooks/test_parity_goldens.py` (lerp cases) | `FeatureInterpolateIntervention` died; `lerp` is the spelling, oracle-captured values replayed |
| `test_isometry_metric.py` | retired | retired | isometry metric of the deleted manifold chain |
| `test_metric.py` | replaced interface | `neural/pytorch_hooks/test_metrics.py` | #208 token-form alignment → the `column_token_id` contract (spec §2.10), exact formulas pinned |
| `test_multi_token.py` | replaced interface | `neural/pytorch_hooks/test_metrics.py::test_class_probs_sums_group_members` | multi-token class probabilities → `class_probs` group sums |
| `test_output_tokens.py` | replaced interface | `tests/tasks/natural_domains_arithmetic/test_natural_domains_arithmetic.py` (form_groups), `neural/pytorch_hooks/test_metrics.py` | the resolver split: `form_groups` moved to `causal/causal_utils.py` (kept-tree coverage), token ids to metrics |
| `test_standardize.py` | retired | retired | the affine standardize round-trip survives as `featurizers.Standardize` but no v1 corpus document uses it and no dedicated new test exists — honest gap in stage coverage |
| `test_steer.py` | replaced interface | `neural/pytorch_hooks/test_write_oracle.py::test_add_scaled_matches_oracle_steer`, `::test_renormalize_restores_the_pre_write_norm` | steering is `add_scaled` (+ optional renormalize); oracle-pinned, parity steer cases replay the old numbers |
| `test_steer_heads.py` | replaced interface | `neural/pytorch_hooks/test_parity_goldens.py` (head cases), `test_positions_frame.py::test_span_is_a_content_frame_window` | head-unit addressing is a site `head` field; multi-position spans are `span` positions |
| `test_steer_manifold.py` | retired | retired | manifold steering of the deleted chain |
| `test_train.py` | replaced interface | `neural/pytorch_hooks/test_train.py` | the `train_interventions` public API died; the document `train` section drives the same fit mechanics |
| `test_train_intervention.py` | replaced interface | `neural/pytorch_hooks/test_train.py::test_das_fit_moves_only_the_rotation_and_is_seeded`, `::test_das_fit_reduces_its_own_objective` | loop orchestration → seeded, param-scoped, objective-reducing fits |
| `test_trained_subspace.py` | replaced interface | `neural/pytorch_hooks/test_run_corpus.py::test_fit_then_apply_roundtrip`, `test_parity_goldens.py` (sub3) | projection round-trip + serialization → the 04→09 fit→apply `ArtifactIdentity` cycle; `LoadedLinear` is the same math |

## tests/analyses

| old file | classification | disposition | note |
|---|---|---|---|
| `__init__.py` ×6 (analyses + 5 subpackages) | retired | retired | package markers, deleted with the tree (two were git-renamed onto new empty `__init__.py`s) |
| `develop_hypothesis/test_develop_hypothesis.py` | retired | retired | pins a deleted hydra analysis (symbolic distinguishability engine); no protocol expression in v1 — flagged at the gate |
| `develop_hypothesis/expected_distinguishability.json` | retired | retired | value pins of the retired analysis above |
| `exploration/test_exploration_cli.py` | retired | retired | pure-logic helpers of the deleted exploration analysis |
| `exploration/test_exploration_smoke.py` | retired | retired | per-mode smoke of the deleted exploration analysis |
| `manifold_bundle_ingest/test_manifold_bundle_ingest.py` | retired | retired | the bundle→characterize producer deleted with the characterize chain |
| `path_patching/test_receiver.py` | retired | retired | analysis-layer receiver-config→`ReceiverSpec` glue; receivers are explicit document reads now (corpus 03) |
| `subspace/test_fixed_orientation.py` | retired | retired | orientation resolution of the deleted loader; the protocol loads rotations via `ArtifactIdentity`, shape-checked on load (`test_run_corpus` roundtrip) |
| `subspace/test_subspace.py` | retired | retired | PCA/DAS analysis wrappers deleted; the DAS fit mechanics live in `neural/pytorch_hooks/test_train.py` |
| `subspace/test_token_position_resolution.py` | retired | retired | hydra-default `token_positions` regression; documents name positions explicitly and unknown names are load errors (`protocol/test_validation_rules.py`) |
| `test_ablation_dispatch.py` | retired | retired | component-string→grid-builder dispatch of the deleted analysis; sites are explicit document entries |
| `test_activation_manifold_subdir.py` | retired | retired | output-path routing of the deleted manifold chain |
| `test_baseline_confusion.py` | retired | retired | confusion-contamination detector of the deleted `baseline` analysis; no protocol expression in v1 |
| `test_baseline_top_logits.py` | retired | retired | the deleted `baseline` record contract; the top-k values themselves are pinned in `neural/pytorch_hooks/test_metrics.py::test_top_k_orders_by_probability` |
| `test_causal_sufficiency_dispatch.py` | retired | retired | dispatch glue of the deleted causal_sufficiency analysis |
| `test_output_manifold_cache.py` | retired | retired | belief-distribution cache of the deleted output_manifold chain |
| `test_spec_loading.py` | retired | retired | WU5 per-cell bundle loaders deleted; loading is the featurizer `file_path` + `ArtifactIdentity` check (`test_run_corpus` roundtrip) |
| `test_subspace_modes_subdir.py` | retired | retired | per-mode output routing of the deleted analysis; sweeps write coordinate-labeled outputs (`protocol/test_sweep.py::coordinate_label`) |

## tests/runner

| old file | classification | disposition | note |
|---|---|---|---|
| `__init__.py` | retired | retired | package marker, deleted with the tree |
| `test_convention.py` | retired | retired | tests hydra discovery + composition of deleted configs |
| `test_dedup_guard.py` | retired | retired | dataset-dedup incoherence guard of the runner's data prep; datasets enter documents pre-built, content-digested |
| `test_directive_validation.py` | retired | retired | runner `_..._` directive vocabulary died; unknown document keys are strict-parse errors (`protocol/test_schema.py`) |
| `test_failed_step_cleanup.py` | retired | retired | half-written-output cleanup of the deleted step runner |
| `test_fanout.py` | retired | retired | SLURM fan-out orchestrator deleted; in-document sweep expansion replaces the manifest fan-out (`protocol/test_sweep.py`) |
| `test_prepare_datasets.py` | retired | retired | the neural-aware data-prep gate died with the runner |
| `test_run_exp_dispatch.py` | retired | retired | `run_exp.sh` SLURM dispatch deleted — this file held the 12 pre-existing baseline failures |
| `test_task_config_validation.py` | retired | retired | required-`task.*`-key validation for hydra configs; document validation is the load-error checklist (`protocol/test_validation_rules.py`) |
| `test_task_setup_figure.py` | retired | retired | the task-setup figure CLI died with the runner; the plot helper's own tests are kept (`tests/io/plots/test_task_setup.py`) |
| `test_taskless_analysis.py` | retired | retired | hydra struct-mode regression of the deleted `run_exp.main()` |

## tests/end_to_end

| old file | classification | disposition | note |
|---|---|---|---|
| `__init__.py`, `_helpers/__init__.py` (2) | retired | retired | package markers, deleted with the tree |
| `_helpers/enumeration.py` | retired | retired | hydra compose harness for the deleted config trees |
| `_helpers/golden.py` | retired | retired | golden-runner capture/replay harness of the deleted runner tier |
| `_helpers/runner_completion.py` | retired | retired | artifact-completion assertion bundle for deleted runners |
| `_helpers/test_runner_completion.py` | retired | retired | unit tests of the retired helper above |
| `configs/` (53 YAML: 3 top-level, `golden/` 21, `model/` 2, `smoke/` 27) | retired | retired | hydra runner configs; the presets moved to `causalab/configs/protocols/*.json` and the corpus documents (`tests/protocols/*.json`) |
| `goldens/` (21 JSON) | retired | retired | runner-golden pins of the deleted chains; the numeric-anchor role passes to `protocol/corpus_digests.json` + the parity goldens — the chat-coherent GPU tier itself is a gap |
| `test_compose.py` | retired | retired | tests hydra composition of deleted configs |
| `test_exploration_pca.py` | retired | retired | GPU golden pin of the deleted exploration analysis on chat-coherent |
| `test_exploration_pca_expected.json` | retired | retired | pins of the retired test above |
| `test_fixed_subspace.py` | retired | retired | fixed-subspace→manifold→path_steering chain deleted; the rotation-loading contract survives as `ArtifactIdentity` (`test_run_corpus` roundtrip) |
| `test_golden_standard.py` | retired | retired | accuracy-floor inspection over the retired golden JSONs |
| `test_goldens.py` | retired | retired | the golden tier retired with the runner; needs a protocol-native golden-tier decision — see gaps |
| `test_smoke.py` | replaced interface | `neural/pytorch_hooks/test_run_corpus.py` | the "runs + artifacts land" tier re-expressed: corpus 01/02/03/06 end to end on tiny-random through the real CLI (incl. `--set` overrides) |
| `update_goldens.py` | retired | retired | regen script for the retired pins; the corpus analog is `protocol/update_corpus_digests.py` |

## tests/io

| old file | classification | disposition | note |
|---|---|---|---|
| `__init__.py`, `plots/__init__.py` (2) | kept frozen | kept | unchanged |
| `plots/test_causal_graph.py` | kept frozen | kept | unchanged |
| `plots/test_string_heatmap.py` | retired | retired | pins the deleted SiteGrid-bound string-heatmap dispatcher (`io/plots/string_heatmap` left in the cut) |
| `plots/test_task_setup.py` | kept frozen | kept | unchanged |
| `test_artifact_viewer.py` | kept frozen | kept | unchanged |
| `test_artifacts.py` | kept frozen | kept (trimmed) | `TestSaveTrainingArtifacts` removed with the Plan-era `save_training_artifacts`; the rest passes unchanged |
| `test_centroids.py` | kept frozen | kept (trimmed) | one case importing the deleted `analyses.subspace._visualization` removed; the rest unchanged |
| `test_counterfactuals.py` | kept frozen | kept | unchanged |
| `test_figure_format.py` | kept frozen | kept (trimmed) | `TestResolveFromAnalysis` (analysis-config resolution) removed with the analyses; the rest unchanged |
| `test_receptive_field.py` | kept frozen | kept | unchanged |
| `test_viewer_spec_merge.py` | kept frozen | kept | unchanged |
| `test_visualizations.py` | kept frozen | kept (trimmed) | the SiteGrid cell helpers (`TestCellsFromSiteGrid` + grid builders) removed with the grid dispatchers; the rest unchanged |

## tests/tasks

| old file | classification | disposition | note |
|---|---|---|---|
| `__init__.py` ×10 (tasks + 9 task packages) | kept frozen | kept | unchanged |
| `*/pinned_samples.json` (5: IOI, MCQA, entity_binding, hex_color, hierarchical_equality) | kept frozen | kept | unchanged pinned-sample data |
| `IOI/conftest.py` | kept frozen | kept (rewired) | `LMPipeline` fixture → `tests/_helpers/pipeline_shim.PipelineShim` over the reference backend's loader |
| `IOI/test_IOI_numerical.py` | kept frozen | kept | unchanged |
| `IOI/test_causal_models.py` | kept frozen | kept | unchanged |
| `IOI/test_readout_token.py` | kept frozen | kept (rewired) | `methods.metric.single_token_id` → `pytorch_hooks.metrics.column_token_id` (the function's new home) |
| `MCQA/test_MCQA_numerical.py` | kept frozen | kept | unchanged |
| `MCQA/test_causal_models.py` | kept frozen | kept | unchanged |
| `MCQA/test_counterfactuals.py` | kept frozen | kept | unchanged |
| `MCQA/test_token_positions.py` | kept frozen | kept (rewired) | `LMPipeline` → `PipelineShim`; same position assertions |
| `entity_binding/conftest.py` | kept frozen | kept (rewired) | fixture → `PipelineShim` |
| `entity_binding/test_entity_binding.py` | kept frozen | kept | unchanged |
| `entity_binding/test_entity_binding_numerical.py` | kept frozen | kept | unchanged |
| `graph_walk/test_graph_walk.py` | kept frozen | kept (rewired) | inline `_tiny_pipeline` helper → `PipelineShim`; same assertions |
| `graph_walk/test_manifold_steering.py` | kept frozen | kept | symbolic graph/eval/counterfactual utilities — untouched by the manifold-chain deletion |
| `hex_color/test_causal_models.py` | kept frozen | kept | unchanged |
| `hex_color/test_hex_color_numerical.py` | kept frozen | kept | unchanged |
| `hierarchical_equality/conftest.py` | kept frozen | kept (rewired) | fixture → `PipelineShim` |
| `hierarchical_equality/test_hierarchical_equality.py` | kept frozen | kept (rewired) | `LMPipeline` now imported from `neural.token_positions` (the structural `EncodingPipeline` protocol home) |
| `hierarchical_equality/test_hierarchical_equality_numerical.py` | kept frozen | kept | unchanged |
| `identity_naming/test_causal_models.py` | kept frozen | kept | unchanged |
| `identity_naming/test_identity_naming.py` | kept frozen | kept | unchanged |
| `natural_domains_arithmetic/test_natural_domains_arithmetic.py` | kept frozen | kept (rewired) | `form_groups` import follows the function to `causal/causal_utils.py` |
| `subject_object_relations/test_causal_models.py` | kept frozen | kept | unchanged |
| `test_loader.py` | kept frozen | kept (rewired) | `resolve_task` follows the function from `runner.helpers` to `causalab/tasks/loader.py` |
| `test_preflight.py` | kept frozen | kept | unchanged (model-free tokenizer pre-flight is task logic) |
| `test_random_words.py` | kept frozen | kept | unchanged |

## tests/causal

| old file | classification | disposition | note |
|---|---|---|---|
| `__init__.py` | kept frozen | kept | unchanged |
| `test_causal_model.py` | kept frozen | kept | unchanged |
| `test_causal_utils.py` | kept frozen | kept | unchanged (now also the home tree of the moved `form_groups`) |
| `test_counterfactual_dataset.py` | kept frozen | kept | unchanged |
| `test_trace.py` | kept frozen | kept | unchanged |

## tests/_helpers

| old file | classification | disposition | note |
|---|---|---|---|
| `__init__.py` | kept frozen | kept | unchanged |
| `task_pins.py` | kept frozen | kept | unchanged |
| `tasks.py` | kept frozen | kept | unchanged |
| `test_tiny.py` | kept frozen | kept | unchanged |
| `tiny.py` | kept frozen | kept | unchanged (`pipeline_shim.py` is a NEW sibling, not a migration of an old file) |

## tests/ root (`tests/test_*.py`, `conftest.py`, `__init__.py`)

| old file | classification | disposition | note |
|---|---|---|---|
| `__init__.py` | kept frozen | kept | unchanged |
| `conftest.py` | kept frozen | kept (trimmed) | tier-marker collection hook kept; the tiny-random hydra guardrail and the `assert_runner_completed` re-export removed with their subjects |
| `test_architecture_layering.py` | kept frozen | kept | unchanged |
| `test_characterize_subspace_judge.py` | retired | retired | judge invariants of the deleted characterize_subspace analysis chain; no protocol expression in v1 — flagged at the gate |
| `test_characterize_subspace_loader.py` | retired | retired | raw-HF loader device regression of the deleted analysis |
| `test_characterize_subspace_smoke.py` | retired | retired | end-to-end smoke of the deleted analysis |
| `test_characterize_subspace_subspace_builder.py` | retired | retired | SAE-cluster subspace producer deleted with the chain; SAE checkpoint IO itself stays covered (`test_io_sae_checkpoints.py`) |
| `test_characterize_subspace_webtext.py` | retired | retired | webtext representation/figure helpers of the deleted analysis |
| `test_io_sae_checkpoints.py` | kept frozen | kept | unchanged |
| `test_llm_judge_primitives.py` | retired | retired | `methods/llm_judge` deleted with the analyses that consumed it |
| `test_logit_lens.py` | replaced interface | `neural/pytorch_hooks/test_read_oracle.py::test_lm_head_read_is_the_model_logits` | the method module died; the lens is a document spelling (corpus 06 injects a residual and reads `lm_head`) |
| `test_runner_env.py` | retired | retired | `.env` loading at runner entry died with the runner |

## Production modules retired without a test successor

The tables above ledger *tests*; these `causalab/` production modules were
deleted by the same refactor and are recorded here so nothing is dropped
silently. Rationale for the execution stack: this is now a public repository —
cluster-specific machinery does not belong in it, documents and workflows are
scheduler-agnostic by design (spec §8, "Execution scale"), and job dispatch is
site tooling that hooks in via the CLI's `--points` shard selector.

| module | role | disposition |
|---|---|---|
| `scripts/run_exp.sh` | runner-config discovery + inline/sbatch dispatch | retired; the CLI verbs replace inline runs, site tooling owns dispatch |
| `causalab/runner/slurm_args.py` | GPU/time/job-name resolution for sbatch | retired with the SLURM path |
| `causalab/runner/fanout.py` | shard fan-out (SLURM array + local CUDA_VISIBLE_DEVICES pool) and `--collect` recombination | retired; in-document sweep expansion + `--points` replace the manifest fan-out (the SLURM array backend was already broken at base — `scripts/fanout_array.sbatch` never existed); cross-point parallelism *inside* a backend is epic objective I11 |
| `causalab/runner/run_pipeline.sh` | serial analysis chain with free-GPU pick | retired with the analyses chains; workflow documents are the successor |
| `causalab/configs/cluster/*.yaml` | site directives (partition/account/qos) | retired; site configuration lives outside the repo |
| `causalab/io/pipelines.py::load_pipeline` | `device_map="auto"` + bf16 + chat-template loading (the 70B path) | replaced by `neural/pytorch_hooks/loading.py` (single device via `--device`/`--dtype`); multi-GPU sharding is epic objective I10, the chat-template path is an acknowledged fidelity gap |
| `causalab/tasks/subject_object_relations/data/curation_sweep.py` | per-relation curation gate on the chat-coherent pipeline | retired with its three deleted imports while its measured table stays load-bearing (task README, `config.py` default); re-expressing it needs the dataset-serialization seam + a first-token metric kind (epic I1) |

## Known coverage gaps

- **Parity captured-goldens re-drive** — landed on this branch:
  `neural/pytorch_hooks/test_parity_goldens.py` replays every portable pinned
  case, but 2 gqa `head_value` pins are explicitly skipped (the §2.4 vocabulary
  has no v_proj-output component) and the skip table is itself under audit.
- **Chat-coherent GPU golden tier** — discharged on this branch by
  `tests/golden/`: paper-provenance goldens (`test_paper_goldens.py`, values
  from published papers / the VeriFires task packages, never a stack run) and
  the drift tier (`tests/golden/drift/`, Qwen3-4B value pins captured by a
  reviewed GPU run — the retired tier's role). Remaining slivers: the drift
  pins await their first cuda capture (the replay skips until then), and the
  old tier's chat template + answer directive are inexpressible in v1 (no
  chat-template code path), so the drift documents use raw completions.
- **Per-head query receivers** — no spec component for the pre-attention
  q_h = W_Q^h·x receiver (`test_head_receivers_hook_oracle.py`'s query half);
  path patching onto query receivers is inexpressible in v1.
- **Cross-model / two-pipeline patching** — one model per document in v1;
  `test_cross_model_hook_oracle.py`'s source-pipeline injection contract has no
  new carrier.
- ~~**Task-driven end-to-end IIA pins**~~ — **closed** by the
  dataset-serialization seam (§2.2): a task's counterfactual dataset is
  serialized by `causalab/tasks/serialize.py` and driven as corpus document
  `10_task_table_iia_im.json`; the pin is
  `tests/neural/pytorch_hooks/test_end_to_end_iia.py`. Values are captured
  fresh, not carried over — the retired test scored generated strings through
  `task.checker`, a document scores an argmax against the answer's declared
  forms, and its coherent-model half used a chat template v1 cannot express.
  The retired pins' *shape* (flat IIA at tiny scale, guarded by a
  non-inertness assertion on the patched logits) is reproduced.
- ~~**Dynamic per-row positions**~~ — **closed**: a position may read a
  per-row `column` (§2.3), as anchor or as a `scope`/`relative_to` anchor,
  resolved per row like a prompt variable. Values a task computes per row
  (MCQA's correct answer symbol) are serialized columns rather than a
  computed-indexer vocabulary. Integer token indices remain out of v1
  deliberately: they would bind a table to one tokenizer.
- **Per-position metric grids** — `{"all": true}` makes an every-token *read*
  expressible, but metric lowering still reduces exactly one position per
  example, so a logit-lens grid comes out as a saved tensor, not a metric
  table. A per-position metric needs a `position` column in the metric table.
