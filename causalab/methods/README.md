# Methods

The interpretability toolbox: a library of interchangeable primitives — subspace
featurizers (DAS, PCA, SAE, …), interventions (interchange, ablation, path
patching, causal tracing, steering), manifold/geometry builders, and scoring
metrics — that analyses select and compose to answer research questions.

**Library contract.** Everything here is *library code*: pure in-memory
functions and classes that take tensors, pipelines, and datasets and **return**
results (dicts, tensors, featurizers). Methods do **no research orchestration**
— no dataset loading from paths, no artifact-directory layout, no Hydra reads,
no embedded hyperparameter defaults. They receive configuration as explicit
keyword arguments (or a pre-resolved config object) from their caller; the
analysis layer owns disk layout, Hydra, and research intent. A few modules ship
a thin composite `save_*` helper beside the method (e.g. `save_pca_results`,
`save_logit_lens_results`) — these bundle disk layout + plot for one method but
consume only `causalab/io/` primitives, the sole documented exception.

Depends on `neural/`, `causal/`, and `io/`; consumed by `analyses/`. Must not
import from `runner/` or `analyses/`. See `docs/CODEBASE.md` §1 (dependency
flow) and §3 (layering invariants) for the full rules.

## Catalog

### Subspace / feature-space construction

| Component | What it provides |
|---|---|
| `pca.py` | `collect_and_compute_PCA` — single-pass feature collection over units + SVD/PCA, returning per-unit rotations that seed a subspace featurizer. |
| `sae.py` | `SAEFeaturizer` — a `Featurizer` wrapping a pretrained SAE's encode/decode pair; `decoder_subspace` builds a `(d_model, k)` basis from selected decoder directions. |
| `trained_subspace/` | `SubspaceFeaturizer` — an orthogonal rotation featurizer (the DAS / learned-subspace primitive) — plus `train_interventions`, the generic training loop shared by DAS (interchange) and mask-based (DBM / Boundless) fitting. |
| `interchange/` | Run interchange interventions (base↔counterfactual activation swaps): `run_layer_scan`, `run_centroid_layer_scan`, `run_single_pair_trace`, cached feature collection — the core localization primitive. |

### Interventions

| Component | What it provides |
|---|---|
| `ablation/` | Zero / mean reference vectors (`make_zero_vectors`, `make_mean_vectors`) and ablation runners/scans (`run_ablation*`) scoring behavioral drop. |
| `steer/` | Additive / replacement steering in feature space (`steer.py`) plus manifold distribution collection and grid generation (`collect.py`). |
| `path_patching/` | Isolate the *direct* sender→receiver edge effect — build receiver/restorer targets and run one-pass (logits) or two-pass (internal receivers) path patching. |
| `causal_tracing/` | ROME-style causal tracing: corrupt the entry, restore one site, score recovery (`run_causal_trace`, `run_causal_trace_scan`, `run_corrupted_floor`). |

### Geometry

| Component | What it provides |
|---|---|
| `spline/` | `SplineManifold` (thin-plate-spline and cubic-spline backends) and `ManifoldFeaturizer` — fit a low-dimensional manifold and use it as a feature space. |
| `pullback/` | Belief-manifold trace-path construction and activation-space trajectory optimization (geodesic, L-BFGS/TPS-spline path fitting). |
| `distances.py` | Distances between output belief distributions (Fisher-Rao, Hellinger, cyclic/linear Wasserstein, log-prob) plus geodesic and conformal path builders. |
| `umap.py` | `UMAPFeaturizer` — a parametric UMAP featurizer with MLP encoder/decoder approximating a UMAP embedding for differentiable inference. |

### Scoring / metrics

| Component | What it provides |
|---|---|
| `scores/` | Path/manifold geometry scores: `coherence` (on-target probability along a path), `distance_from_behavior_manifold`, `isometry` (activation-vs-belief manifold correlation). |
| `metric.py` | Intervention scoring: `class_probabilities`, `kl_divergence`/`hellinger_distance`, the `InterchangeMetric` protocol, logit / prob / logit-diff / distribution-shift metric builders, and base-accuracy computation. |
| `output_tokens.py` | Resolve a causal model's per-variable `output_tokens` declaration into tokenizer-aware score-token id groups (and their labels/values). |
| `filter.py` | `filter_dataset` — drop counterfactual examples where the neural pipeline and the causal model disagree. |

### Analysis helpers

| Component | What it provides |
|---|---|
| `logit_lens.py` | Project intermediate residual-stream activations through the final layer norm + unembedding to vocabulary logits (`compute_logit_lens`). |
| `attention_pattern_analysis.py` | Extract per-head attention matrices across prompts and reduce them (per-token-type averages, entropy/self/previous-token statistics). |
| `generation.py` | `greedy_output` — one prompt → the model's stripped greedy continuation (normalizes `pipeline.generate`'s str-or-list shape). |
| `standardize.py` | `StandardizeFeaturizer` — a bijective affine `(x − mean)/std` featurizer, useful for composing before another featurizer. |
| `llm_judge/` | Domain-neutral LLM-judge primitives: `call_llm` (OpenRouter/OpenAI with retries), `resolve_credentials`, `extract_json_response`, forbidden-substring guards. |

`__init__.py` re-exports only the concrete `Featurizer` subclasses
(`SubspaceFeaturizer`, `StandardizeFeaturizer`, `ManifoldFeaturizer`,
`ManifoldProjectFeaturizer`, `SAEFeaturizer`, `UMAPFeaturizer`) so importing the
package registers them for `Featurizer.from_dict` dispatch on load.

## How methods are used

Analyses import methods and select which one to use by config. The technique for
*constructing* an intervened feature space — PCA, DAS, DBM, Boundless DAS,
fixed rotation — is chosen with `subspace.method: das` (see
`causalab/configs/analysis/subspace.yaml`); the localization primitive with
`locate.method: interchange`. Method-specific hyperparameters (epochs, learning
rate, regularization) live under the method's config block, never in the method
code.

The research-question wrappers that load data, lay out artifacts, and drive
these methods end to end live in `causalab/analyses/` (e.g. `subspace/`,
`locate/`, `path_steering/`); for how to run them, see the runner's README.
