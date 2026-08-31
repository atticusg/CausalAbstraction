# Architecture

## 1. Package structure

| Module | Named for |
|---|---|
| `causal/` | causal model primitives |
| `tasks/` | task definitions (causal models + counterfactual generators) + `serialize.py`, which writes them out as dataset tables |
| `protocol/` | the engine-free document layer |
| `neural/engines/pytorch_hooks/` | the reference execution engine |
| `analysis/` | numerical analysis a workflow `script` step runs: fits, statistics, intervention operands |
| `workflow/` | the workflow document model, runner, and CLI verbs |
| `io/` | disk I/O + shared plotting primitives |
| `configs/` | shipped documents: `protocols/` (flat), `runs/` (split into `application` + `method`, §1.1), `methods/` (reusable halves), `workflows/` — JSON, not code |

**Dependency flow:** `tasks/` and `causal/` are independent. `protocol/` is torch-free and links against no execution engine — the CLI imports the reference engine lazily. `neural/engines/pytorch_hooks/` implements `protocol.engine.Engine`. `steps/` depends only on `protocol/` and is torch-free at module level: a step script's numerics are imported inside its `main`, so listing the shipped scripts and hashing one cost nothing but stdlib. `workflow/` depends on `protocol/`, drives whichever engines it is handed, and reaches `steps/` lazily when it runs a script step; the workflow loader likewise reaches the shipped-script directory through a function-local import, so `protocol/` keeps no module-level edge to anything that executes. `io/` depends only on `neural/`, `tasks/`, `causal/`. `tests/test_architecture_layering.py` enforces the static half of all this; `tests/protocol/test_load_is_torch_free.py` the behavioural half.

## 2. The protocol layer (`causalab/protocol/`)

The normative spec is [`docs/intervention_protocol.md`](intervention_protocol.md); this is the module map.

| module | owns |
|---|---|
| `schema.py` | typed document model; closed vocabularies (components, `do` mechanisms, metric kinds) |
| `loader.py` | strict load, `--set` overrides, the §5 validation checklist, column checks |
| `canonical.py` | canonical form + digests (§7) |
| `sweep.py` | axis expansion, point cap, coordinate labels (§3) |
| `bundles.py` | addressing one entry inside a saved `.safetensors` bundle: key grammar, coordinate selection (§2.5, §2.6) |
| `tables.py` | metric tables on disk — native JSON, an array of row objects. Torch-free and pandas-free, so the engine writes through it and step scripts read through it |
| `plan.py` | model graph → forward groups, content dedup (§4) |
| `engine.py` | `Engine` ABC, `ExecutionRequest`/`RunResult`, capability routing (§8) |
| `resolve.py` | `ResolutionEnv`: the `DatasetResolver` contract (digest / columns / rows) with `FileDatasets` (JSON tables), `FileArtifacts`, `ArtifactIdentity` build/check |
| `registry.py` | static model metadata (widths per component); built-in entries for the models the corpus and goldens name |
| `workflow.py` | the workflow *document* model: parse, the 11-rule checklist, the locator+selector reference grammar, derived schedule, script hashing, digests |
| `cli.py` | `causalab run/validate/explain/digest`, dispatching on document type; `--device/--dtype/--points/--resume` on `run` |

Documents are pure data. Sweeps expand at load into point protocols; the campaign digest names the document, each point's digest is the provenance unit.

## 3. The reference engine (`causalab/neural/engines/pytorch_hooks/`)

Implements the §8 services with raw pytorch hooks, CPU or a single accelerator (`device`/`dtype` constructor args — `cuda`, `cuda:1`, `mps`):

| module | service |
|---|---|
| `loading.py` | model+tokenizer bundles (left padding, eager attention, frozen weights) |
| `sites.py` | component vocabulary → module taps; Llama-tree (Llama/Qwen/Mistral/Gemma) and GPT-2-tree families |
| `encoding.py` | tokenization, char→token spans, `PositionFrame`, position specs → indices (chat-prefix lengths are a field, not a code path yet) |
| `mechanisms.py` | the closed `do` set; absolute-then-additive order per address |
| `featurizers.py` | featurizer kinds + error-term contract |
| `metrics.py` | metric lowering over one lm_head read; single-token column resolution |
| `executor.py` | one forward group per (model, input), whole batch at once; edit/read hook wiring |
| `train.py` | the `train` loop for trainable featurizers |
| `outputs.py` | JSON metric tables and safetensors tensor files, coordinate-keyed, identity-stamped |
| `engine.py` | `PytorchHooksEngine`: capabilities `{grad, paired_forward, full_logits, pytorch_fn_local}` |

Known limits (tracked in the intervention-protocol epic): one device per run (no `device_map` sharding), one batch per forward group (no microbatching), no `attention_probs`, no chat-template path.

## 4. The workflow runner (`causalab/workflow/`)

Executes workflow documents: topological step order from derived references, per-step output dirs under `<out-root>/<output_dir>/`, an artifact overlay so later steps resolve earlier steps' products, protocol and script steps, a `_step.json` record per step and a `workflow.json` run manifest. There is no publication step — the run tree *is* the publication (spec §0). The runner knows only the step graph: device/dtype live in the engines it is handed, and job dispatch is site tooling outside the repo (spec §8, "Execution scale").

**Step scripts.** A `script` step names its code with a locator —
`{"module": "causalab.analysis.fit_pca"}` or `{"path": "scripts/probe.py"}` — so
the shipped ones are filed **by subject** rather than in one namespace:

| module | what it holds |
|---|---|
| `causalab/analysis/` | numerical analysis: `fit_pca`, `harvest_difference`, `head_stats`, `paired_ttest`. Fits, statistics, and the operands an intervention consumes. Importable and testable without the workflow layer |
| `causalab/io/plots/workflow_figures.py` | the heatmap/lines renderer, beside the other 17 plot modules and reusing `figure_format` for the png-over-pdf default |
| `causalab/workflow/scripts/select.py` | the one script whose purpose *is* wiring: reduce a table to the values a later document's `set` reads |
| `causalab/io/step_io.py` | what a script uses for IO: JSON tables and values objects, safetensors with `slot`/`entry` addressing, the identity a tensor output inherits |
| `causalab/io/step_record.py` | the `_step.json` format — its writer (used by the runner), its reader, and the shared aggregation rule `select` and `workflow_figures` both call, so a figure and a chosen value never disagree about what a row is |
| `causalab/workflow/isolate.py` | the entry point for an isolated (subprocess) step |

A script is one function, `main(inputs, outputs) -> None`, that creates every
output it declares. The runner verifies they arrived, checks a declared table's
columns, and stamps ArtifactIdentity on safetensors outputs — so a script cannot
forget provenance, which is what a later protocol step's identity check depends
on. A script is **found and hashed, never imported** at load
(`importlib.util.find_spec` resolves a module to a file without executing it):
`validate`/`digest` stay torch-free, and the hash in the digest is what makes
`--resume` correct.

**Why two packages.** `protocol/` is the intervention protocol alone and must
not import the workflow layer — that is what lets someone use it on its own, and
`tests/test_architecture_layering.py` enforces it. The dependency runs
`workflow/` → `io/` → `protocol/`, one way, and `causalab/cli.py` sits above
both, dispatching on the document's `steps` section.

## 5. Datasets are build products

A document names a dataset ref; a resolver reads bytes (`protocol/resolve.py`). Nothing generates a table during a load, so `validate` / `explain` / `digest` need no task code, no tokenizer and no network, and a document's digest is a function of committed bytes. `causalab/tasks/serialize.py` + `scripts/build_task_dataset.py` are the other side: task package → deterministic table + a `<ref>.manifest.json` provenance sidecar. Everything per-row or task-semantic (answer forms, values that place a position per row) is a column written there, never a document-side computation (spec §2.2).

## 6. Configs are documents

`causalab/configs/protocols/*.json` are the shipped protocol documents (most are byte-identical to a corpus document under `tests/protocols/`; the exceptions are the presets with no corpus twin — `mean_harvest`, `mean_ablation`, `dbm_apply`); `causalab/configs/runs/weekdays_8b_interchange.json` carries the same experiment as one document split into its transferable and its input-bound halves, and `causalab/configs/methods/interchange.json` is that method on its own (spec §1.1, implemented in `protocol/method.py`); `causalab/configs/workflows/weekdays_8b.json` is the worked workflow. There is no Python config system: a "config" is a protocol or workflow document, overridden ad hoc with `--set` and promoted into a file when it matters.

## 7. Tests

See [`docs/TESTS.md`](TESTS.md) for the tier taxonomy and pinned-artifact discipline, and [`docs/test_migration.md`](test_migration.md) for the old-suite → new-suite ledger.
