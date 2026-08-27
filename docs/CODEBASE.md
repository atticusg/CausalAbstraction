# Architecture

## 1. Package structure

| Module | Named for |
|---|---|
| `causal/` | causal model primitives |
| `tasks/` | task definitions (causal models + counterfactual generators) + `serialize.py`, which writes them out as dataset tables |
| `protocol/` | the backend-free document layer |
| `neural/pytorch_hooks/` | the reference execution backend |
| `transform/` | the registry of deterministic, versioned ops a workflow `transform` step runs |
| `workflow/` | the workflow runner |
| `io/` | disk I/O + shared plotting primitives |
| `configs/` | method presets and workflow documents (JSON, not code) |

**Dependency flow:** `tasks/` and `causal/` are independent. `protocol/` is torch-free and links against no execution engine — the CLI imports the reference backend lazily. `neural/pytorch_hooks/` implements `protocol.backend.Backend`. `transform/` depends only on `protocol/` and is torch-free at module level: its op *records* are what load-time validation reads, so an op's numerics are imported inside its function body. `workflow/` depends on `protocol/`, drives whichever backends it is handed, and reaches `transform/` lazily when it runs a transform step; the workflow loader likewise reaches the op registry through a function-local import, so `protocol/` keeps no module-level edge to anything that executes. `io/` depends only on `neural/`, `tasks/`, `causal/`. `tests/test_architecture_layering.py` enforces the static half of all this; `tests/transform/test_load_is_torch_free.py` the behavioural half.

## 2. The protocol layer (`causalab/protocol/`)

The normative spec is [`docs/intervention_protocol.md`](intervention_protocol.md); this is the module map.

| module | owns |
|---|---|
| `schema.py` | typed document model; closed vocabularies (components, `do` mechanisms, metric kinds) |
| `loader.py` | strict load, `--set` overrides, the §5 validation checklist, column checks |
| `canonical.py` | canonical form + digests (§7) |
| `sweep.py` | axis expansion, point cap, coordinate labels (§3) |
| `bundles.py` | addressing one entry inside a saved `.safetensors` bundle: key grammar, coordinate selection (§2.5, §2.6) |
| `tables.py` | metric tables on disk — native JSON, an array of row objects. Torch-free and pandas-free, so the backend writes through it and step scripts read through it |
| `plan.py` | model graph → forward groups, content dedup (§4) |
| `backend.py` | `Backend` ABC, `ExecutionRequest`/`RunResult`, capability routing (§8) |
| `resolve.py` | `ResolutionEnv`: the `DatasetResolver` contract (digest / columns / rows) with `FileDatasets` (JSON tables), `FileArtifacts`, `ArtifactIdentity` build/check |
| `registry.py` | static model metadata (widths per component); built-in entries for the models the corpus and goldens name |
| `workflow.py` | the workflow *document* model: parse, checklist, derived schedule, digests |
| `cli.py` | `causalab run/validate/explain/digest`, dispatching on document type; `--device/--dtype/--points` on `run` |

Documents are pure data. Sweeps expand at load into point protocols; the campaign digest names the document, each point's digest is the provenance unit.

## 3. The reference backend (`causalab/neural/pytorch_hooks/`)

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
| `backend.py` | `PytorchHooksBackend`: capabilities `{grad, paired_forward, full_logits, pytorch_fn_local}` |

Known limits (tracked in the intervention-protocol epic): one device per run (no `device_map` sharding), one batch per forward group (no microbatching), no `attention_probs`, no chat-template path.

## 4. The workflow runner (`causalab/workflow/`)

Executes workflow documents: topological step order from derived references, per-step output dirs under the run tree, an artifact overlay so later steps resolve earlier steps' products, transform/select/plot steps, save publication, and a `workflow.json` manifest. The runner knows only the step graph — device/dtype live in the backends it is handed, and job dispatch is site tooling outside the repo (spec §8, "Execution scale").

**Transform ops (`causalab/transform/`)** are what a `transform` step runs (workflow spec §2.4):

| module | owns |
|---|---|
| `schema.py` | the slot kinds (`Table` with declared columns, `Tensor`) and the parameter primitives, plus `TransformError` |
| `registry.py` | the `TransformOp` record, the `@register` decorator, and `lookup("name@version")` with suggestions |
| `io.py` | reading inputs and writing outputs — JSON tables, `.safetensors` bundles, and the identity a tensor output is stamped with |
| `ops/` | the registered ops, one module each, numerics imported inside the function body |

An op is a pure `(inputs, params) -> {slot: value}` function; the runner owns paths, formats and provenance so an op's unit test needs no filesystem. Adding one means adding a record, a body and an oracle test — a document can never introduce an op, which is what keeps a workflow run a pure function of the document.

## 5. Datasets are build products

A document names a dataset ref; a resolver reads bytes (`protocol/resolve.py`). Nothing generates a table during a load, so `validate` / `explain` / `digest` need no task code, no tokenizer and no network, and a document's digest is a function of committed bytes. `causalab/tasks/serialize.py` + `scripts/build_task_dataset.py` are the other side: task package → deterministic table + a `<ref>.manifest.json` provenance sidecar. Everything per-row or task-semantic (answer forms, values that place a position per row) is a column written there, never a document-side computation (spec §2.2).

## 6. Configs are documents

`causalab/configs/methods/*.json` are the nine method presets (byte-comparable to the corpus documents under `tests/protocols/`); `causalab/configs/workflows/weekdays_8b.json` is the worked workflow. There is no Python config system: a "config" is a protocol or workflow document, overridden ad hoc with `--set` and promoted into a file when it matters.

## 7. Tests

See [`docs/TESTS.md`](TESTS.md) for the tier taxonomy and pinned-artifact discipline, and [`docs/test_migration.md`](test_migration.md) for the old-suite → new-suite ledger.
