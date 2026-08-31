# CausaLab codebase overview

CausaLab tests causal hypotheses about language-model internals. Researchers write
JSON documents that describe an experiment. Engines execute those documents, and
the resulting artifacts carry the document and point identities needed to verify
what produced them.

## Package map

| Directory | Responsibility |
|---|---|
| `causalab/causal/` | High-level causal models and counterfactual reasoning |
| `causalab/tasks/` | Task packages and conversion into serialized dataset tables |
| `causalab/protocol/` | Parsing, validation, canonicalization, sweeps, planning, and engine contracts |
| `causalab/neural/engines/` | Execution backends; `pytorch_hooks` is the reference engine |
| `causalab/analysis/` | Deterministic numerical functions used by workflow script steps |
| `causalab/workflow/` | Workflow loading, dependency derivation, execution, and manifests |
| `causalab/io/` | JSON, safetensors, step records, and plotting support |
| `causalab/configs/` | Shipped protocols, reusable methods, complete runs, and workflows |

See [`../../docs/CODEBASE.md`](../../docs/CODEBASE.md) for the authoritative
module-level map.

## Documents

An intervention protocol contains, in order:

- a model and serialized input tables;
- named positions and activation sites;
- optional featurizers and free parameters;
- activation reads and interventions;
- intervened model variants and metrics;
- optional training for trainable featurizers;
- a complete output manifest.

A method file is the reusable half of a protocol. It fixes the experimental logic
while leaving the model, data, and open site addresses to an application. A
complete run document may contain both halves. The application may fill open
fields, but it may not override a value fixed by the method.

A workflow contains protocol steps and deterministic Python script steps. File
references create dependencies automatically. The runner writes one directory per
step, `_step.json` records, and a top-level `workflow.json` manifest.

## Data and artifacts

Protocol loading never imports a task or generates examples. Dataset references
resolve to JSON bytes beneath `--data-root`. Build those bytes beforehand with
`scripts/build_task_dataset.py`. Values that vary by row, including answer forms
and token anchors, must be columns in that table.

Structured outputs use JSON. Dense tensors use safetensors. Tensor artifacts carry
identity metadata such as the model, site, dtype, producing engine, and producer
digest. A later protocol refuses a fitted artifact whose identity conflicts with
the consuming declaration.

## Sweeps and execution

A sweep is an explicit `{"sweep": ...}` wrapper on a named field. Several axes
form a cross product. Expansion is deterministic: the whole document has a
campaign digest and every expanded point has its own digest. `--points START:STOP`
runs a point range without changing either identity.

The reference engine uses one device per run. Execution scale belongs outside the
document. An external system may distribute independent point ranges, but the
document does not name a scheduler or machine.

## Model support

The reference engine supports the model families and activation components in its
registry and site implementation. Run `causalab explain` before loading weights;
it reports required capabilities and the forward plan. An unregistered Hugging
Face model can be registered during `run` when its configuration maps cleanly to a
supported family. Unsupported architectures require engine work, not a document
workaround.

## Tests

The normal CPU suite is:

```bash
uv run pytest -m "not golden"
```

Golden tests use real checkpoints and accelerators. See
[`../../docs/TESTS.md`](../../docs/TESTS.md) for the complete tier definitions.
