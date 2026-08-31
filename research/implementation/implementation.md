# CausaLab implementation

Use this directory when the research protocol requires a change to CausaLab or a
new executable document. CausaLab represents experiments as intervention protocol
documents and workflows, not as Hydra analyses.

## Choose the relevant guide

| Goal | Guide |
|---|---|
| Understand the package and execution model | [`causalab-overview.md`](causalab-overview.md) |
| Build and run a protocol or workflow | [`running-experiments.md`](running-experiments.md) |
| Choose an existing protocol or script | [`ANALYSIS_GUIDE.md`](ANALYSIS_GUIDE.md) |
| Diagnose a failure | [`COMMON_PROBLEMS.md`](COMMON_PROBLEMS.md) |
| Add or update a task package | [`setup-task/setup-task.md`](setup-task/setup-task.md) |
| Define a reusable intervention method | [`setup-methods/setup-methods.md`](setup-methods/setup-methods.md) |
| Add deterministic processing to a workflow | [`setup-analyses/setup-analyses.md`](setup-analyses/setup-analyses.md) |

The normative interfaces are
[`../../docs/intervention_protocol.md`](../../docs/intervention_protocol.md) and
[`../../docs/workflow_protocol.md`](../../docs/workflow_protocol.md). The code is
authoritative when a research guide and an interface disagree.

## Current execution model

1. A task package generates a deterministic JSON dataset table.
2. An intervention protocol names the model, data, activation sites, reads,
   interventions, metrics, and outputs.
3. Explicit `sweep` wrappers expand one document into reproducible points.
4. A workflow connects protocol runs with deterministic Python script steps.
5. `causalab validate`, `explain`, `digest`, and `run` operate on either document
   type.

The reference engine runs on CPU, CUDA, or MPS. One protocol run uses one device.
External infrastructure may distribute point ranges, but hosts, schedulers, and
queues never appear in a protocol or workflow document.

## Before changing code

- Read [`../../docs/CODEBASE.md`](../../docs/CODEBASE.md).
- Find the closest shipped document under `causalab/configs/`.
- Confirm the desired component, intervention, metric, featurizer, and engine
  capability already exist in the closed protocol vocabulary.
- Prefer a new document or workflow over new Python.
- Add Python only for a genuinely new engine primitive, task, or deterministic
  workflow calculation.

Run the smallest relevant unit tests first, followed by the CPU suite described in
[`../../docs/TESTS.md`](../../docs/TESTS.md).
