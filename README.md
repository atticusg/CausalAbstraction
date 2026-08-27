# Causal Abstraction for Mechanistic Interpretability

![Tests](https://github.com/goodfire-ai/causalab/workflows/Tests/badge.svg)

A framework for **mechanistic interpretability** — reverse-engineering the algorithms language models use internally using **causal abstraction**.

You write a high-level causal model describing *how you think* an LM solves a task, then run experiments to test whether the LM's internal components actually implement that algorithm. Every experiment is a serializable **intervention protocol** — a JSON document naming sites, reads, edits, intervened models, and metrics — validated, digested, and executed by a backend. The document is the seam: backends (pytorch hooks today; tensor-parallel engines tomorrow) implement against the same format.

## Quick Start

1. **Clone and install:**
   ```bash
   git clone https://github.com/goodfire-ai/causalab.git
   cd causalab
   uv sync
   ```
2. **Read the two specs.** [`docs/intervention_protocol.md`](docs/intervention_protocol.md) — the document format (sections, the `do` algebra, sweeps, validation, digests, the backend contract). [`docs/workflow_protocol.md`](docs/workflow_protocol.md) — chaining protocol runs with script steps: inputs, one Python script, declared outputs.
3. **Run a method preset:**
   ```bash
   uv run causalab explain  causalab/configs/methods/interchange.json --data-root <data>
   uv run causalab run      causalab/configs/methods/interchange.json \
       --data-root <data> --out runs/interchange --device cuda --dtype bf16
   ```

## The CLI

| verb | effect |
|---|---|
| `run <doc>` | validate, expand, plan, execute, stamp |
| `validate <doc> [--data]` | the spec §5 load-error checklist; `--data` also checks column references |
| `explain <doc>` | models, forward plan, point count, derived `requires`, digest, save products |
| `digest <doc>` | the campaign digest |

Common flags: `--set path=value` (ad-hoc override — exploration only), `--data-root` / `--artifacts-root` (resolution roots), `--device` / `--dtype` (reference-backend placement, `run` only), `--points START:STOP` (execute one shard of a swept campaign — the seam external schedulers dispatch on; digests are unaffected). The same verbs dispatch on workflow documents (they carry a `steps` section).

**Execution scale is not document vocabulary.** Documents and workflows never name devices, hosts, or job systems: backends own intra-run execution, and job dispatch is site tooling outside this repository (spec §8, "Execution scale").

## Method presets

The golden-corpus documents ship as user-facing presets in [`causalab/configs/methods/`](causalab/configs/methods/):

| preset | method |
|---|---|
| `harvest` | activation harvesting at named sites/positions |
| `interchange` | interchange intervention + IIA scoring |
| `path_patching` | sender→receiver path patching with off-path freezing |
| `das` | trained orthogonal-subspace interchange (DAS) |
| `dbm` | differential binary masking through a trained gate |
| `hydra_effect` | resample-ablation + downstream direct-effect probes |
| `weekdays_locate_scan` | layer × position interchange scan (one shared harvest) |
| `weekdays_das_sweep` | k × seed DAS fits at a located cell |
| `weekdays_das_apply` | apply a fitted rotation (ArtifactIdentity-checked) |

[`causalab/configs/workflows/weekdays_8b.json`](causalab/configs/workflows/weekdays_8b.json) chains locate → select → fit → apply → plots as one workflow document (two step types: `protocol` and `script`).

## Repository layout

```
causalab/
├── protocol/        # backend-free document layer: load, validate, canonicalize,
│                    #   digest, sweep expansion, backend routing, workflow model, CLI
├── neural/
│   ├── pytorch_hooks/  # the reference backend: sites, positions, mechanisms,
│   │                   #   featurizers, metrics, train loop, stamping
│   └── token_positions.py
├── steps/           # the Python a script step runs: IO helpers + the shipped `causalab:*` scripts
├── workflow/        # the workflow runner: run-tree overlay, script invocation, manifest
├── causal/          # causal model primitives
├── tasks/           # task definitions (causal models + counterfactual generators)
├── io/              # disk I/O + plotting primitives
└── configs/         # method presets (JSON documents) + workflow documents
docs/                # the two specs, CODEBASE.md, TESTS.md, test_migration.md
tests/               # tiered suite — see docs/TESTS.md
```

## Core concepts

- **Causal model**: your hypothesis about how the LM solves a task — variables, values, parent–child dependencies, mechanisms (`causalab/causal/`).
- **Task**: a prompt distribution plus a causal model and counterfactual generators (`causalab/tasks/`).
- **Intervention protocol**: one experiment as data — which activations are read, which are edited (`swap`, `add_scaled`, `gaussian`, …), in which intervened models, scored by which metrics. Sweeps expand a document into a campaign of points with content-deduped shared work.
- **Workflow**: a chain of protocol executions plus script steps, with dependencies derived from references — never authored ordering. Everything a step declares is published where it lands; there is no save manifest.

## Tests

See [`docs/TESTS.md`](docs/TESTS.md). CPU tiers run with `uv run pytest -m "not golden"` (what CI runs). The `golden` tier runs real models on an accelerator: paper-provenance goldens (`tests/golden/`) and the chat-coherent drift pins (`tests/golden/drift/`).

## History

The Hydra runner, `analyses/` chains, `methods/` as Python, SLURM dispatch, and the notebook demos were retired in the protocol refactor (PR #20); [`docs/test_migration.md`](docs/test_migration.md) is the ledger. Their intervention cores return as method presets and workflow documents.
