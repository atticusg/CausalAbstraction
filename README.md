# Causal Abstraction for Mechanistic Interpretability

![Tests](https://github.com/goodfire-ai/causalab/workflows/Tests/badge.svg)

A framework for **mechanistic interpretability** — reverse-engineering the algorithms language models use internally using **causal abstraction**.

You write a high-level causal model describing *how you think* an LM solves a task, then run experiments to test whether the LM's internal components actually implement that algorithm. Every experiment is a serializable **intervention protocol** — a JSON document naming sites, reads, edits, intervened models, and metrics — validated, digested, and executed by an engine. The document is the seam: engines (pytorch hooks today; tensor-parallel engines tomorrow) implement against the same format.

## Quick Start

1. **Clone and install:**
   ```bash
   git clone https://github.com/goodfire-ai/causalab.git
   cd causalab
   uv sync
   ```
2. **Run a demo.** [`demos/`](demos/) is one markdown file per research question, with the documents that answer it. Two need no GPU: [causal_model](demos/causal_model/causal_model.md), which has no network in it at all, and [01_define](demos/onboarding_tutorial/01_define.md). The other eight run in under two minutes on one H100 between them. The format is [`docs/demos.md`](docs/demos.md).
3. **Read the two specs.** [`docs/intervention_protocol.md`](docs/intervention_protocol.md) — the document format (sections, the `do` algebra, sweeps, validation, digests, the engine contract). [`docs/workflow_protocol.md`](docs/workflow_protocol.md) — chaining protocol runs with script steps: inputs, one Python script, declared outputs.
4. **Run a shipped protocol:**
   ```bash
   uv run causalab explain  causalab/configs/protocols/interchange.json --data-root <data>
   uv run causalab run      causalab/configs/protocols/interchange.json \
       --data-root <data> --out runs/interchange --device cuda --dtype bf16
   ```
   Or run the same experiment as one document split into its two halves — the
   **method** is the transferable half, the **application** names the network,
   the data, the addresses and the precision (spec §1.1):
   ```bash
   uv run causalab explain causalab/configs/methods/interchange.json          # what must be bound
   uv run causalab run     causalab/configs/runs/weekdays_8b_interchange.json \
       --data-root <data> --out runs/interchange --device cuda
   ```

## The CLI

| verb | effect |
|---|---|
| `run <doc>` | validate, expand, plan, execute, stamp |
| `validate <doc> [--data]` | the spec §5 load-error checklist; `--data` also checks column references |
| `explain <doc>` | models, forward plan, point count, derived `requires`, digest, save products |
| `digest <doc>` | the campaign digest |

Common flags: `--set path=value` (ad-hoc override — exploration only), `--data-root` / `--artifacts-root` (resolution roots), `--device` (reference-engine placement, `run` only), `--dtype` (shorthand for `--set model.dtype=…`: precision is a document fact, so it enters the digest), `--points START:STOP` (execute one shard of a swept campaign — the seam external schedulers dispatch on; digests are unaffected). The same verbs dispatch on workflow documents (they carry a `steps` section).

`run` also writes `<out>/protocol.json`: the canonical document (every default materialized — dtype and quantization included), its digest, the per-point provenance digests, and the method it was composed from. That file is what someone reproducing the run reads first.

**Execution scale is not document vocabulary.** Documents and workflows never name devices, hosts, or job systems: engines own intra-run execution, and job dispatch is site tooling outside this repository (spec §8, "Execution scale").

## Shipped documents

The golden-corpus documents ship as user-facing presets in [`causalab/configs/protocols/`](causalab/configs/protocols/) — complete protocol documents, network and all:

| preset | experiment |
|---|---|
| `harvest` | activation harvesting at named sites/positions |
| `interchange` | interchange intervention + IIA scoring |
| `path_patching` | sender→receiver path patching with off-path freezing |
| `attention_band_patch` | contiguous layer bands in one forward, several bands per document |
| `multi_position_patch` | several writes on one site at disjoint positions |
| `mean_harvest` / `mean_ablation` | harvest a corpus mean at save time, then swap it in |
| `das` | trained orthogonal-subspace interchange (DAS) |
| `dbm` | differential binary masking through a trained gate |
| `random_subspace_control` | the matched-k random subspace every DAS cell is read against |
| `hydra_effect` | resample-ablation + downstream direct-effect probes |
| `probe_generate` / `probe_variable` | greedy-decode under a steer, read the continuation back |
| `weekdays_locate_scan` | layer × position interchange scan (one shared harvest) |
| `weekdays_das_sweep` | k × seed DAS fits at a located cell |
| `weekdays_das_apply` | apply a fitted rotation (ArtifactIdentity-checked) |
| `dbm_apply` | apply a fitted gate — DBM's held-out half of the pair above |

A **fit** document's saved score is its *training* score: `dbm.json`'s and
`weekdays_das_sweep.json`'s `iia.json` are computed over the split they trained
on. The held-out number comes from the matching `*_apply` document, or from the
fit's own `train_eval.json` when `train.eval` declares a split.

[`causalab/configs/runs/`](causalab/configs/runs/) holds the same experiment as a **split run document** (`application` + `method` in one file, spec §1.1), and [`causalab/configs/methods/`](causalab/configs/methods/) the reusable method on its own — the network- and data-independent half. [`causalab/configs/workflows/weekdays_8b.json`](causalab/configs/workflows/weekdays_8b.json) chains locate → select → fit → apply → plots as one workflow document (two step types: `intervention_protocol` and `script`).

## Repository layout

```
causalab/
├── protocol/        # engine-free document layer: load, validate, canonicalize,
│                    #   digest, sweep expansion, engine routing, workflow model, CLI
├── neural/
│   ├── shared/      # what every engine uses: sites, encoding, layouts,
│   │                #   mechanisms, featurizers, metrics, outputs, executor base
│   ├── engines/
│   │   ├── pytorch_hooks/    # the reference engine: hooks, decode, train loop
│   │   └── nnsight_tracing/  # the nnsight engine: traces (the 'nnsight' extra)
│   └── token_positions.py
├── analysis/        # numerical analysis a script step runs (fits, statistics, operands)
├── workflow/        # the workflow runner: run-tree overlay, script invocation, manifest
├── causal/          # causal model primitives
├── tasks/           # task definitions (causal models + counterfactual generators)
├── io/              # disk I/O + plotting primitives
└── configs/         # protocols/ (flat documents) + runs/ (split ones) +
                    #   methods/ + workflows/ — JSON, no Python config system
demos/               # one markdown demo per research question + its documents
docs/                # the two specs, demos.md, CODEBASE.md, TESTS.md, test_migration.md
tests/               # tiered suite — see docs/TESTS.md
```

## Core concepts

- **Causal model**: your hypothesis about how the LM solves a task — variables, values, parent–child dependencies, mechanisms (`causalab/causal/`).
- **Task**: a prompt distribution plus a causal model and counterfactual generators (`causalab/tasks/`).
- **Method / application**: the two halves one document may be written in — the method is what transfers (hypothesis, reads, writes, metrics, save), the application is what cannot (which network, which data, which addresses, which precision). One file is still one run: the halves compose into an ordinary protocol document, digest for digest.
- **Intervention protocol**: one experiment as data — which activations are read, which are edited (`swap`, `add_scaled`, `gaussian`, …), in which intervened models, scored by which metrics. Sweeps expand a document into a campaign of points with content-deduped shared work.
- **Workflow**: a chain of protocol executions plus script steps, with dependencies derived from references — never authored ordering. Everything a step declares is published where it lands; there is no save manifest.

## Tests

See [`docs/TESTS.md`](docs/TESTS.md). CPU tiers run with `uv run pytest -m "not golden"` (what CI runs). The `golden` tier runs real models on an accelerator: paper-provenance goldens (`tests/golden/`) and the chat-coherent drift pins (`tests/golden/drift/`).

## History

The Hydra runner, `analyses/` chains, `methods/` as Python and SLURM dispatch were retired in the protocol refactor (PR #20); [`docs/test_migration.md`](docs/test_migration.md) is the ledger. Their intervention cores return as shipped protocol documents and workflows. The notebook demos return as [`demos/`](demos/) — markdown around runnable documents, since a notebook's reason to exist was carrying its own execution.
