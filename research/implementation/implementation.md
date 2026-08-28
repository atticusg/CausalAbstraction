# Causalab implementation

> **Stale — predates the protocol refactor.** This document was written against
> the Hydra runner (`scripts/run_exp.sh`), the `causalab/analyses/` tree,
> `methods/` as Python, and SLURM dispatch, all of which were retired in PR #20.
> The scientific guidance remains correct, but the commands, configuration
> formats, and code paths are no longer valid. Rewriting it is tracked as future
> work. For the current
> protocol, start at [`../answer-research-question/answer-research-question.md`](../answer-research-question/answer-research-question.md).

Orientation for causalab plus the codebase how-to: what causalab is and which models
it supports, build new tasks / methods / analyses, run configured experiments, and
find your way around the codebase. This document routes to the references below — read
the matching one in full before starting. The scientific judgment for *what* to
build lives in the research protocol and supporting documents beside this one;
this document is *how* to realize it against the library.

**Read first:** [`causalab-overview.md`](causalab-overview.md) covers the
**model-support boundary** (causalab's harness loads transformer LMs; for other models
the design knowledge still transfers), **getting started** (onboarding notebooks + the
weekdays demo), the skill family, the analysis catalog, and how a run works.

All code and commands here run against the causalab library at
`~/.silico/libraries/causalab-internal` — `cd` there to run experiments or author new
task/method/analysis code. Silico pins and delivers it; treat it as a read-only
working checkout.

## Codebase overview

- [`causalab-overview.md`](causalab-overview.md) — the causalab codebase map: the
  model-support boundary, getting started, the skill family, key directories, how an
  experiment runs, the analyses catalog, and the end-to-end pipelines. Read on demand for
  orientation.

## Building blocks

- [`setup-task/setup-task.md`](setup-task/setup-task.md) — create or investigate a task
  package (causal model, counterfactuals, token positions, checker/metrics). Its
  instructions cover the package layout, the five task-quality objectives, and
  building a spec from an idea or a paper PDF.
- [`setup-methods/setup-methods.md`](setup-methods/setup-methods.md) — scaffold reusable
  interpretability primitives (featurizers, scorers, distances, …).
- [`setup-analyses/setup-analyses.md`](setup-analyses/setup-analyses.md) — scaffold
  research-question analysis wrappers (Hydra entry points).

## Running and debugging

- [`running-experiments.md`](running-experiments.md) — how to run a configured
  experiment on causalab: compose the runner config, pre-flight the tokenizer,
  dispatch (locally / slurm / fanned out), and verify artifacts. Takes an experiment
  to run, however specified — the stage sequencing and gating is owned by the caller
  (the silico planner/worker, or one of the workflow documents beside this one).
- [`ANALYSIS_GUIDE.md`](ANALYSIS_GUIDE.md) — the analysis catalog: dependency DAG, the
  "research question → analysis" table, per-analysis decisions, auto-discovery rules.
- [`COMMON_PROBLEMS.md`](COMMON_PROBLEMS.md) — recurring failure modes and their fixes.

## Report-format templates

Per-method output contracts for tone, structure, and figures:

- [`references/interchange-das-localization-report-format.md`](references/interchange-das-localization-report-format.md)
  — neural localization via interchange / activation-patching / DAS.
- [`../answer-research-question/hypothesis-generation/hypothesis-report-format.md`](../answer-research-question/hypothesis-generation/hypothesis-report-format.md)
  — causal-model certification, alongside the hypothesis-generation step.

## Narration

Report the engineering change and its result in plain sentences. Omit file paths and
step numbers.
