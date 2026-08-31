# Answer a research question

This six-step protocol turns a question about a language model's internal workings
into a supported claim. This file defines the order, roadmap, and routing between
steps.

The goal is to identify meaningful intermediate causal variables inside the
model. Behavioral analysis establishes a task the model can perform and describes
its errors. Exploration produces competing guesses about the internal process.
Hypothesis generation expresses those guesses as explicit high-level causal
models, and hypothesis testing uses counterfactual datasets to test individual
intermediate variables.

Read this file before entering any step.

Read [`../causal-handbook.md`](../causal-handbook.md) for the scientific framework
behind intermediate variables, causal models, counterfactual datasets, and the
limits of each intervention method.

These steps are dependent phases. They are not six work streams that can begin
together. Behavioral analysis is the first gate: complete it before launching
exploratory experiments, generating hypotheses, or starting any later work. A
phase may launch its own independent units in parallel, but only while that phase
is active. The next phase stays blocked until the active phase is complete.

## The flow

```
behavioral analysis              required first; blocks all later phases
        │
        ▼
identify critical tokens         blocks every exploratory experiment
        │
        ▼
exploratory experimentation      launch five initial experiments in parallel
        ├── logit lens
        ├── PCA
        ├── residual stream patching ──▶ residual stream DAS
        ├── attention output patching ─▶ attention head DBM
        └── MLP output patching ───────▶ MLP neuron DBM
        │
        │ each follow-up waits only for its parent patching experiment
        │ all initial and applicable follow-up experiments must finish
        ▼
hypothesis generation            explicit causal models and variables
        │
        ▼
hypothesis testing               counterfactual tests of individual variables
        │ strong evidence
        ▼
generalize results
        │
        ▼
save results
```

Exploratory jobs on separate branches may run concurrently. Residual stream DAS,
attention head DBM, and MLP neuron DBM each start as soon as their own parent
patching experiment identifies signal. Hypothesis generation cannot begin until
the exploratory phase and its table of candidate variables are complete.

The return paths after a failed or ambiguous hypothesis test are described under
"Routing after hypothesis testing" below.

| Step | Document | The question it answers |
|---|---|---|
| 1 | [`behavioral-analysis/`](behavioral-analysis/behavioral-analysis.md) | Can the model do this at all, and what does it get wrong? |
| 2 | [`exploratory-experimentation/`](exploratory-experimentation/exploratory-experimentation.md) | What candidate intermediate variables and neural locations does the initial evidence suggest? |
| 3 | [`hypothesis-generation/`](hypothesis-generation/hypothesis-generation.md) | Which explicit causal models and intermediate variables could explain the evidence? |
| 4 | [`hypothesis-testing/`](hypothesis-testing/hypothesis-testing.md) | Does a counterfactual test support an individual intermediate variable over its alternatives? |
| 5 | [`generalize-results/`](generalize-results/generalize-results.md) | How far does the claim reach beyond the setting it was tested in? |
| 6 | [`save-results/`](save-results/save-results.md) | What gets written down, and where? |

## Start with a roadmap

Before running an experiment, copy
[`ROADMAP_TEMPLATE.md`](ROADMAP_TEMPLATE.md) into your working directory as
`ROADMAP.md` and plan each step.

State a question that has a definite answer. For example, ask "is the answer
computed at the final token or carried from the operands?" rather than "how does
the model do X?" Also record what the answer would change, what you expect from
each step, the routing conditions for later phases, and what result would stop
the project. Behavioral analysis is never optional.

After every step, update the plan and append what happened, how it differed from
your expectation, and what changes next. Never edit earlier log entries.

During exploration, maintain the table of candidate intermediate variables in
`ROADMAP.md`. Record competing guesses, supporting and conflicting observations,
possible neural locations, the experiment that could distinguish them, and their
current status. Do not delete rejected candidates.

## A step is a phase, not an experiment

Each step is a phase that may contain several experiments. Divide it into units in
the roadmap. Each unit must produce a result that you can record.

"Understand the attention pattern" is not a unit. "Ablate each head at the final
token and record which ones change the output" is. Record null results so they are
not repeated.

Units within the active phase may run concurrently. For example, behavioral
analysis may evaluate several prompt formats in separate GPU jobs within one
experiment. This does not unblock exploratory experimentation: every unit still
belongs to behavioral analysis, and that phase must finish before the pipeline
advances.

After that gate passes, identify critical token locations. Then launch logit lens,
PCA, residual stream patching, attention output patching, and MLP output patching
in parallel. Each reuses the selected prompt, model configuration, code, and
evaluation setup from behavioral analysis. The three learned localization jobs
are individually gated by their parent patching results. Hypothesis generation
remains blocked until the exploratory phase is complete.

## Routing after hypothesis testing

**Strong positive result** — the hypothesis beat its alternatives on a test with
the power to separate them. Go to generalize results.

If the result is null, ambiguous, or contradicts the hypothesis, choose between
the following two paths:

- **Return to hypothesis generation** when the evidence suggests another
  hypothesis, or when the counterfactual dataset could not distinguish two
  candidates.
- **Back to exploratory experimentation** when it doesn't. Generating a hypothesis
  without new evidence would be guessing, so collect more evidence first.

If you are about to propose a third hypothesis without new evidence, return to
exploratory experimentation.

Record the result and the routing decision in `ROADMAP.md` first — a refutation is
a finding and belongs in the writeup.

## Where causalab lives

These documents are prose; the code is the **causalab** library, cloned by the Lab
under `~/.silico/libraries/causalab-internal`. Paths naming package modules,
presets, or specs are relative to that checkout. Run causalab commands from there
and treat it as read-only.

## Working directory

Store everything under one working directory. Set `$WORKDIR` to its absolute path:

```
$WORKDIR/
├── ROADMAP.md          the plan and its revision log
├── docs/               protocol and workflow documents you author
├── data/               serialized dataset tables
├── runs/               run trees produced by execution
└── result/             the writeup and the figures it references
```

## What is not in the six steps

The steps also link to implementation guidance and two older research paths.

- [`../implementation/`](../implementation/) — the causalab codebase: task
  packages, protocol and workflow authoring, execution, and common failures. These
  guides describe the current protocol runner.
- [`../explore-subspace/`](../explore-subspace/) — verifying a subspace's
  description before designing experiments against it. Its commands predate the
  protocol refactor.
- [`../subspace-causal-analysis-pipeline/`](../subspace-causal-analysis-pipeline/)
  — the older end-to-end path for a *given* subspace. Its commands also predate the
  protocol refactor.

## Execution stubs

The protocol refactor retired the Hydra runner, the `analyses/` tree, `methods/`
as Python, and SLURM dispatch. These documents preserve the scientific guidance
and mark each place where a concrete command used to appear:

> **Execution: stub.** What this needs, and why it isn't written yet.

[`../README.md`](../README.md) indexes them.
