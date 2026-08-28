# Answer a research question

This six-step protocol turns a question about a language model's internal workings
into a supported claim. This file defines the order, roadmap, and routing between
steps.

Read this file before entering any step.

## The flow

```
                    ┌──────────────────────────────┐
                    │     behavioral analysis      │
                    │   can the model do it, and   │
                    │   how does it fail?          │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
        ┌──────────▶│ exploratory experimentation  │
        │           │   cheap probes for signal    │
        │           └──────────────┬───────────────┘
        │                          │
        │                          ▼
        │           ┌──────────────────────────────┐
        │  ┌───────▶│    hypothesis generation     │
        │  │        │  a causal model + the        │
        │  │        │  counterfactuals to test it  │
        │  │        └──────────────┬───────────────┘
        │  │                       │
        │  │                       ▼
        │  │        ┌──────────────────────────────┐
        │  └────────┤      hypothesis testing      │
        │           │   the hypothesis against     │
        └───────────┤   its alternatives           │
                    └──────────────┬───────────────┘
                                   │  strong positive result
                                   ▼
                    ┌──────────────────────────────┐
                    │      generalize results      │
                    │   how far does the claim     │
                    │   actually reach?            │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │         save results         │
                    └──────────────────────────────┘
```

| Step | Document | The question it answers |
|---|---|---|
| 1 | [`behavioral-analysis/`](behavioral-analysis/behavioral-analysis.md) | Can the model do this at all, and what does it get wrong? |
| 2 | [`exploratory-experimentation/`](exploratory-experimentation/exploratory-experimentation.md) | What cheap internal evidence is there, before committing to a hypothesis? |
| 3 | [`hypothesis-generation/`](hypothesis-generation/hypothesis-generation.md) | What algorithm might the model be running, and what would distinguish it from the alternatives? |
| 4 | [`hypothesis-testing/`](hypothesis-testing/hypothesis-testing.md) | Does the evidence favor this hypothesis over its alternatives? |
| 5 | [`generalize-results/`](generalize-results/generalize-results.md) | How far does the claim reach beyond the setting it was tested in? |
| 6 | [`save-results/`](save-results/save-results.md) | What gets written down, and where? |

## Start with a roadmap

Before running an experiment, copy
[`ROADMAP_TEMPLATE.md`](ROADMAP_TEMPLATE.md) into your working directory as
`ROADMAP.md` and plan each step.

State a question that has a definite answer. For example, ask "is the answer
computed at the final token or carried from the operands?" rather than "how does
the model do X?" Also record what the answer would change, what you expect from
each step, when you can skip a step, and what result would stop the project.

After every step, update the plan and append what happened, how it differed from
your expectation, and what changes next. Never edit earlier log entries.

## A step is a phase, not an experiment

Each step is a phase that may contain several experiments. Divide it into units in
the roadmap. Each unit must produce a result that you can record.

"Understand the attention pattern" is not a unit. "Ablate each head at the final
token and record which ones change the output" is. Record null results so they are
not repeated.

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
