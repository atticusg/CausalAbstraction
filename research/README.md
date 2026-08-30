# causalab capability docs

These documents explain how to test whether an internal representation causes a
model behavior rather than merely correlating with it. Methods include causal
abstraction, interchange interventions, DAS, DBM, steering, and ablation.

## Start here

**[`answer-research-question/`](answer-research-question/answer-research-question.md)**
contains the six-step protocol:

1. [Behavioral analysis](answer-research-question/behavioral-analysis/behavioral-analysis.md) — can the model do it, and how does it fail?
2. [Exploratory experimentation](answer-research-question/exploratory-experimentation/exploratory-experimentation.md) — parallel logit lens, PCA, and counterfactual patching experiments
3. [Hypothesis generation](answer-research-question/hypothesis-generation/hypothesis-generation.md) — the causal model and the counterfactuals that test it
4. [Hypothesis testing](answer-research-question/hypothesis-testing/hypothesis-testing.md) — the hypothesis against its alternatives
5. [Generalize results](answer-research-question/generalize-results/generalize-results.md) — how far the claim reaches
6. [Save results](answer-research-question/save-results/save-results.md)

Read it first. It contains the flow chart, roadmap, and rules for responding to a
negative test.

## Silico entry point

Silico's **`silico-capabilities:causalab-pipeline`** skill directs the agent to
this directory in the installed CausaLab checkout. The scientific and execution
guidance lives here so it can change with the library it describes.

## Reference trees

The steps link to these supporting documents when needed:

- [`implementation/`](implementation/) explains how to use the causalab codebase.
  It covers building a task package, methods and analyses, running experiments,
  the analysis catalog, common failures, and the localization report format.
- [`explore-subspace/`](explore-subspace/) explains how to verify a supplied
  description of a subspace before designing causal experiments for it.
- [`subspace-causal-analysis-pipeline/`](subspace-causal-analysis-pipeline/)
  contains the older complete pipeline for deciding whether a given subspace is
  causal.

The implementation guides describe the current protocol runner. The
`explore-subspace` and older subspace pipeline documents still predate the protocol
refactor; use their scientific guidance, but verify every command against the
current implementation guides.

## Relationship to the causalab library

This directory is not a Python package. Most `.py` and `.yaml` files are templates
or examples. The standalone [`../scripts/run_hypothesis_generation.py`](../scripts/run_hypothesis_generation.py)
harness is the exception.

The `causalab/` package remains the source of truth for executable behavior. Its
protocol and workflow specifications are in [`../docs/`](../docs/), reusable
methods are in `causalab/configs/methods/`, and the worked workflow is
`causalab/configs/workflows/weekdays_8b.json`. Keep this research guidance aligned
with those interfaces.

## Entry points

Start with
[`answer-research-question/answer-research-question.md`](answer-research-question/answer-research-question.md)
for a new research question. Use the reference trees above when the question
starts from an existing subspace or requires changes to the CausaLab codebase.

## Remaining execution stub

The protocol refactor retired the Hydra runner, the `analyses/` tree, Python
methods, and SLURM dispatch. Missing replacements are marked with

> **Execution: stub.** …

The remaining stub is:

| Document | Waiting on |
|---|---|
| `behavioral-analysis/behavioral-analysis.md` | A protocol document that can decode a batch of prompts greedily and display the results. |

`hypothesis-testing/` has no stub: it runs on the shipped method presets in the
library's `causalab/configs/methods/` and the worked workflow in
`causalab/configs/workflows/weekdays_8b.json`.
