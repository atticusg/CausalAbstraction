# Answer a research question

This six-step protocol turns a question about a language model's internal workings
into a supported claim. This file defines the order, roadmap, and routing between
steps.

The goal is to build a causal account of the computation at every layer of the
model. The account should explain how relevant input tokens enter the
computation, how attention and MLP layers move or transform that information,
which meaningful intermediate variables appear, and how the output forms.
Behavioral analysis establishes a task the model can perform. Exploration traces
the input and output variables and produces competing guesses about the internal
process. Hypothesis generation expresses those guesses as explicit causal models.
Hypothesis testing uses counterfactual datasets to test one intermediate variable
through the same six intervention methods used during exploration.

The pipeline is designed for tasks whose important input and output locations can
be identified. The original prompt may look natural. Before internal analysis,
turn it into a controlled task with named variables, token or span rules, and
counterfactual examples that remain as close as practical to the original input.
The scientific leverage comes from comparing nearby inputs whose small,
understood differences reveal causal structure.

Read this file before entering any step.

Read [`../causal-handbook.md`](../causal-handbook.md) for the scientific framework
behind intermediate variables, causal models, counterfactual datasets, and the
limits of each intervention method.

These steps are dependent phases. Behavioral analysis is the first gate, and
exploration must finish before hypothesis generation begins. After exploration,
the dependency is tracked per hypothesis rather than by one global barrier.
Generate datasets for many intermediate variables in parallel. As soon as one
variable's datasets are ready, launch its six hypothesis-testing experiments even
while dataset work continues for other variables. Iterate between generation and
testing until the evidence supports a small set of main claims.

## The flow

```
behavioral analysis              establish the behavior and controlled task
        │
        ▼
decompose the output             one child investigation per meaningful target
        │
        ▼
identify critical tokens         blocks every exploratory experiment
        │
        ▼
exploratory experimentation      complete before hypothesis work
        ├── logit lens
        ├── PCA
        └── six intervention experiments:
            ├── residual stream patching ──▶ residual stream DAS
            ├── attention output patching ─▶ attention head DBM
            └── MLP output patching ───────▶ MLP neuron DBM
        │
        │ each follow-up waits only for its parent patching experiment
        │ all initial and applicable follow-up experiments must finish
        ▼
hypothesis loop                  many variables proceed concurrently
        ├── generate datasets for variable A ──▶ six tests for A ──┐
        ├── generate datasets for variable B ──▶ six tests for B ──┤ revise
        └── generate datasets for variable C ──▶ six tests for C ──┘ and repeat
        │
        │ select the small set of main claims
        ▼
generalize main claims           prompt templates, related tasks, natural text
        │
        ▼
final synthesis                  LaTeX, PDF, figures, complete experiment index
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
`ROADMAP.md`. Also create `INTERMEDIATE_VARIABLE_IDEAS.md` from
[`INTERMEDIATE_VARIABLE_IDEAS_TEMPLATE.md`](INTERMEDIATE_VARIABLE_IDEAS_TEMPLATE.md)
and `REPORT_PLAN.md` from
[`REPORT_PLAN_TEMPLATE.md`](REPORT_PLAN_TEMPLATE.md).

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

Maintain the layer-by-layer causal account in the same roadmap. Include every
attention and MLP layer, relevant token positions, supported variables, direct
intervention evidence, and unresolved gaps. Update it after exploration,
hypothesis testing, and every generalization experiment. The final report should
explain this table, not reconstruct the model's story from scattered reports.

Maintain `INTERMEDIATE_VARIABLE_IDEAS.md` from the beginning. It may contain
speculative ideas, but every idea must have a visible status. Use it to decide
which labels PCA should display and which symbols or answer candidates logit lens
should inspect. Do not treat an entry in this document as evidence.

Maintain `REPORT_PLAN.md` throughout the investigation. Record candidate main
claims, decisive experiments and figures, likely appendix material, null results,
and evidence still needed. Generalization should focus on the subset of main
claims for which a transfer test is meaningful. The final report is assembled
only after those generalization experiments finish.

## Multi-token outputs

If the output contains several tokens, do not treat the whole generation as one
undifferentiated target. Follow
[`behavioral-analysis/output-decomposition.md`](behavioral-analysis/output-decomposition.md).

For a standardized sequence, create a shared parent setup and one child
investigation for each meaningful output token or semantic slot. Analyze each
target first while conditioning on the correct preceding output tokens. Treat
those preceding tokens as explicit input variables. Later, repeat selected tests
with the model's own generated prefix to measure error propagation.

For free-form text, do not create one investigation for every literal token.
Decompose the generation into scientific subquestions that each end in one
next-token prediction. Identify the target by semantic role when its absolute
position varies. If no stable semantic target and nearby counterfactual can be
defined, narrow the question before proceeding.

## A step is a phase, not an experiment

Each step is a phase that may contain several experiments. Divide it into units in
the roadmap. Each unit must produce a result that you can record.

"Understand the attention pattern" is not a unit. "Ablate each head at the final
token and record which ones change the output" is. Record null results so they are
not repeated.

Independent units may run concurrently. For example, behavioral analysis may
evaluate several prompt formats in separate GPU jobs within one experiment. This
does not unblock exploratory experimentation: behavioral analysis must finish
before the pipeline advances. After exploration, use the dependency for each
intermediate variable instead of waiting for all work in a named phase.

After that gate passes, identify critical token locations and enumerate the input
variables and output variable. Then launch logit lens, PCA, residual stream
patching, attention output patching, and MLP output patching in parallel. Reuse
each patching result to evaluate every applicable input variable and the output
variable. The three learned localization jobs are individually gated by their
parent patching results and use separate supervised training jobs for each
variable. Hypothesis generation remains blocked until the exploratory phase is
complete.

Hypothesis generation runs one dataset-design experiment for each proposed
intermediate variable. Launch many of these experiments in parallel. As soon as
one is complete, hypothesis testing runs residual stream patching,
attention output patching, MLP output patching, residual stream DAS, attention
head DBM, and MLP neuron DBM for that variable. Treat these as six distinct
experiments and run them concurrently whenever their data and location inputs are
ready. Testing for one variable may overlap with hypothesis generation for other
variables. Together, their reports should extend the layer-by-layer causal
account rather than return six disconnected scores.

## Routing after hypothesis testing

**Strong positive result** — the hypothesis beat its alternatives on a test with
the power to separate them. Add it to `REPORT_PLAN.md` as a candidate main claim,
then continue other useful generation and testing work. Do not send every positive
result directly to generalization.

If the result is null, ambiguous, or contradicts the hypothesis, choose between
the following two paths:

- **Return to hypothesis generation** when the evidence suggests another
  hypothesis, or when the counterfactual dataset could not distinguish two
  candidates.
- **Back to exploratory experimentation** when it doesn't. Generating a hypothesis
  without new evidence would be guessing, so collect more evidence first.

If you are about to propose a third hypothesis without new evidence, return to
exploratory experimentation.

After enough iterations to support a coherent causal account, select the small
set of main claims in `REPORT_PLAN.md`. Generalize only those claims, or the subset
for which a particular generalization test is meaningful. Assemble the final
report after generalization, not during the generation and testing loop.

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
├── INTERMEDIATE_VARIABLE_IDEAS.md
├── REPORT_PLAN.md
├── OUTPUT_TARGETS.md    required when the output has several meaningful targets
├── output-targets/     child investigations for meaningful output targets
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
