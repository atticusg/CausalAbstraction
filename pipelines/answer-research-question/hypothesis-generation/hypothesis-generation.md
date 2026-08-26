# Step 3 — Hypothesis generation

Turn the observations from exploratory experimentation into a **causal model** —
an explicit, runnable account of the algorithm you think the network implements —
together with the **counterfactual datasets** that could tell that account apart
from its rivals.

Two products, and the second is the one people skip. A hypothesis with no
counterfactual that separates it from an alternative is not testable, and
discovering that after the GPU time is spent is the expensive way to learn it.
Everything in this step runs on CPU and never touches the target network.

## What a hypothesis is here

A hypothesis is a high-level causal model **plus a subset of its variables**,
eventually claimed to be a faithful abstraction of a specific neural location. The
variable subset is the part that carries the content: "the model computes a carry
bit" is not a hypothesis until you say that the carry bit — and not the operands,
and not the answer — is what is represented at some place in the network.

Two reference hypotheses are always in play and bound what any result can mean:

- **The null** — intervene on nothing. If your hypothesis cannot be distinguished
  from doing nothing, there is no experiment.
- **Full mediation** — transplant the entire output. The ceiling.

## Step 3a: draft the candidate causal model(s)

Write one or more runnable causal models. Inputs are leaf variables; intermediates
are derived from their explicit parents; every model defines a prompt variable and
an output variable. Usually you want intermediates that decompose the computation
— a carry bit, a matched group, an equality — because each intermediate is a thing
you can then hypothesize about.

Start from the closest analogue among the shipped tasks rather than from nothing:
positional decomposition (`causalab/tasks/entity_binding/`), boolean logic
(`hierarchical_equality/`), modular arithmetic (`natural_domains_arithmetic/`),
geometric structure (`graph_walk/`). Say which pattern yours resembles and where
it differs.

**One value per variable.** The core constraint: one variable per conceptual unit,
each holding a single value, never a list, dict, or tuple bundle. Each piece has to
be localizable on its own, and a bundled variable cannot be. Entity binding is the
exemplar — separate `entity_g{g}_e{e}`, `query_e{e}`, `positional_query_e{e}`,
`positional_answer` variables rather than one binding dict. To support an arbitrary
number of entities or positions, generate a *family* of variables in a loop inside
a factory function rather than a single list variable.

**Make mechanisms total.** When variables are coupled — an entity token only valid
within its domain, say — the mechanisms that read them must handle every
combination, because interchange routinely constructs input combinations the
sampler never would. `examples/unified_arithmetic/` shows both the
family-of-variables factory and the total-encoder pattern; see its `README.md`.

**Check for dead ends.** Any intermediate you intend to hypothesize about must lie
on a live path to the output. If the output is computed without reading it, no
dataset can distinguish it from the null.

**Check the five task quality objectives** — granularity, grading totality, input
determinism, single-token decoding, value coverage — at
[`../../implementation/setup-task/instructions/task_quality_objectives.md`](../../implementation/setup-task/instructions/task_quality_objectives.md).

Scaffolds to edit: [`templates/models.py`](templates/models.py).

## Step 3b: curate the competing hypotheses

Choose the variable subsets that form distinct hypotheses, and designate the focal
**targets** — usually a *group* rather than a single one, since as you move across
the transformer you expect co-occurring variables to share a location. Everything
else becomes an alternative measured against the targets.

Prune with architecture reasoning. Causal attention means information flows left to
right, so a subset is only realizable at a location at or after the tokens carrying
that information. This typically removes more of the hypothesis space than any
other single consideration.

The null and full-mediation references are added automatically; you do not list
them by hand.

## Step 3c: design the counterfactual datasets

A counterfactual dataset is a set of (base, counterfactual) input pairs. Which
hypotheses a dataset can separate is a property of how its pairs are constructed,
so this is a design problem, not a sampling problem. Build at all three altitudes —
they do different jobs and you want all of them.

- **Wide** — random resampling under task-appropriate balancing, or a systematic
  manipulation (swap order, shuffle, hold the template and resample the infills).
  Broad coverage. These stay useful when the causal model turns out to be wrong,
  which is their main virtue.
- **Narrow** — sharply targeted pairs holding one variable fixed while flipping
  another, built to separate two specific hypotheses. Built on purpose, in response
  to a specific confound.
- **Single-token** — base and counterfactual differ by exactly one token realizing
  a single input variable. They distinguish few hypotheses, because moving one
  thing leaves most things confounded, but with everything else fixed they trace
  one piece of information through the network. Build one for any variable you
  intend to follow in a downstream localization.

Each generator returns an example with exactly one counterfactual input. For every
dataset record its altitude, its **train/eval split**, and what the eval split
holds out — entities within templates, or whole templates. Downstream supervised
localizers have to respect held-out data, and you need to be able to watch for
overfitting.

Scaffold to edit: [`templates/counterfactuals.py`](templates/counterfactuals.py).

## Step 3d: certify distinguishability

Before running anything on a GPU, check on CPU which of your hypotheses each
dataset can actually tell apart. This is what makes the step worth doing: it is a
pure causal-model computation — run each hypothesis's intervention on each pair,
see whether the resulting outputs differ — and it costs nothing.

Two outputs to produce:

- **Per dataset, a target-centric table.** For each target, the rate at which each
  alternative's intervened output differs from the target's, plus the target's own
  rate against the null and against full mediation.
- **From one large random run** (100,000 pairs is a reasonable default), the
  **always-confounded groups**: sets of hypotheses that no sampled pair
  deconfounds.

### Reading the numbers

They are interpretive baselines, not pass/fail gates.

- **~0.50 on a wide dataset is partial separation.** A dataset may confound two
  alternatives with each other and still cleanly separate both from the target,
  which is frequently the intent.
- **Near-zero against the null means the dataset has no power for that target.** A
  narrow dataset built to sharpen one target typically reads zero for a different
  target. Expected, not a problem.
- **Carry the baselines into interpretation.** If an alternative sits at 0.70
  against the target on a dataset, then a later neural result of ~0.70 on that
  dataset is confounded with that alternative. This is the number you will want
  when reading the localization, so record it.
- **Distinguish the two grades of confounding.** A per-dataset 0.00 is *fixable* —
  design a sharper narrow dataset. Two hypotheses that *no* pair in the large
  random run deconfounds are **confounded everywhere**: they are the same
  hypothesis, so pick one representative and stop. A hypothesis no pair
  deconfounds from the null is inert under the sampler — drop it or change the
  sampler.
- **Always-confounded is strong evidence, not proof.** A rare pair could still
  deconfound them at finite N.

If a contest you care about is confounded on every dataset but the hypotheses are
*not* confounded everywhere, go back to Step 3c and design a sharper narrow
dataset.

### Two things this check will not catch

- **It is only as broad as the pairs you sampled.** If deconfounding two hypotheses
  requires a particular kind of pair — cross-domain pairs, in the shared-calculator
  example — then the large random run must sample those, or genuinely distinct
  hypotheses will be reported as confounded.
- **Shared-mechanism hypotheses are invisible to it.** If your hypothesis is that
  one module is reused across conditions, every pair confounds "shared" with
  "split". That is settled by a cross-condition train/eval split — a localizer
  trained on one condition and evaluated on a held-out one — which is a
  generalization experiment, not this matrix.

> **Execution: stub.** The distinguishing primitive is
> `causalab.causal.causal_utils.can_distinguish_with_dataset`, and
> `find_live_paths` in the same module does the dead-end check from Step 3a. Both
> are library calls and still exist. What was retired is the `develop_hypothesis`
> analysis and the Hydra runner that drove it over a hypotheses directory and
> wrote `distinguishability.json`. There is no replacement harness — the matrix
> and the large random run currently have to be driven by hand against those
> primitives. Fill in once causalab stabilizes.

## If you are critiquing an existing task

When the task already exists in `causalab/tasks/`, you are diagnosing whether its
shipped counterfactuals distinguish the hypotheses you care about. Skip the
from-scratch authoring: import the task's `CAUSAL_MODEL` and declare the competing
hypotheses over it, wrap its shipped `COUNTERFACTUAL_GENERATORS`, and point the
random generator at its random counterfactual dataset. Then run 3d as normal.

The output is a diagnosis — which contests the shipped datasets deconfound, which
they leave confounded and what narrow dataset would fix each, and which hypotheses
no pair can deconfound at all. `hierarchical_equality` (one wide dataset; `result`,
`{left,right}`, and `output` confounded on every pair) and `entity_binding` (where
`swap_query_group` sharply deconfounds the positional target) are the reference
walk-throughs.

## Building the task package

If the hypothesis is going to be tested, the causal model and generators have to
become a real task package with serialized dataset tables. That is mechanics rather
than science, and it lives in
[`../../implementation/setup-task/setup-task.md`](../../implementation/setup-task/setup-task.md).
Note that the dataset tables are build products: everything per-row or
task-semantic — answer forms, per-row position anchors — is a column written when
the table is built, never computed inside a protocol document.

## Typical units of work

- One unit per candidate causal model drafted.
- One unit per counterfactual dataset designed and implemented.
- One unit for the distinguishability matrix, plus one per iteration when a
  confound sends you back to design a sharper dataset.
- One unit for the hypotheses document.

## Write it down

Fill [`HYPOTHESES_TEMPLATE.md`](HYPOTHESES_TEMPLATE.md) into `$WORKDIR` as
`HYPOTHESES.md`: the candidate models, the competing hypotheses and which are
confounded everywhere, the datasets with their roles and splits, and the
distinguishability matrix with your reading of it.

If this step produces a standalone report, [`hypothesis-report-format.md`](hypothesis-report-format.md)
is the fixed format downstream readers expect.

## Handoff

Update `ROADMAP.md`, then go to
[`../hypothesis-testing/hypothesis-testing.md`](../hypothesis-testing/hypothesis-testing.md)
with a hypothesis that is certified distinguishable from at least one alternative
you care about.
