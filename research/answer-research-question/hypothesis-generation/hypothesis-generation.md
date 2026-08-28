# Step 3 — Hypothesis generation

Build two things on a CPU: a runnable **causal model** of the algorithm you think
the network uses, and **counterfactual datasets** that distinguish it from
alternatives. A hypothesis without such a dataset is not testable. This step does
not use the target network.

## What a hypothesis is here

A hypothesis pairs a causal model with **a subset of its variables** and claims
that a specific neural location represents them. "The model computes a carry bit"
is too broad; name the location and distinguish the carry bit from the operands and
answer.

Two reference hypotheses are always in play and bound what any result can mean:

- **The null** does not intervene on anything. If your hypothesis cannot be
  distinguished from doing nothing, no experiment can test it.
- **Full mediation** transplants the entire output representation. It establishes
  the highest score that the method can reach.

## Step 3a: draft the candidate causal model(s)

Write one or more runnable causal models. Define inputs as leaf variables,
intermediates from explicit parents, and a prompt and output variable. Use
intermediates such as a carry bit or equality to expose the parts you want to test.

Start from the closest analogue among the shipped tasks rather than from nothing:
positional decomposition (`causalab/tasks/entity_binding/`), boolean logic
(`hierarchical_equality/`), modular arithmetic (`natural_domains_arithmetic/`),
geometric structure (`graph_walk/`). Say which pattern yours resembles and where
it differs.

**Store one value in each variable.** Do not combine conceptual units in a list,
dictionary, or tuple because each must be independently localizable. The entity
binding task uses separate `entity_g{g}_e{e}`, `query_e{e}`,
`positional_query_e{e}`, and `positional_answer` variables. Use a factory function
to generate any variable family.

**Handle every input combination.** Interchange can construct combinations that
the sampler never produces, such as an entity token in the wrong domain. See
`examples/unified_arithmetic/README.md` for a complete encoder.

**Check for dead ends.** Read the causal model's parent relationships and confirm
that every intermediate in a hypothesis contributes to the output. If the output
is computed without reading an intermediate, no dataset can distinguish an
intervention on that intermediate from the null hypothesis.

**Check the five task quality objectives** — granularity, grading totality, input
determinism, single-token decoding, value coverage — at
[`../../implementation/setup-task/instructions/task_quality_objectives.md`](../../implementation/setup-task/instructions/task_quality_objectives.md).

Scaffolds to edit: [`templates/models.py`](templates/models.py).

## Step 3b: curate the competing hypotheses

Choose distinct variable subsets as hypotheses. Mark the focal subsets as
**targets** and compare them with the rest. A target may group related variables
that could share a location.

Prune with architecture reasoning. Causal attention means information flows left to
right, so a subset is only realizable at a location at or after the tokens carrying
that information. This typically removes more of the hypothesis space than any
other single consideration.

The null and full-mediation references are added automatically; you do not list
them by hand.

## Step 3c: design the counterfactual datasets

A counterfactual dataset contains (base, counterfactual) input pairs. Its design
determines which hypotheses it can distinguish. Build all three types:

- **Wide** datasets use random resampling with balancing appropriate to the task,
  or a systematic manipulation such as swapping order, shuffling, or holding the
  template fixed while resampling its contents. They provide broad coverage and
  remain useful when the causal model turns out to be wrong.
- **Narrow** datasets use targeted pairs that hold one variable fixed while
  changing another. Design each narrow dataset to separate two specific hypotheses
  that another dataset confounds.
- **Single token** datasets contain a base input and counterfactual that differ by
  exactly one token representing a single input variable. They distinguish few
  hypotheses because moving one thing leaves most things confounded. With
  everything else fixed, however, they trace
  one piece of information through the network. Build one for any variable you
  intend to follow in a downstream localization.

Each example must contain exactly one counterfactual input. For every dataset,
record its type, **train/eval split**, and what evaluation holds out. Supervised
methods must use this split so that you can detect overfitting.

Scaffold to edit: [`templates/counterfactuals.py`](templates/counterfactuals.py).

## Step 3d: certify distinguishability

Before using a GPU, run each hypothesis's intervention on every pair and compare
the outputs. This CPU check shows which hypotheses each dataset can distinguish.

Two outputs to produce:

- **Per dataset, a target-centric table.** For each target, the rate at which each
  alternative's intervened output differs from the target's, plus the target's own
  rate against the null and against full mediation.
- **From one large random sample** (100,000 pairs is a reasonable default), the
  groups of hypotheses that no sampled pair distinguishes.

### Reading the numbers

Use these numbers as baselines for interpretation, not as pass or fail thresholds.

- **~0.50 on a wide dataset is partial separation.** A dataset may confound two
  alternatives with each other and still cleanly separate both from the target,
  which is frequently the intent.
- **A score near zero against the null means that the dataset cannot test that
  target.** A narrow dataset built to test one target will often read zero for a
  different target. This is expected and does not indicate a problem.
- **Carry the baselines into interpretation.** If an alternative sits at 0.70
  against the target on a dataset, then a later neural result of ~0.70 on that
  dataset is confounded with that alternative. This is the number you will want
  when reading the localization, so record it.
- **Distinguish two kinds of confounding.** If a pair of hypotheses receives 0.00
  on one dataset, design a more targeted narrow dataset. If no pair in the large
  random run distinguishes two hypotheses, they are **confounded everywhere** in
  the sampled data. Keep one as the representative hypothesis. If no pair
  distinguishes a hypothesis from the null, the sampler cannot test it. Drop the
  hypothesis or change the sampler.
- **Failure to distinguish is evidence, not proof.** A rare pair may distinguish
  them even if the sample did not.

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

### Run the check

Put the completed `models.py` and `counterfactuals.py` files in one directory,
then run the harness from the CausaLab repository root:

```bash
uv run python scripts/run_hypothesis_generation.py "$WORKDIR/HYPOTHESES"
```

The harness calls CausaLab's `distinguishability_report`, adds the null and
full-mediation references when they are absent, runs every designed dataset, and
runs the broad random check with 100,000 pairs. It writes
`$WORKDIR/HYPOTHESES/distinguishability.json` and prints a compact summary. Use
`--n`, `--random-n`, `--seed`, or `--output` when the defaults do not fit the
investigation.

The harness imports and executes the two Python files, so only run it on modules
you trust.

## If you are critiquing an existing task

For a task in `causalab/tasks/`, test whether its existing counterfactuals
distinguish the hypotheses you care about. Import its `CAUSAL_MODEL`, declare the competing
hypotheses over it, wrap its shipped `COUNTERFACTUAL_GENERATORS`, and point the
random generator at its random counterfactual dataset. Then run 3d as normal.

Report which comparisons the datasets can make, what narrow dataset would resolve
each remaining ambiguity, and which hypotheses no pair distinguishes.
`hierarchical_equality` (one wide dataset; `result`,
`{left,right}`, and `output` confounded on every pair) and `entity_binding` (where
`swap_query_group` sharply deconfounds the positional target) are the reference
walk-throughs.

## Building the task package

If the hypothesis is going to be tested, the causal model and generators have to
become a real task package with serialized dataset tables. That is mechanics rather
than science, and it lives in
[`../../implementation/setup-task/setup-task.md`](../../implementation/setup-task/setup-task.md).
The dataset tables are generated files. Store information that varies by row or
describes the task, such as answer forms and position anchors, in columns when
building the table. Do not compute that information inside a protocol document.

## Typical units of work

- One unit per candidate causal model drafted.
- One unit per counterfactual dataset designed and implemented.
- One unit for the distinguishability matrix, plus one per iteration when a
  confound sends you back to design a sharper dataset.
- One unit for the hypotheses document.

## Write it down

Fill [`HYPOTHESES_TEMPLATE.md`](HYPOTHESES_TEMPLATE.md) and save it as
`$WORKDIR/HYPOTHESES.md`. Include the models, hypotheses, datasets and splits,
distinguishability matrix, and interpretation.

If this step produces a standalone report, [`hypothesis-report-format.md`](hypothesis-report-format.md)
is the fixed format downstream readers expect.

## Handoff

Update `ROADMAP.md`, then go to
[`../hypothesis-testing/hypothesis-testing.md`](../hypothesis-testing/hypothesis-testing.md)
with a hypothesis that is certified distinguishable from at least one alternative
you care about.
