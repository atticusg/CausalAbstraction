# Step 4 — Hypothesis testing

Test the hypothesis **against its alternatives**, not only against the null. A
result that differs from the null but cannot be distinguished from another
hypothesis does not support the target hypothesis.

## The test

You arrive with a target, alternatives, and the datasets that distinguish them.
Three rules make the experiment a valid test.

**Use a dataset that distinguishes the hypotheses.** For each comparison, use the
dataset identified in step 3 and report its CPU baseline. A neural score of 0.70 is
not evidence for the target if an alternative also scored 0.70.

**Run the controls.** Full mediation should reach the maximum score; if it does
not, stop. Use the null and a site that should not contain the variable as negative
controls. A method that scores well everywhere is measuring something else.

**Hold out data and vary the seed.** DAS and DBM can overfit. Report training and
evaluation results together, including variation across seeds. Choose rank from
the IIA versus `k` curve, not the highest score.

Compare the result with the CPU baseline and the positive control's measured
maximum, not with 1.0. State which alternatives remain.

## Running it

The library includes nine preset method documents at
`~/.silico/libraries/causalab-internal/causalab/configs/methods/`. Each is a
complete protocol document. Copy the closest preset, set its model and dataset,
and adjust the remaining fields.

| Preset | Use it when |
|---|---|
| `interchange` | The core test — you have a location in mind and want to know whether the variable is there. |
| `weekdays_locate_scan` | You don't yet know where to look: a layer × position scan. |
| `das` | The variable isn't axis-aligned. Learns which subspace mediates. |
| `weekdays_das_sweep` / `weekdays_das_apply` | Choosing the rank and seed, then evaluating the winner on held-out data. |
| `dbm` | Which *dimensions* carry it, rather than assuming a dense subspace. |
| `path_patching` | The question is about a path between components, not a single site. |
| `hydra_effect` | What a component's removal does downstream, accounting for compensation. |
| `harvest` | A building block — feeding a fit, a mean reference, or an external analysis. |

A full test locates the relevant area, selects a cell, fits several values of `k`
and seeds, selects a fit, evaluates held out data, and plots the result. Put these
stages in a **workflow document** so their dependencies are recorded.
`causalab/configs/workflows/weekdays_8b.json` is exactly this chain and is the
worked example to copy. `causalab explain <doc>` and `causalab validate <doc>
--data` are cheap and catch most of what otherwise fails an hour into a sweep.

Report against
[`../../implementation/references/interchange-das-localization-report-format.md`](../../implementation/references/interchange-das-localization-report-format.md).

## Typical units of work

- One for the positive and negative controls.
- One per contested alternative — each is a separate comparison, often on a
  separate dataset.
- One per stage of the locate → fit → apply chain.

## Routing out

**Strong positive result** → [`../generalize-results/generalize-results.md`](../generalize-results/generalize-results.md).

**Anything else** → back to
[hypothesis generation](../hypothesis-generation/hypothesis-generation.md) if the
evidence in hand already suggests another hypothesis worth proposing, or back to
[exploratory experimentation](../exploratory-experimentation/exploratory-experimentation.md)
if it doesn't.

First check whether the test worked. A test that lacks enough statistical power or
has a broken setup can produce the same null result as a wrong hypothesis. If the
positive control did not reach its expected maximum, the sample was too small, the
metric could not increase, or the dataset could not distinguish the competing
hypotheses, fix the test and repeat this step. Do not route to another step yet.

Write the negative result and the routing decision into `ROADMAP.md`. Refutations
belong in the writeup regardless of how the investigation ends.
