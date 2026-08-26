# Step 4 — Hypothesis testing

Test the hypothesis **against its alternatives**. Not against nothing — against
the specific rival accounts that hypothesis generation identified and certified as
distinguishable. A result that beats the null and nothing else is the most common
way to produce a confident wrong answer.

This is the step where the compute goes, and it is the only step with a branch out
of it. Read the routing section at the bottom before you start, so you know what
the possible outcomes are while you are still able to design for them.

## What "against its alternatives" means

You come in with a target hypothesis (a causal model plus a variable subset), a set
of alternatives, and — from Step 3d — the knowledge of which counterfactual
datasets can separate which pairs. The experiment has to be run on a dataset that
*has power for the contest you care about*. Running the headline test on your
broadest dataset is the default mistake: it will confound the target with exactly
the alternatives you most wanted to rule out.

For each contest, use the dataset that Step 3d showed deconfounds it, and report
the result next to the CPU baseline for that pair. A neural score of 0.70 where an
alternative sat at 0.70 on the CPU matrix is not evidence for the target.

## The methods

The nine shipped presets are in `causalab/configs/methods/`. Each is a complete
protocol document: copy one, point it at your model and dataset, and adjust.

| Preset | What it does | Use it when |
|---|---|---|
| `interchange` | Swap an activation from the counterfactual into the base at a fixed site and position; score with `match` (IIA) and `logit_diff` | The core test. You have a specific location in mind and want to know whether the variable is there. |
| `weekdays_locate_scan` | Layer × position interchange scan over one shared harvest | You do not yet know where to look. This is the localization sweep. |
| `path_patching` | Sender → receiver patching with off-path freezing | The question is about a *path* between components, not a single site. |
| `das` | Train an orthogonal rotation so a k-dimensional subspace carries the variable, then interchange in that subspace | The variable is not axis-aligned. This is the discovery method: it *learns* which subspace mediates. |
| `weekdays_das_sweep` | k × seed DAS fits at a located cell | Choosing the rank, and checking the fit is not seed-luck. |
| `weekdays_das_apply` | Apply a fitted rotation, identity-checked | Evaluating a learned subspace on held-out data. |
| `dbm` | Differential binary masking through a trained gate | You want to know *which dimensions* carry it, rather than assuming a dense subspace. |
| `hydra_effect` | Resample-ablation plus downstream direct-effect probes | Measuring what a component's removal does downstream, accounting for compensation. |
| `harvest` | Activation harvesting at named sites and positions | A building block — feeding a fit, a mean reference, or an external analysis. |

## Sequencing: locate, then fit, then apply

The standard shape of a full test is a chain, and it is worth running it as a
**workflow document** rather than three commands, because the dependencies are
then derived and recorded rather than remembered:

```
locate   →  interchange scan over a layer × position grid
select   →  pick the best cell
fit      →  DAS sweep over k × seed at that cell
select   →  pick the winning (k, seed)
apply    →  evaluate that rotation on the held-out split
plot     →  the scan heatmap and the IIA-vs-k curve
```

`causalab/configs/workflows/weekdays_8b.json` is exactly this, end to end, and is
the worked example to copy. The runner derives step order from references — you
never author the ordering — and the artifact overlay lets `apply` resolve `fit`'s
output by path. A selection that matches no entry, or more than one, is a load
error before anything runs.

Before running: `causalab explain <doc>` reports the forward plan, the point count,
and the save products, and `causalab validate <doc> --data` runs the load-error
checklist including column references. Both are cheap and catch most of what
otherwise fails an hour into a sweep.

## The controls that make it a test

**A positive control.** Something you already know the answer to, run through the
same pipeline. Full mediation — transplanting the whole output representation —
should score at ceiling. If it doesn't, the pipeline is broken and nothing else in
the run means anything. Run it first.

**A negative control.** The null, and a site where the variable should not be. A
method that scores well everywhere is measuring something other than what you
think.

**Held-out evaluation, for anything trained.** DAS and DBM both fit parameters, so
both can overfit, and a rotation trained and evaluated on the same distribution
will report a number that does not survive contact with new inputs. Use the eval
split that Step 3c defined, and report train and eval scores side by side. The gap
is part of the result.

**Seed variation.** A single DAS fit is one draw. The `weekdays_das_sweep` shape —
k × seed — exists because the seed-to-seed variance is frequently comparable to the
effect being reported.

**Rank discipline.** A higher-dimensional subspace can always fit better; at
sufficient rank DAS will find something regardless of whether the model uses it.
Report IIA against k and argue for the rank you chose from that curve, not from the
best number.

## Reading the result

- **Compare to the CPU baseline, always.** Step 3d told you what score would
  indicate confounding on this dataset. Put the two numbers next to each other.
- **The ceiling is full mediation, not 1.0.** Normalize against what the positive
  control achieved.
- **A high IIA at many locations is a finding about your dataset**, usually that it
  lacks the power to separate locations, rather than a finding that the variable is
  everywhere.
- **Beating the null is the floor, not the result.** State explicitly which
  alternatives were ruled out and which remain live. The remaining live alternatives
  are what determines whether you go to Step 5 or back into the loop.

## Typical units of work

- One unit for the positive and negative controls.
- One unit per contested alternative — each is a separate comparison and often a
  separate dataset.
- One unit per stage of the locate → fit → apply chain.
- One unit per robustness check you commit to (seeds, ranks, held-out split).

## Reporting

The fixed format for localization results — the positive-control discipline, the
variable-by-variable structure, and the plot-type rules — is at
[`../../implementation/references/interchange-das-localization-report-format.md`](../../implementation/references/interchange-das-localization-report-format.md).

## Routing out of this step

**Strong positive result** — the target beat its alternatives on a dataset with the
power to separate them, the controls behaved, and the held-out numbers held up. Go
to [`../generalize-results/generalize-results.md`](../generalize-results/generalize-results.md).

**No strong positive result** — record the negative result, then choose a door:

- **Back to [hypothesis generation](../hypothesis-generation/hypothesis-generation.md)**
  if the evidence you already have suggests another hypothesis worth proposing. A
  clean refutation usually does: the observations that motivated the dead
  hypothesis generally motivate a live one. This is also the door for "the test
  couldn't separate two candidates" — that is a counterfactual-design problem,
  fixed by building a sharper narrow dataset, not by running the test again.
- **Back to [exploratory experimentation](../exploratory-experimentation/exploratory-experimentation.md)**
  if it doesn't. If the existing observations no longer support any hypothesis you
  haven't already tried, the next hypothesis would be a guess. Go get new signal
  first.

**Before either**, rule out the boring explanation: an underpowered or broken test
produces the same null as a wrong hypothesis. If the positive control did not hit
ceiling, if the sample size was small, if the metric had no headroom, or if the
dataset had no power for this contest per Step 3d — then you have not tested
anything yet and the fix is to re-run this step properly, not to re-route.

Write the negative result and the routing decision into `ROADMAP.md`. Refutations
belong in the final writeup regardless of how the investigation ends.
