# Step 4 — Hypothesis testing

Test one proposed intermediate variable through the same six intervention methods
used during exploration. Compare the variable with the input, output, and
competing intermediate variables that the report from hypothesis generation
identified as plausible alternatives. A result that differs from the null but remains
consistent with one of those alternatives does not support the target.

## Inputs

Begin only when the intermediate variable has its own hypothesis-generation
experiment containing:

- a runnable causal model with the variable defined explicitly;
- broad and narrow counterfactual datasets;
- training and evaluation splits;
- CPU distinguishability baselines for every plausible alternative;
- `distinguishability.json` and `report.html`.

Reuse the selected prompt, model, critical token rules, locations, model-loading
code, and execution setup from exploration. Do not rebuild the task or rerun
exploration under a new setup.

## Six experiments per intermediate variable

Create one hypothesis-testing directory for the intermediate variable with six
separate experiments:

```text
$WORKDIR/hypothesis-testing/{intermediate-variable}/
├── residual-stream-patching/
├── attention-output-patching/
├── mlp-output-patching/
├── residual-stream-das/
├── attention-head-dbm/
├── mlp-neuron-dbm/
└── report.html
```

Launch all six as concurrently as their inputs and available compute allow. The
three patching experiments do not depend on each other. The DAS and DBM
experiments may start immediately because exploration already supplied candidate
locations and hypothesis generation supplied the supervised intermediate
variable and its datasets.

Several intermediate variables may be under test at once. Their six-experiment
groups are independent unless they share a required artifact explicitly. Do not
serialize testing across variables merely to preserve the visual order of the
roadmap.

Within each method experiment, expand independent locations, layer bands,
dimensions, regularization values, datasets, and seeds into separate CausaLab
jobs. Run those jobs in parallel and shard them with nonoverlapping `--points`
ranges. They remain one research experiment because they use one method to answer
one question about the intermediate variable.

### 1. Residual stream patching

Patch the complete residual stream at every candidate layer and token location.
For spans, patch each position separately and the whole span jointly. Use the
broad and narrow counterfactual datasets designed for this intermediate variable.

### 2. Attention output patching

Patch complete attention outputs over the candidate five-layer and ten-layer
bands from exploration. Test every relevant token position and span.

### 3. MLP output patching

Patch complete MLP outputs over the candidate five-layer and ten-layer bands from
exploration. Test every relevant token position and span.

For all three patching experiments, execute each model intervention once and save
the output and logits. Reuse that saved result to score the target intermediate
variable and every plausible input, output, or intermediate alternative. Do not
repeat the same forward pass to change only the variable used for evaluation.

### 4. Residual stream DAS

Train DAS with the intermediate variable from the causal model as its supervision.
Do not substitute an input token or the output label. Fit the candidate residual
stream locations identified during exploration. Use the planned dimension sweep,
held-out evaluation pairs, and exactly three recorded random seeds.

### 5. Attention head DBM

Train grouped attention head masks with the intermediate variable as supervision.
Cover every head in each promising layer band and use exactly three recorded
random seeds. The grouped head gate described in the exploratory method guide is
required; a mask over individual feature coordinates is not a head mask.

### 6. MLP neuron DBM

Train MLP neuron masks with the intermediate variable as supervision. Cover the
candidate neurons in every promising layer band and use exactly three recorded
random seeds.

The learned methods use separate training runs because their supervision changes
with the target intermediate variable. Do not reuse a DAS or DBM fit trained to
localize an input or output variable during exploration.

## Controls and interpretation

Use the dataset that hypothesis generation showed can distinguish each comparison
and report its CPU baseline. Run complete representation patching as the positive
control. If it does not reach its measured maximum, stop and repair the test. Use
the null and a site that should not contain the target as negative controls.

DAS and DBM must report training and held-out evaluation results for all three
seeds. Select dimensions and masks using a rule fixed before inspecting the final
evaluation split. Compare neural results with the CPU distinguishability baseline
and the measured positive-control maximum, not with an assumed score of 1.0.

Evidence for the intermediate variable requires a result that:

- reproduces on held-out pairs and across the three seeds for learned methods;
- beats matched random subspaces or masks;
- remains distinct from every plausible alternative tested by that dataset; and
- occupies locations consistent with left-to-right information flow.

## Running the methods

The protocol presets live in `causalab/configs/protocols/`. Start from
`interchange.json`, `weekdays_locate_scan.json`, `das.json`,
`weekdays_das_sweep.json`, `weekdays_das_apply.json`, `dbm.json`,
`path_patching.json`, `hydra_effect.json`, or `harvest.json` as appropriate. The
worked locate-to-fit-to-apply workflow is
`causalab/configs/workflows/weekdays_8b.json`.

Use `causalab explain <doc>` and `causalab validate <doc> --data` before launching
a sweep. Use the implementation report guidance at
[`../../implementation/references/interchange-das-localization-report-format.md`](../../implementation/references/interchange-das-localization-report-format.md)
for individual DAS runs.

## Result contract

After all six experiments finish, write one self-contained `report.html` for the
intermediate variable using
[`intermediate-variable-report.md`](intermediate-variable-report.md). The report
must integrate the experiments into one layer-by-layer causal account rather than
present six unrelated leaderboards.

The experiment is incomplete until the six method artifacts and the combined HTML
report exist. If the attention head DBM execution stub prevents that experiment,
mark the report incomplete and name the missing evidence explicitly.

After each result, update `INTERMEDIATE_VARIABLE_IDEAS.md`, the layer-by-layer
causal account in `ROADMAP.md`, and `REPORT_PLAN.md`. Run as many generation and
testing iterations as the evidence requires. Select main claims only when their
tests are sharp enough to distinguish them from plausible alternatives.

## Routing out

**Strong positive result** → mark it as a candidate main claim. After the useful
generation and testing iterations are complete, select the subset that warrants
[`generalization`](../generalize-results/generalize-results.md).

**Anything else** → return to
[hypothesis generation](../hypothesis-generation/hypothesis-generation.md) when
the evidence already suggests another intermediate variable or a sharper dataset.
Return to
[exploratory experimentation](../exploratory-experimentation/exploratory-experimentation.md)
when a new internal explanation requires more evidence first.

Before routing backward, confirm that the positive control worked and that the
dataset had enough power to distinguish the target. Record the result and routing
decision in `ROADMAP.md`. Refutations remain part of the final causal account.
