# Attention head DBM

Use Desiderata-Based Masking (DBM) to select attention heads within every layer
band and token location that showed signal in full attention output patching.

The candidate mask covers every attention head in every layer of the promising
band. Train the heads in the band jointly so the result can select a distributed
set rather than ranking heads one at a time.

## Masking unit

One mask value controls one complete attention head contribution. All feature
coordinates belonging to that head must share the same binary mask value. Use
`attention_premix` to intervene on a head's contribution before the output
projection. `attention_result` is derived and read-only in the current protocol
runner, so it is not a valid write target.

The current CausaLab `gate` featurizer learns one mask value per feature
coordinate. It does not tie all coordinates of a head to one value. Do not report
a normal feature gate as attention head DBM. Before running this experiment, add
or use a grouped gate whose units are attention heads and whose mask spans every
head in the selected layer band. Validate that zeroing one learned unit removes
the complete contribution of exactly one head.

> **Execution: stub.** CausaLab does not yet provide this grouped head gate. The
> standard `gate` featurizer is not an acceptable substitute because it selects
> coordinates within heads rather than whole heads.

## Data and objective

Use the same single-token counterfactual dataset and direction as the parent
patching experiment. Train the grouped mask to preserve the counterfactual output
effect while penalizing the number of selected heads. Evaluate on held-out pairs.

Run a separate jointly trained mask for every supervised input or output
variable and every promising combination of layer band and token location. For
spans, test every position separately and the whole span jointly. Sweep
regularization strength when needed to produce a sparsity curve rather than
selecting one arbitrary penalty after seeing the result. Run exactly three
recorded random seeds.

Expand the variable, band, position, regularization, and seed axes into separate
jobs. Launch them in parallel within the attention DBM experiment and shard them
as described in
[`exploratory-experimentation.md`](exploratory-experimentation.md#shard-one-experiment-correctly).

## Controls

Include:

- the parent full-output patch as a positive control;
- random head sets matched to each learned mask's size;
- held-out intervention performance;
- three seeds;
- a mask-size versus effect curve.

The mask is not meaningful if the full-output positive control fails or if
matched random head sets perform equally well.

## Report contract

Write `result/exploration/attention-head-dbm.html` as a comprehensive,
self-contained explorer. It must provide:

- a layer-by-head map of learned mask values and hard selections;
- selectors for supervised input or output variable, parent band, token position
  or span, regularization, seed, and data split;
- held-out effect and mask size for every fit;
- the full-output positive control and matched random-head controls;
- mask-size versus effect curves;
- exact examples and intervened outputs for selected fits;
- stability of selected heads across seeds;
- the grouped-gate implementation identity used for the run.

Open on held-out results aggregated across seeds. Never present individual feature
coordinates as attention heads.

## Handoff

Add stable head groups and their token locations to the table of candidate variables.
State whether the heads appear to move, preserve, or transform the localized input
variable.
