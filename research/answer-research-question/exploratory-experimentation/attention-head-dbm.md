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


## Non-Llama towers

Check the model's hookpoint table before choosing the band width, because
`attention_premix` is not present at every layer of a hybrid stack.

- **On a hybrid attention/linear-attention tower, only the full-attention layers
  have heads.** Qwen3.6-35B-A3B is 40 layers, 10 of them full attention (at 3,
  7, … 39) and 30 Gated DeltaNet. `attention_premix` exists on those 10 only, so
  a "contiguous band of five layers" contains **one or two** attention layers,
  and 30 layers contribute nothing to a head mask. Define the bands over the
  attention layers that exist rather than over depth indices, and say which
  layers a band actually covered.
- **`attention_output` does exist tower-wide**, so the parent full-output
  patching experiment is well defined on such a model even where the interior is
  not. When the interior is unavailable, the parent result is the evidence, and
  the report should say the head-level decomposition was not attempted rather
  than reporting an empty one.
- The linear-attention layers have their own interior (`delta_*` components).
  Masking it is a different experiment with a different unit, not a
  drop-in substitute for a head mask.

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

The mask is not meaningful if the full-output positive control fails **at the
readout cell** — below it, a full swap scoring under a sparse mask is an
expected finding, not a broken control — or if matched random head sets perform
equally well.

Report the **null** and the **measured ceiling** beside every fit. Beating a
matched random mask is necessary and not sufficient — a fit can clear its random
control and still sit below the score with nothing intervened on at all.

## Where the held-out number comes from

A DBM **fit** document's own `iia.json` is its *training* score, computed on the
split it trained on. The held-out number is one of two other files:

- `train_eval.json`, written per evaluation round when the fit declares
  `train.eval` with a split;
- the score of an **apply** document —
  `causalab/configs/protocols/dbm_apply.json` — which loads the fitted `theta`
  by `file_path`, carries no `train` block, and reproduces the fit's hard
  `θ > 0` mask exactly. Point it at any split or condition; its ArtifactIdentity
  check refuses a gate fitted at another model, dtype or site.

Cross-dataset and cross-condition evaluation needs the apply document; a fit
cannot answer for a split it never declared. Never report a fit's `iia.json` as
a localization result.

**Read `fit_diagnostics.json` before believing any DBM number.** A gate is a
*hard* `θ > 0` mask in eval mode, so an unseparated θ makes the mask a coin flip
on gradient noise — roughly half the coordinates swap, which at the readout
layer can score 1.000 while meaning nothing. `decisive_fraction` near 0 says the
gate never committed. Report `hard_mask_size` beside every effect: a mask is a
claim about *how few* coordinates carry the variable.

## Report contract

Write `result/exploration/attention-head-dbm.html` as a comprehensive,
self-contained explorer. It must provide:

- a layer-by-head map of learned mask values and hard selections;
- selectors for supervised input or output variable, parent band, token position
  or span, regularization, seed, and data split;
- held-out effect and mask size for every fit, and the file each came from
  (`train_eval.json` or an apply document — never a fit's `iia.json`);
- `decisive_fraction` from `fit_diagnostics.json` beside every effect;
- the full-output positive control, the null, the measured ceiling, and matched
  random-head controls;
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
