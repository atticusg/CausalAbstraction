# MLP neuron DBM

Use Desiderata-Based Masking (DBM) to select individual MLP neuron outputs within
every layer band and token location that showed signal in full MLP output
patching.

## Masking unit

Target `mlp_activation` when that component is the model family's vector of MLP
neuron activations before the output projection. Verify the component against the
model's hookpoint table before running: model families do not all expose their MLP
interior in the same way.

Use one DBM gate value per neuron and train all neuron masks in the promising
layer band jointly. The current CausaLab `gate` featurizer already learns one
value per feature coordinate, so it can select individual neurons when attached
to a valid `mlp_activation` site. Use one gate per layer and include every gate in
the joint training objective.


## Non-Llama towers

The masking unit above assumes a dense MLP with an `act_fn` — a Llama-shaped
block. Check the model's hookpoint table before designing this experiment,
because on a mixture-of-experts tower the component this document names **does
not exist**.

- **`mlp_activation` has no tensor on an MoE block.** There is no `act_fn` to
  tap, and both engines refuse the component by name. Measured on
  Qwen3.6-35B-A3B, whose 40 layers are all MoE (256 experts, top-8, plus a
  shared expert).
- **The substitutes are `expert_activation` and `shared_expert_activation`.**
  The shared expert fires for every token, so a mask over
  `shared_expert_activation` is a neuron mask in the ordinary sense and the rest
  of this document applies unchanged.
- **A routed-expert mask is not a neuron mask.** A routed coordinate is a
  *(slot, feature)* pair, and which expert occupies a slot is decided per token.
  Two tokens' coordinate 17 are the same neuron only if both were routed to the
  same expert. So a mask learned over routed coordinates is comparable **only
  across tokens that share a routing**, and "this neuron carries the variable"
  is not a claim it can support. Either condition the analysis on a fixed
  routing, or state the mask as being over slots and say so in the report.

Whatever you choose, record the component you actually tapped and the model's
hookpoint identity next to every fit. A silently substituted component is the
kind of thing a reader cannot recover from the numbers.

## Data and objective

Use the same single-token counterfactual dataset and intervention direction as the
parent patching experiment. Train the mask to preserve the counterfactual output
effect while penalizing the number of selected neurons. Evaluate on held-out
pairs.

Run a separate jointly trained mask for every supervised input or output variable
and every promising combination of layer band and token location. For spans,
test every position separately and the whole span jointly. Sweep regularization
strength and run exactly three recorded random seeds.

Expand the variable, band, position, regularization, and seed axes into separate
jobs. Launch them in parallel within the MLP DBM experiment and shard them as
described in
[`exploratory-experimentation.md`](exploratory-experimentation.md#shard-one-experiment-correctly).

## Controls

Include:

- the parent full-output patch as a positive control;
- random neuron sets matched to each learned mask's size and layer distribution;
- held-out intervention performance;
- three seeds;
- a mask-size versus effect curve.

The mask is not meaningful if the full-output positive control fails **at the
readout cell** — below it, a full swap scoring under a sparse mask is an
expected finding, not a broken control — or if a matched random mask performs
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

Write `result/exploration/mlp-neuron-dbm.html` as a comprehensive,
self-contained explorer. It must provide:

- selectors for supervised input or output variable, parent band, layer, token
  position or span, regularization, seed, and data split;
- the selected neurons and soft mask values within each layer;
- held-out effect and mask size for every fit, and the file each came from
  (`train_eval.json` or an apply document — never a fit's `iia.json`);
- `decisive_fraction` from `fit_diagnostics.json` beside every effect;
- the full-output positive control, the null, the measured ceiling, and matched
  random-neuron controls;
- mask-size versus effect curves;
- exact examples and intervened outputs for selected fits;
- overlap and stability of selected neurons across seeds;
- the hookpoint identity and neuron count for every layer.

Use a layer selector and sortable neuron table instead of trying to display every
neuron from every layer at once. Open on held-out results aggregated across seeds.

## Handoff

Add stable neuron groups and their token locations to the table of candidate
variables. State whether their effects suggest preservation, transformation, or
output construction.
