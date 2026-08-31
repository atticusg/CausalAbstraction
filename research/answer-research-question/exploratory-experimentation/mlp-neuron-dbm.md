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

## Data and objective

Use the same single-token counterfactual dataset and intervention direction as the
parent patching experiment. Train the mask to preserve the counterfactual output
effect while penalizing the number of selected neurons. Evaluate on held-out
pairs.

Run a separate jointly trained mask for every promising combination of layer band
and token location. For spans, test every position separately and the whole span
jointly. Sweep regularization strength and several seeds.

Shard independent band, position, regularization, and seed points within the MLP
DBM experiment as described in
[`exploratory-experimentation.md`](exploratory-experimentation.md#shard-one-experiment-correctly).

## Controls

Include:

- the parent full-output patch as a positive control;
- random neuron sets matched to each learned mask's size and layer distribution;
- held-out intervention performance;
- several seeds;
- a mask-size versus effect curve.

The mask is not meaningful if the full-output positive control fails or if a
matched random mask performs equally well.

## Report contract

Write `result/exploration/mlp-neuron-dbm.html` as a comprehensive,
self-contained explorer. It must provide:

- selectors for parent band, layer, token position or span, regularization, seed,
  and data split;
- the selected neurons and soft mask values within each layer;
- held-out effect and mask size for every fit;
- the full-output positive control and matched random-neuron controls;
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
