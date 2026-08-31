# Full MLP output patching

Test whether complete MLP outputs across a contiguous span of layers are
sufficient to transfer the changed input variable. This experiment identifies
promising layer bands before DBM searches for individual MLP neurons.

## Data and direction

Use the same single-token counterfactual pairs as the other patching experiments.
Read every source activation from the counterfactual run and insert it into the
original run. Run only this direction.

## Layer bands and token locations

Test every contiguous band of five layers and every contiguous band of ten
layers. A band intervention patches the complete `mlp_output` at every layer in
that span during the same forward pass.

For every band, patch:

- the changed token;
- every critical position after the changed token;
- each position in a critical span separately;
- each critical span jointly.

Save whether the intervention moves the original answer to the counterfactual
answer, the answer logit difference, and the underlying outputs.

Save each intervention once. Use that result to calculate separate scores for
every applicable input variable and the output variable. Do not repeat the model
intervention to change only the variable used for evaluation.

The current protocol sweep language cannot express all dependent writes in a
layer band through one start-layer sweep. Author one explicit protocol document
for each band, keep all documents inside this experiment, and shard each document
over its position and data axes with `--points`.

## Signal and DBM gate

Predeclare the reproducibility and effect threshold that identifies a promising
band. Start MLP neuron DBM for every promising band and token location as soon as
its full-output result is available. A band with no reproducible effect does not
receive a DBM follow-up.

## Report contract

Write `result/exploration/mlp-output-patching.html` as a comprehensive,
self-contained explorer. It must provide:

- a band-start-by-token heatmap with a width selector for five or ten layers;
- selectors for causal variable, pair, metric, individual position, and joint
  span;
- the exact original and counterfactual inputs and outputs;
- the patched output and relevant logits for a selected band;
- aggregate effect, sample count, and variability for every band, location, and
  causal variable;
- visible markers for every band sent to MLP neuron DBM;
- a view showing where MLP-mediated signal appears and disappears across depth
  and token position.

The reader must be able to inspect every example underlying an aggregate cell.

## Handoff

Write the promising layer bands and token locations to a machine-readable artifact
for MLP neuron DBM. Add the implied candidate variables to `ROADMAP.md`.
