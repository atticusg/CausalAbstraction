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

A band patch is several **dependent** writes in one forward: the chosen start
layer determines every layer in the span. The sweep language cannot express that
— a sweep expands one axis into *independent* points — but that does not mean
one document per band. `intervened_models` (§2.9) names the *set* of writes in
force for one forward, so every band is an entry over **one** table of per-layer
writes:

```json
"sites":  { "a10": {"component": "mlp_output", "layer": 10}, "…": "one per layer" },
"reads":  { "v_a10": {"site": "a10", "pos": "tap", "model": "original", "input": "counterfactual"} },
"writes": { "w10":  {"site": "a10", "pos": "tap", "do": {"swap": "v_a10"}} },
"intervened_models": {
  "band5_L10":  {"input": "base", "writes": ["w10", "w11", "w12", "w13", "w14"]},
  "band5_L15":  {"input": "base", "writes": ["w15", "w16", "w17", "w18", "w19"]},
  "band10_L10": {"input": "base", "writes": ["w10", "…", "w19"]}
}
```

Overlapping bands reuse writes rather than restating them, and every band shares
**one** counterfactual-harvest forward. That is one model load for the whole
scan instead of one per band, at ~1–2 min each: a verified A3B document carried
41 sites, 40 writes and 67 intervened_models — every 5- and 10-layer band of a
40-layer tower — with 136 save entries, in a single campaign.

Copy `causalab/configs/protocols/attention_band_patch.json` — the same shape,
with `mlp_output` sites instead of `attention_output` — and extend it: one entry
per layer in `sites`/`reads`/`writes`, one entry per band in
`intervened_models`. Shard the remaining position and data axes with `--points`.

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
