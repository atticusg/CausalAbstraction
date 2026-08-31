# Report contract for one intermediate variable

Write one self-contained HTML report at
`$WORKDIR/hypothesis-testing/{intermediate-variable}/report.html`. Embed all data,
styles, and plotting code so the report opens without a network connection.

Use direct technical language. The report should answer one question: what does
the combined intervention evidence establish about this intermediate variable
from the model's input through its output?

## Required section order

### 1. Variable and test

Define the target intermediate variable, its possible values, its causal parents
and children, and the proposed neural locations. Show one concrete task example
with the target value annotated.

List the plausibly confusable input, output, and intermediate variables tested.
Link each comparison to its broad or narrow counterfactual dataset and show its
CPU distinguishability baseline.

### 2. Layer-by-layer causal account

Place this before the individual method results. Show one shared model diagram
with every attention and MLP layer. For each layer, state only what the evidence
supports about where the target appears, moves, changes, or affects the output.
Mark a layer as unresolved when the experiments do not support a claim.

The reader must be able to select a layer and see:

- relevant input and counterfactual examples;
- residual stream patching and DAS results;
- attention output patching and attention head DBM results;
- MLP output patching and MLP neuron DBM results;
- the target and alternative scores;
- the supporting artifact paths.

### 3. Complete-output patching

Provide separate residual stream, attention output, and MLP output views. Each
view must include layer and token heatmaps, selectors for broad and narrow
datasets, target and alternative scores, positive and negative controls, and a
drilldown to every underlying example.

### 4. Learned localization

Provide separate DAS, attention head DBM, and MLP neuron DBM views. Show held-out
results for all three seeds, matched random controls, the complete-output positive
control, selected dimensions or mask sizes, and stability across seeds. Identify
the exact artifact for every learned fit.

### 5. Alternatives and conclusion

Use a compact table with one row for every plausible alternative. State whether
the evidence distinguishes the target from it, which dataset and methods support
that judgment, and what ambiguity remains.

End with:

- the strongest supported claim about the target variable;
- the layers and token locations covered by that claim;
- unresolved parts of the layer-by-layer causal account;
- failed or missing experiments;
- the next generalization experiment or the reason to return to an earlier phase.

## Presentation rules

- Open on held-out results aggregated across the three seeds.
- Never headline the best seed or a training score.
- Keep input, intermediate, and output variables visually distinct.
- Show exact examples behind every aggregate result.
- Mark missing evidence as unresolved rather than filling the gap with an
  interpretation.
- Do not claim that a component computes the variable when the evidence shows
  only that the variable can be decoded or transplanted there.
