# PCA experiment

This guide adapts the PCA method and explorer contracts from Silico's dimension
reduction pipeline for the CausaLab exploratory phase.

Run centered PCA over residual stream activations at every model layer and every
critical token position. For a critical span, fit and display every position
separately. Do not silently pool distinct positions. A separate pooled analysis
may be added when position is retained as metadata and the report makes the
pooling explicit.

## Harvest and fit

Reuse the behavioral evaluation inputs and labels. Expand to a few thousand
representative inputs when the behavioral dataset is smaller. Preserve the
selected prompt, model revision, tokenizer, and exact token location rules.

Use `causalab/configs/protocols/harvest.json` to save `block_output` at the
selected layers and positions. Keep each layer and position in the same PCA
research experiment. Shard the harvest campaign with `--points` as described in
[`exploratory-experimentation.md`](exploratory-experimentation.md#shard-one-experiment-correctly).

Fit the mean and PCA basis on training activations only. Freeze them before
transforming validation or test activations. Run
`{"module": "causalab.analysis.fit_pca"}` as a workflow script step with the
harvested tensor as `acts` and the requested component count as `k`. It writes the
basis and explained-variance spectrum.

The current `fit_pca` script does not save the training mean or projected
coordinates. Add a deterministic companion workflow script that computes and
saves the training mean, then uses that frozen mean and the fitted basis to
project every split. Save at least the first ten components, projected
coordinates, labels, source metadata, and the original high-dimensional vectors
needed for neighbor inspection. Do not recompute the mean on validation or test
data.

Color every projection by all labels inherited from behavioral analysis,
including relevant input values, expected answer, model answer, correctness,
prompt format when more than one remains, and token location. Add other candidate
variables from the roadmap ledger as color options, but label them as exploratory.

Apply SVD or eigendecomposition to centered observations. PCA finds orthogonal directions that maximize variance; on centered data this also minimizes linear reconstruction error (the two views coincide only because the data were centered first).

**Interpretability applications**

- Plot activation trajectories (across tokens, context length, training time, interventions) and condition means for concepts, tasks, languages, or latent parameters.
- Identify high-variance directions in the residual stream or module outputs, build subspaces before probes/geometric models, and quantify participation ratio, effective rank, and anisotropy by layer.

**Checks and caveats**

- Fit on the intended population, not a visually curated subset; distinguish global, per-layer, per-condition, and centroid PCA.
- Measure distances, curvature, separability, and intervention effects in the original or adequately retained space, not solely in two PCs.
- PCA does not preserve pairwise distances in general: truncation shrinks each squared distance by the variance in the discarded directions. Quantify that distortion when distance matters, or use random projection for a direct guarantee.

**Visualize it**

- Build `result/exploration/pca.html` with
  [`pca-explorer.md`](pca-explorer.md) after the coordinates have been computed.
  This is a comprehensive, self-contained explorer. Building the page requires no
  model or GPU.

## Report contract

The explorer contract is mandatory. In addition to its general PCA controls, this
report must provide selectors for model layer, critical token position, and data
split. It must open on a layer and component view selected by a rule stated in the
report, not by manually choosing the most visually persuasive plot.

Show explained variance, participation ratio, label association scores, and
held-out projections. Clicking a point must reveal the exact prompt, selected
token, expected output, model output, and nearest neighbors in the original
residual stream space.

## Handoff

Add geometric patterns to the table of candidate variables with their layer, token
position, component numbers, label associations, and important counterexamples.
State explicitly that PCA is correlational and does not show that the model uses a
candidate variable.
