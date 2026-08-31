# Residual stream DAS

Run Distributed Alignment Search (DAS) at every residual stream location with
reproducible signal in full-vector patching. DAS tries to localize the observable
input variable within a smaller residual stream subspace.

This is exploratory localization. The target is the changed input variable from
the single-token counterfactual dataset, not an intermediate variable from a
causal model. Do not interpret a successful fit as evidence that an explicit
high-level causal variable has already been established.

## Targets

Create a DAS target for every layer and individual token position that passed the
full-vector signal gate. When a joint span passed the gate, also train one shared
subspace across every position in that span. The same learned basis is applied to
each token vector; do not concatenate token vectors.

Within the residual stream DAS experiment, launch a separate training job for
each combination of:

- supervised input variable or output variable;
- residual stream layer;
- individual token position or shared span;
- subspace dimension;
- random seed.

Sweep dimensions `[2, 4, 8, 16, 32]` and exactly three recorded random seeds. Use
held-out counterfactual pairs for evaluation. Never report training accuracy as
the localization result.

Expand these combinations into independent CausaLab points and run them in
parallel within this one method experiment. Do not serialize jobs for different
variables, locations, dimensions, or seeds when sufficient compute is available.

Start from `causalab/configs/protocols/weekdays_das_sweep.json`. Replace its rank
sweep with `[2, 4, 8, 16, 32]`, use the location artifact from residual stream
patching, and use the single-token counterfactual dataset. Keep the dimension and
seed axes in the protocol campaign and shard expanded points with `--points`.

**Which file holds the held-out number.** `weekdays_das_sweep.json` is the *fit*
document, and the three files it can write are three different questions:

| file | what it is |
|---|---|
| `iia.json` | the **training** score — the fit re-scored on the split it trained on. Never the localization result. |
| `train_eval.json` | the score on the split `train.eval` declares, written per evaluation round. This is the honest held-out number *for that split*. |
| `rot.safetensors` | the fitted rotation, stamped with its ArtifactIdentity. |

A cross-dataset or cross-condition evaluation is a separate **apply** document:
`causalab/configs/protocols/weekdays_das_apply.json`, which loads
`rot.safetensors` by `file_path`, carries no `train` block, and scores whatever
data you point it at. The identity check refuses a rotation fitted at another
model, dtype, site, `k`, or parametrization, so an apply cannot silently
evaluate the wrong fit. `causalab/configs/workflows/weekdays_8b.json` chains
locate → fit → select → apply and is the shape to copy.

The gap this closes is measured: on the addition investigation a fit's own
`iia.json` read **0.542** where the held-out value was **0.221**. Reporting the
first is reporting a subspace that is not there.

## Controls

For every location and dimension, include:

- a matched-dimension random subspace control;
- an identity or full-vector positive control;
- held-out performance on pairs not used to fit the subspace;
- exactly three recorded seeds to reveal unstable fits.

A DAS result is not usable when the positive control fails **at the readout
cell** — a mid-tower full swap scoring below a low-rank edit is an expected
finding, not a broken control (see
[`../hypothesis-testing/hypothesis-testing.md`](../hypothesis-testing/hypothesis-testing.md),
"Controls and interpretation").

A learned subspace must beat its matched random control and reproduce across
held-out pairs and seeds. **That is necessary and not sufficient.** Report the
**null** and the **measured ceiling** beside every fit, and read the three
together:

- the *null* is the score with nothing intervened on — what the dataset gives
  you for free;
- the *matched random subspace* is what a rank-k edit gives you for free;
- the *measured ceiling* is the best any intervention reached at that cell, and
  it is what a score should be read as a fraction of — never an assumed 1.0.

The two criteria disagree, and on a real result they disagreed about the
headline. A carry variable scored **0.221** against a matched random control of
**0.000** — clears the random control, so "passes" — while its own null was
**0.578** and the measured ceiling **1.000**. It is below the null. A fit that
does not beat the null has localized nothing, however far it is above random.

## Report contract

Write `result/exploration/residual-stream-das.html` as a comprehensive,
self-contained explorer. It must provide:

- selectors for supervised input or output variable, token position or span,
  layer, dimension, seed, and split;
- held-out intervention performance by layer and dimension, and the file it came
  from (`train_eval.json` or an apply document — never a fit's `iia.json`);
- the matched random-subspace control beside every DAS result;
- the null and the measured ceiling beside every DAS result;
- the full-vector positive-control score from the parent patching experiment;
- seed variability and train-versus-evaluation curves;
- the exact base and counterfactual examples for selected points;
- the learned artifact identity and path for every fit;
- a clear marker for locations that pass all controls.

Open on held-out results aggregated across seeds. Do not headline the best seed or
training score.

## Handoff

Add stable localized input-variable subspaces to the roadmap ledger as evidence
for candidate intermediate variables. Preserve the distinction between the
observable input variable used for supervision and the intermediate variable that
may later be proposed in hypothesis generation.
