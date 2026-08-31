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

Run one experiment for each combination of:

- input variable;
- residual stream layer;
- individual token position or shared span;
- subspace dimension;
- random seed.

Sweep dimensions `[2, 4, 8, 16, 32]` and at least three random seeds. Use
held-out counterfactual pairs for evaluation. Never report training accuracy as
the localization result.

Start from `causalab/configs/protocols/weekdays_das_sweep.json`. Replace its rank
sweep with `[2, 4, 8, 16, 32]`, use the location artifact from residual stream
patching, and use the single-token counterfactual dataset. Keep the dimension and
seed axes in the protocol campaign and shard expanded points with `--points`.

## Controls

For every location and dimension, include:

- a matched-dimension random subspace control;
- an identity or full-vector positive control;
- held-out performance on pairs not used to fit the subspace;
- several seeds to reveal unstable fits.

A DAS result is not usable when the positive control fails. A learned subspace
must beat its matched random control and reproduce across held-out pairs and
seeds.

## Report contract

Write `result/exploration/residual-stream-das.html` as a comprehensive,
self-contained explorer. It must provide:

- selectors for input variable, token position or span, layer, dimension, seed,
  and split;
- held-out intervention performance by layer and dimension;
- the matched random-subspace control beside every DAS result;
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
