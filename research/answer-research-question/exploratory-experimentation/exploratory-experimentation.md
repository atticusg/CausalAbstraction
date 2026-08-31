# Step 2 — Exploratory experimentation

Use exploratory experiments to find candidate intermediate causal variables and
the neural locations that may carry them. These experiments suggest what the
model may be doing. They do not establish that a candidate variable exists.

Use [`../../causal-handbook.md`](../../causal-handbook.md) for the distinction
between exploratory correlation, localization of an observable input variable,
and evidence for an intermediate causal variable.

This phase starts only after behavioral analysis is complete. Reuse its selected
prompt, model and revision, dataset and scoring code, code that loads the model,
and working job setup. Extend the working behavioral experiment instead of
reimplementing the task.

## Execution graph

Critical token locations are a blocking prerequisite. Identify them before
launching any exploratory experiment.

```
identify critical tokens and spans
        │
        ├── logit lens
        ├── PCA
        ├── full residual stream patching ──▶ residual stream DAS
        ├── full attention output patching ─▶ attention head DBM
        └── full MLP output patching ───────▶ MLP neuron DBM
```

Launch the five experiments on the left immediately and in parallel after the
critical locations are fixed. Each experiment is independent and has its own
protocol document, run directory, and report.

The three experiments on the right are individually gated:

- Start residual stream DAS as soon as full residual stream patching identifies
  locations with signal.
- Start attention head DBM as soon as full attention output patching identifies
  promising layer bands and token locations.
- Start MLP neuron DBM as soon as full MLP output patching identifies promising
  layer bands and token locations.

These follow-up jobs may run concurrently. A follow-up does not wait for an
unrelated branch. If a full-vector patching experiment finds no signal, record the
null result and close that branch without launching its follow-up.

Hypothesis generation remains blocked until all five initial experiments and
every applicable follow-up are complete, their reports exist, and their
observations have been added to the table of candidate variables in `ROADMAP.md`.

## Shared data and locations

Use the single-token counterfactual dataset throughout the patching and fitting
experiments. Each pair must differ at exactly one input token, and that change
must change the model's output. The changed token realizes the observable input
variable that DAS and DBM try to localize.

Run each method at every critical token location. For a critical span, inspect
each position separately. Patching experiments must also patch the whole span
together. DAS runs at every individual location with patching signal and also
learns one shared subspace across an entire span with signal. The shared
subspace is applied at each position; the token vectors are not concatenated.

For patching, use activations from the counterfactual run as the source and insert
them into the original run. Patch only the changed position and locations after
it. Causal attention prevents an earlier position from receiving information
introduced by a later token.

## Experiments and report contracts

| Order | Method | Instructions and report contract |
|---|---|---|
| prerequisite | Critical token selection | [`identify-critical-tokens.md`](identify-critical-tokens.md) |
| initial, parallel | Logit lens | [`logit-lens.md`](logit-lens.md) |
| initial, parallel | PCA | [`pca.md`](pca.md) |
| initial, parallel | Full residual stream patching | [`residual-stream-patching.md`](residual-stream-patching.md) |
| initial, parallel | Full attention output patching | [`attention-output-patching.md`](attention-output-patching.md) |
| initial, parallel | Full MLP output patching | [`mlp-output-patching.md`](mlp-output-patching.md) |
| after residual patching | Residual stream DAS | [`residual-stream-das.md`](residual-stream-das.md) |
| after attention patching | Attention head DBM | [`attention-head-dbm.md`](attention-head-dbm.md) |
| after MLP patching | MLP neuron DBM | [`mlp-neuron-dbm.md`](mlp-neuron-dbm.md) |

Every method document specifies whether its deliverable is a comprehensive
interactive explorer or a smaller self-contained HTML report. Do not replace an
interactive contract with a static image. All reports must embed their data and
assets and work without network access.

## Shard one experiment correctly

A method remains one research experiment even when its sweep is too large for one
GPU job. Put every independent layer, position, head, neuron, dimension, and seed
axis in a protocol document. Run `explain` to inspect its expanded point count:

```bash
uv run causalab explain experiment.json --data-root "$DATA_ROOT"
```

Divide that point range into non-overlapping half-open intervals. Launch every
shard with the same protocol document, data root, model precision, and code. Only
the point interval and shard output directory may differ:

```bash
uv run causalab run experiment.json \
  --data-root "$DATA_ROOT" \
  --out "$WORKDIR/exploration/method/shard-00" \
  --points 0:32 --device cuda --dtype bf16

uv run causalab run experiment.json \
  --data-root "$DATA_ROOT" \
  --out "$WORKDIR/exploration/method/shard-01" \
  --points 32:64 --device cuda --dtype bf16
```

The document digest identifies the campaign, and each expanded point has its own
digest. `--points` does not change either identity. Do not generate a different
document per shard, use `--set` differently across shards, overlap point ranges,
or omit points. Before interpreting the experiment, verify that the union of
shard manifests contains every expected point exactly once. Assemble report data
by point digest and sweep coordinates.

A simultaneous five-layer or ten-layer patch contains several dependent writes:
the chosen start layer determines every layer in the band. The current protocol
sweep language cannot express that dependency as one axis. Author one protocol
document for each explicit band, keep every document under the same method
experiment directory, and shard each document over its remaining position and
data axes. These documents are separate campaigns but one reported experiment.

`--points` applies to protocol documents, not workflow documents. If a workflow
contains a large method, shard that method's protocol experiment directly and
run dependent analysis only after its shards are complete.

## Candidate intermediate variables

Keep a running ledger in `ROADMAP.md`. Add a candidate when one or more
exploratory results suggest a stable quantity, state, or transformation inside
the model. Keep competing explanations separate.

For each candidate, record:

- a plain definition of the proposed variable;
- its possible values;
- the layers, token locations, and components that may carry it;
- the observations supporting it;
- observations that conflict with it;
- which experiment should distinguish it from competing candidates;
- its status: active, rejected, merged with another candidate, or promoted into
  an explicit causal-model hypothesis.

Exploratory DAS and DBM localize an observable input variable. They do not by
themselves promote that variable into a causal-model hypothesis. That happens in
hypothesis generation, where the candidate becomes an explicit intermediate
variable in a high-level causal model.

## Handoff

The handoff contains all method reports and the updated table of candidate
variables. State which branches found signal, which returned null results, where the
signal appeared and disappeared across layers and tokens, and which competing
internal explanations remain.

Update `ROADMAP.md`, then go to
[`../hypothesis-generation/hypothesis-generation.md`](../hypothesis-generation/hypothesis-generation.md).
