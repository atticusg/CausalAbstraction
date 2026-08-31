# Full residual stream patching

Trace where the changed input variable is carried by patching the complete
residual stream vector. This experiment localizes signal before DAS searches for
a smaller subspace.

## Data and direction

Use the single-token counterfactual dataset. Each pair must differ at exactly one
token, and the model's output must change. Read the source activation from the
counterfactual run and insert it into the original run. Do not run the reverse
direction in this exploratory experiment.

## Sweep

At every model layer, patch `block_output` at:

- the changed token;
- every other critical position after the changed token;
- every position within a critical span separately;
- every critical span jointly.

Do not patch positions before the changed token. Save whether the intervention
moves the original answer to the counterfactual answer, the counterfactual versus
original answer logit difference, and the full model output needed to inspect
failures.

Save each intervention once. From that saved result, calculate a separate score
for the changed input variable, every other applicable input variable, and the
output variable. Do not rerun the forward pass merely to change which causal
variable is being evaluated.

Use `causalab/configs/protocols/weekdays_locate_scan.json` as the starting pattern
for the layer and position sweep. Keep all layer and position points in one
campaign and shard it with `--points` when needed.

Its second position is `{"index": -1, "scope": {"variable": "entity"}}` — the
**last token of** the entity span, not the span. Write it that way for any
variable whose value tokenizes to different lengths across rows: a bare
`{"variable": …}` window is ragged, and a ragged *write* is refused at run time
([V19]). `validate --data` cannot catch that for you — it has no tokenizer, so
it checks that the variable exists and nothing about the width of what it
resolves to. When you do want the whole span, patch it and expect the refusal on
any table whose values are not uniform; when you want one token per row, scope
an index.

## Signal and DAS gate

Before launching, define the minimum effect that counts as signal and how many
pairs must reproduce it. Use both output changes and logit differences. A single
large outlier is not a location.

Every individual position and joint span with reproducible signal becomes a DAS
target. Start its DAS jobs immediately; they do not wait for PCA, logit lens, or
the attention and MLP branches.

## Report contract

Write `result/exploration/residual-stream-patching.html` as a comprehensive,
self-contained explorer. It must provide:

- a layer-by-token heatmap for the intervention effect;
- selectors for causal variable, pair, metric, individual position, and joint
  span;
- the exact original input, counterfactual input, changed token, and both normal
  outputs;
- the patched output and relevant logits for a selected cell;
- aggregate effect, sample count, and variability for every cell and causal
  variable;
- visible markers for every location that passed the DAS signal gate;
- a list of locations where the signal first appears, moves to another token, or
  disappears.

The default view must show the aggregate counterfactual-versus-original logit
difference. The reader must be able to move from an aggregate cell to every
underlying example.

## Handoff

Write the signal locations to a machine-readable artifact for the DAS experiment
and add the implied candidate variables to the roadmap ledger.
