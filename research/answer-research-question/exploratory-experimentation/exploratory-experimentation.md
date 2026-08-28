# Step 2 — Exploratory experimentation

Use inexpensive experiments to inform a hypothesis, not to prove one. Run the five
methods below in roughly this order and skip those that do not fit the question.

Everything here is **suggestive**. A logit lens heatmap will always contain some
structure, and three interchange pairs provide a clue rather than a reliable rate.

Before each method, predict what it would show if your guess were right. Remove
impossible locations before a sweep: with causal attention, information flows from
left to right, so token 5 cannot read a fact introduced at token 12.

## Essential tokens

Several methods are scoped to the tokens that matter, so identify them once here
and reuse the list. An essential token is one whose replacement changes the output.

- Aim to be **exhaustive**, not obvious — anything you can substitute to flip the
  answer counts, not just the operands.
- Confirm each empirically: edit that token alone and check the output moves.
  Intuition is wrong about this often enough to be worth the compute.
- Record each token's surface text and position — per-input positions when the
  position varies across inputs.

## Probe

Decode prompts greedily to verify essential tokens and check each input set before
spending substantial compute.

## Logit lens

Decode intermediate activations through the model's own output head to see what
each (layer, position) would predict if the network stopped there. Run it over
**all tokens and all layers** for a dozen inputs, saving top-k per cell rather than
top-1 — the per-position structure and the runner-up tokens are most of the value.

- Report the depth at which the answer first appears, and what is predicted at the
  essential-token positions before then.
- Early layers frequently decode to noise: the lens applies the final layer norm
  and unembedding to an activation never meant to be read there. That is a
  projection artifact, not a finding.
- "Decodable from layer L" bounds where to look. It does not mean layer L computes
  it, and it does not mean anything downstream reads it.

## Paired examples (interchange)

Take a base input and a counterfactual that differ in the variable you care about,
swap the activation at each (layer, position) cell in turn, and record where the
output flips. At this stage, sweep only one to three pairs.

- Sample a few base inputs once and reuse them across every essential token, so
  the results are comparable.
- Verify each pair actually flips the output first. A pair where nothing moves
  teaches nothing.
- Cells where the output flips are where that token's information is being read.
  Report the sparse nonzero cells and note what gets predicted when it flips.
- Step 4 repeats this method with enough examples to support a reliable conclusion:
  [`../hypothesis-testing/hypothesis-testing.md`](../hypothesis-testing/hypothesis-testing.md).

## Knockout (ablation)

Remove a component's contribution and measure how much the behavior degrades — the
direct test of **necessity**. With no task label, grade against the model's own
unablated output.

- **Always use zero and mean replacement.** Zero may be unlike any normal
  activation. When the two disagree, use the mean result and report both.
- **Two metrics.** The fraction of outputs that *change* is interpretable and
  coarse; the drop in the predicted token's logit is graded, and routinely reveals
  sensitivity the first reports as all-or-nothing.
- **Test attention and MLPs.** Test every attention head and contiguous ranges of
  1, 3, 5, and 10 MLP layers. For MLPs, report how the effect grows with width.
- **Near-zero drop means not necessary under this ablation**, not inert.
  Redundant components each compensate for the other's removal and both read as
  unnecessary; ablate them jointly if you suspect it.
- Scope to the essential tokens by default. The head scan is the expensive part.

## PCA

Collect the residual stream at a critical token across a few thousand inputs, fit
centered PCA per layer, and colour the projection by every label the task supports
— the token's own value, the correct answer, the model's answer,
correct-vs-incorrect. What the top components organize by, and the depth at which
that organization appears, is often the most informative thing in this step.

- Offer every colour scheme the task supports; the structure is usually visible
  under one and invisible under the others.
- Structure you can see is not structure the model uses. PCA shows information is
  present and linearly available, not that anything reads it.

## Typical units of work

- One per (method × site family) — "logit lens over all layers and positions" is
  one; "ablate every attention head" is another.
- One per essential token, for the methods run per token.
- One for assembling the observations into the input for hypothesis generation.
  That is real work, not bookkeeping.

## Handoff

You leave with observations and the strength of evidence behind each one.
Update `ROADMAP.md`, then go to
[`../hypothesis-generation/hypothesis-generation.md`](../hypothesis-generation/hypothesis-generation.md).

If you arrived here *from* hypothesis testing, do not re-run what you already ran.
Choose sites, positions, or groups of inputs that you have not examined. Use the
failed test to guide that choice. A hypothesis refuted at one location may describe
the mechanism correctly but place it at the wrong location.

## Running these experiments

Write protocol documents for the model, serialized dataset, sites, metrics, and
saved outputs. Inspect each document before running it:

```bash
uv run causalab explain experiment.json --data-root "$DATA_ROOT"
uv run causalab run experiment.json \
  --data-root "$DATA_ROOT" --out "$WORKDIR/exploration/run-name" \
  --device cuda --dtype bf16
```

Use these shipped documents as concrete starting points:

- **Probe:** `tests/protocols/12_probe_variable_im.json` greedily generates eight
  tokens, saves the text and top prediction at each step, and can harvest an
  activation where the generated answer appears. Change the dataset and
  `max_new_tokens` for the task.
- **Interchange:**
  `causalab/configs/protocols/weekdays_locate_scan.json` swaps a counterfactual
  activation across a layer and position grid. A sweep is written directly on a
  named field. For example,
  `"layer": {"sweep": {"range": [0, 32]}}` covers layers 0 through 31, and
  `"tap": {"sweep": [{"index": -1}, {"variable": "subject"}]}` crosses
  those layers with two positions. Replace the data, sites, and metrics, and use
  only one to three pairs during exploration.
- **Knockout:** define a write that swaps the selected activation with the scalar
  `0.0` for zero replacement. For mean replacement, first save the site's read
  with `"reduce": "mean"`, then load that saved tensor through a `params` entry
  and swap it into the same site. Put a sweep on the site's layer, position, or
  head field. Generate separate documents for dependent ranges such as every
  contiguous 1, 3, 5, and 10 layer block. The protocol sweep forms a cross
  product; it does not express a start layer whose valid values depend on the
  range width.
- **Logit lens:** read the residual stream at a swept source layer, swap that read
  into the final `block_output`, then read `lm_head` at the same position and save
  a `top_k` metric. This applies the model's final normalization and output head
  to each source activation. The direct-effect probes in
  `causalab/configs/protocols/hydra_effect.json` use this same read, inject, and
  decode pattern.
- **PCA:** use `causalab/configs/protocols/harvest.json` to save residual-stream
  activations. Add a workflow script step that runs
  `{"module": "causalab.analysis.fit_pca"}` with the harvested tensor as `acts`
  and the requested component count as `k`. It writes `basis.safetensors` and a
  JSON table containing each component's explained variance. Join labels from
  the serialized dataset when plotting the projections.

Protocol sweeps replace the retired fan-out runner. Use `--points START:STOP` to
run part of a large sweep without changing its campaign or point identifiers.
