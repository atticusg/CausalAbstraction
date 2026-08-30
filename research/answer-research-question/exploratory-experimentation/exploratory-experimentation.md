# Step 2 — Exploratory experimentation

Use inexpensive experiments to inform a hypothesis, not to prove one. This phase
starts only after behavioral analysis is complete. Reuse its selected prompt,
model configuration, dataset and scoring code, code that loads the model, and
working job setup. Extend that code instead of building a separate implementation
of the task.

The three required experiments are logit lens, PCA, and counterfactual patching.
They are separate experiments and may launch in parallel. Hypothesis generation
remains blocked until all three finish and their observations have been assembled.

Everything here is suggestive. A logit lens heatmap will always contain some
structure, a PCA projection only shows what is linearly available, and a small
patching study provides locations to investigate rather than a reliable effect
size.

Before launching each experiment, state what it would show if the initial guess
were right. Exclude causally impossible locations: with causal attention,
information moves from left to right, so an earlier token cannot contain
information introduced by a later token.

## Logit lens experiment

Decode intermediate residual stream activations through the model's output head
to see what each layer and token position would predict if the network stopped
there. Run the experiment over all tokens and all layers for about twelve inputs.
Save the top predictions at each cell rather than only the highest-scoring token.

- Report the depth at which the answer first appears and what is predicted at the
  positions relevant to the task before then.
- Early layers frequently decode to noise. The lens applies the final layer norm
  and output head to an activation that was not trained to be read there. Treat
  this as a projection artifact rather than a finding.
- Decodability at a layer bounds where to look. It does not show that the layer
  computes the answer or that a later component uses the decoded information.

## PCA experiment

Collect the residual stream at token positions relevant to the task across a few
thousand inputs. Fit centered PCA separately at each layer. Color the projections
by every label supported by the behavioral evaluation, including each relevant
token's value, the correct answer, the model's answer, and whether the answer was
correct.

- Provide every applicable color scheme. Structure may be visible under one label
  and invisible under another.
- Report what the leading components organize and the depth at which that
  organization appears.
- Visible structure does not show that the model uses it. PCA shows that
  information is present and linearly available.

## Counterfactual patching experiment

Construct pairs in which the counterfactual differs from the original input at
exactly one token. Verify that this single change also changes the model's output.
Discard pairs that differ at more than one token or do not change the output.

Patch in one direction only: take activations from the counterfactual run and
insert them into the original run. Sweep every model layer at the changed token's
position and at every token position relevant to the task after it. Do not patch
earlier positions, because causal attention prevents them from receiving
information from the changed token.

The patching branch contains three separate experiments. They use the same pairs
and may also run concurrently:

1. **Residual stream.** Patch the complete residual stream vector at one layer and
   position at a time. This traces where the changed information is carried from
   its source through later positions.
2. **Attention outputs.** Patch the complete attention output across contiguous
   bands of layers at the same positions. Include bands of width one so every
   layer is tested, then test wider bands. This shows whether a span of attention
   layers is sufficient to transfer the changed behavior.
3. **MLP outputs.** Patch the complete MLP output across contiguous bands of layers
   at the same positions. Include bands of width one so every layer is tested,
   then test wider bands. This shows whether a span of MLP layers is sufficient to
   transfer the changed behavior.

For every intervention, report whether the original output moves to the
counterfactual output and report the change in the relevant output logits. The
goal is to trace mediation from the changed token through later positions relevant
to the task, not merely to find a cell with a nonzero effect.

## Units of work

- One logit lens experiment over all layers and token positions.
- One PCA experiment over all layers and positions relevant to the task.
- One experiment that patches the residual stream.
- One experiment that patches bands of attention outputs.
- One experiment that patches bands of MLP outputs.
- One synthesis that compares the observations from all three experiments.

## Handoff

The phase is complete only when all three default experiments have finished and
their observations have been assembled with the strength and limitations of each
result. Update `ROADMAP.md`, then go to
[`../hypothesis-generation/hypothesis-generation.md`](../hypothesis-generation/hypothesis-generation.md).

If you arrived here from hypothesis testing, use the failed test to define a new
set of exploratory experiments. Do not rerun completed sweeps. Hypothesis
generation remains blocked until the revised exploratory phase is complete.

## Running these experiments

Extend the behavioral analysis code and protocol documents for the sites, metrics,
and saved outputs. Keep the same selected prompt and model configuration. Inspect
each protocol document before running it:

```bash
uv run causalab explain experiment.json --data-root "$DATA_ROOT"
uv run causalab run experiment.json \
  --data-root "$DATA_ROOT" --out "$WORKDIR/exploration/run-name" \
  --device cuda --dtype bf16
```

Use these shipped documents as concrete starting points:

- **Logit lens:** read the residual stream at a swept source layer, insert that
  read into the final `block_output`, then read `lm_head` at the same position and
  save a `top_k` metric. This applies the model's final normalization and output
  head to each source activation. The direct-effect protocols in
  `causalab/configs/protocols/hydra_effect.json` use the same read, insert, and
  decode pattern.
- **PCA:** use `causalab/configs/protocols/harvest.json` to save residual stream
  activations. Add a workflow script step that runs
  `{"module": "causalab.analysis.fit_pca"}` with the harvested tensor as `acts`
  and the requested component count as `k`. It writes `basis.safetensors` and a
  JSON table containing each component's explained variance. Join labels from the
  behavioral evaluation when plotting the projections.
- **Counterfactual patching:** start from
  `causalab/configs/protocols/weekdays_locate_scan.json`, which inserts a
  counterfactual activation across a layer and position grid. A sweep is written
  directly on a named field. For example,
  `"layer": {"sweep": {"range": [0, 32]}}` covers layers 0 through 31. Build
  separate protocol documents for the residual stream, contiguous bands of
  attention layers, and contiguous bands of MLP layers. Each document must use
  pairs that differ at exactly one token and must insert the counterfactual
  activation into the original run.

Protocol sweeps replace the retired fan-out runner. Use `--points START:STOP` to
run part of a large sweep without changing its campaign or point identifiers.
