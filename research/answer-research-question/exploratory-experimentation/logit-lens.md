# Logit lens experiment

Run logit lens over the residual stream at every model layer and every critical
token position. For a critical span, decode every position separately.

Use the selected prompt, model, inputs, and answer scoring from behavioral
analysis. Include about twelve representative inputs by default, with correct and
incorrect behavioral examples when both exist.

Read `INTERMEDIATE_VARIABLE_IDEAS.md` and include the proposed symbols or output
tokens associated with the current child output target. Label each one with the
idea's current status. Selecting a symbol for inspection does not make it an
established intermediate variable.

## Method

Read `block_output` at the selected source layer and position, insert that vector
at the final `block_output`, and read `lm_head` at the same position. Save at
least the top ten decoded tokens and their logits for every layer, position, and
input.

The direct-effect protocols in
`causalab/configs/protocols/hydra_effect.json` demonstrate this read, insert, and
decode pattern. Sweep all source layers and critical positions in one protocol
campaign, then shard its expanded points as described in
[`exploratory-experimentation.md`](exploratory-experimentation.md#shard-one-experiment-correctly).

## Interpretation

Report when the final answer and relevant intermediate symbols become decodable.
Also report when a token appears and later disappears.

Early residual streams were not trained to be decoded by the final layer norm and
output head. Early noise is a projection artifact, not evidence that the model is
computing nonsense. Decodability shows that information is linearly available; it
does not show that a layer computed it or that later components use it.

## Report contract

Write `result/exploration/logit-lens.html` as a comprehensive, self-contained
explorer. It must provide:

- selectors for input, critical location, and decoded token rank;
- a layer-by-token heatmap for the selected decoded token or answer;
- the top decoded tokens and logits when the reader selects a cell;
- the exact input, expected answer, and model output beside the heatmap;
- overlays marking critical positions and spans;
- a comparison between original and single-token counterfactual inputs;
- a compact list of depths where candidate symbols appear or disappear;
- a warning that logit lens is correlational and uses the final decoder out of
  distribution at early layers.

The report must open on the answer token for a representative input. Embed all
data and plotting code; do not require a network connection.

## Handoff

Add candidate variables suggested by the decoded trajectories to the roadmap
ledger. Record the exact layers and token positions that motivate each candidate.
