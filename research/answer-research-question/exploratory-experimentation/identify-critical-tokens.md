# Identify critical tokens and spans

Complete this step before launching any exploratory experiment. The output is a
fixed set of token locations that every method will inspect.

Use the tokenizer from the behavioral evaluation. Never identify positions from
rendered text alone. Record the exact token string, token id, zero-based index,
and character offsets for representative inputs.

## What to include

Include both kinds of locations below.

### Tokens that affect the outcome

Start with tokens whose content can change the final answer. Use the behavioral
task and its single-token counterfactual pairs to identify them. A pair qualifies
when the original and counterfactual inputs differ at exactly one token and the
model's output changes.

Record the input variable realized by the token and the observed output change.
If the variable spans several tokens, record the full span and every position in
it.

### Tokens where information may accumulate

Add locations that may collect information even when changing their surface text
is not a valid task counterfactual. Inspect at least:

- periods and other sentence-ending punctuation;
- a period together with any following space when the tokenizer joins or splits
  them in a relevant way;
- newline tokens;
- separators between task sections;
- the final prompt token before the model predicts the answer.

Add other aggregation points implied by the prompt structure. State why each one
may accumulate information.

## Required artifact

Write `$WORKDIR/exploration/critical-locations.json`. For each named location,
store its position rule, whether it is one token or a span, its category, and its
rationale. Use position rules such as `variable`, `column`, `span`, `scope`, and
`relative_to` instead of hard-coded token indices when the position varies by
input.

Verify every rule on representative inputs before launching later jobs. A rule
that resolves to the wrong token on any input is not complete.

## Report contract

Write `result/exploration/critical-tokens.html` as a lightweight, self-contained
token viewer. It must provide:

- an input selector;
- the exact rendered prompt and model output;
- a row of clickable tokens showing token text, id, and index;
- visible highlights for every selected token and span;
- the category and rationale for the selected location;
- the output change for outcome-sensitive tokens;
- four or five representative inputs for each location rule.

The report is an audit of position selection, not a general tokenizer explorer.
Do not add unrelated token statistics or model-internal results.

## Completion gate

This step is complete when the JSON artifact and HTML report exist, every
location rule resolves correctly, and the set includes outcome-sensitive tokens,
plausible accumulation points, and the final token before prediction.
