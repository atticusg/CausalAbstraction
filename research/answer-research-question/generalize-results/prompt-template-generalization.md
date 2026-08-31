# Generalize across prompt templates

Test whether the same intermediate variable and layer-by-layer causal account
survive when the task remains the same but its prompt changes. A translation into
another language is a prompt-template change when the task, labels, and grading
rule remain unchanged.

## Design

Choose several prompt templates that materially change wording, formatting,
tokenization, or answer position. Include another language when it is scientifically
useful. Run behavioral analysis for every template and retain only templates where
the model exhibits the intended behavior well above random chance.

For each retained template:

1. Identify its critical tokens and spans again.
2. Map the original input, intermediate, and output variables to their semantic
   roles in the new template.
3. Build broad and narrow counterfactual datasets and certify their
   distinguishability.
4. Repeat all six intervention experiments.
5. Compare the new results with the original account by semantic role and by
   absolute token position.

This experiment should deliberately move important variables to new token
positions. It must distinguish a mechanism that follows a semantic role from one
that is tied to a fixed index.

## HTML report contract

Write `result/generalization/prompt-templates.html` as a self-contained
interactive report. It must provide:

- one tab per exact prompt template, with representative inputs and outputs;
- behavioral performance for every template;
- aligned token views showing how each semantic role moves;
- the six intervention results for every template;
- held-out DAS and DBM results for all three seeds;
- side-by-side layer accounts for the original and new templates;
- a language selector when translations are included;
- exact examples behind aggregate cells;
- a final table stating which claims transfer, change location, or fail.

The report must separate a failure of the behavior, a failure of the
counterfactual test, and a failure of the mechanism to generalize.
