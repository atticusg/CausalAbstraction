# Generalize across related tasks

Test whether a different task uses the same intermediate variable. The tasks must
share a concrete causal quantity or operation, not merely a topic. Examples
include a comparison variable reused across two decision tasks or a variable that
identifies a spatial location in two different input formats.

## Design

Define the proposed shared variable independently of either task's surface text.
For the new task:

1. Establish reliable behavior and perform error analysis.
2. Write a causal model that contains the proposed shared variable.
3. Identify plausibly confusable input, output, and intermediate variables.
4. Build broad and narrow counterfactual datasets and certify
   distinguishability.
5. Identify critical tokens and spans.
6. Repeat all six intervention experiments.

Use a train and evaluation split that can test shared structure when possible.
For example, fit DAS or DBM on one task and evaluate the learned artifact on the
other task without refitting. Report this transfer separately from a new fit on
the related task. A successful new fit shows that both tasks contain a similar
variable; cross-task transfer is stronger evidence that they reuse a common
representation.

## HTML report contract

Write `result/generalization/related-tasks.html` as a self-contained interactive
report. It must provide:

- concrete examples and behavioral performance for both tasks;
- the causal definition of the proposed shared variable in each task;
- the counterfactual datasets and CPU distinguishability baselines;
- the six intervention results for both tasks;
- separate results for a new fit and cross-task transfer;
- held-out DAS and DBM results for all three seeds;
- aligned layer-by-layer causal accounts;
- exact examples behind aggregate results;
- a final statement of what is genuinely shared and what remains task-specific.

Do not call the variable shared when only its name or possible values match. The
causal role and intervention evidence must match as well.
