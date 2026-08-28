# Derive a task specification from a paper

Read the paper's task definition, examples, answer space, data-generation rules,
and evaluation method. Then fill [`../TASK_TEMPLATE.md`](../TASK_TEMPLATE.md).

Separate claims made by the paper from choices needed for a CausaLab task. A paper
may leave prompt rendering, equivalent answer forms, invalid combinations,
balancing, or counterfactual construction unspecified. Mark those gaps explicitly
and request a decision instead of guessing.

Apply [`task_quality_objectives.md`](task_quality_objectives.md). Pay particular
attention to multi-token answers, ambiguous grading, bundled causal variables, and
mechanisms that are undefined on counterfactual combinations the original dataset
never sampled.
