# Step 5 — Generalize results

A tested intermediate variable initially applies only to the model, task, prompt,
output target, prefix condition, token positions, datasets, and evaluation split
used in hypothesis testing. Generalization repeats the six core intervention
experiments in new settings and asks which parts of the layer-by-layer causal
account survive.

Do not generalize every positive result. Read `REPORT_PLAN.md`, select the main
claims that need broader evidence, and state which kind of generalization is
meaningful for each one. Several claims may share one generalization experiment.
Claims that remain outside this phase keep the narrower scope established during
hypothesis testing.

Run three canonical generalization experiments:

1. [New prompt templates](prompt-template-generalization.md), including another
   language when it expresses the same task.
2. [Related tasks](related-task-generalization.md) that may use the same
   intermediate variable for a different behavior.
3. [Naturally occurring text](in-the-wild-generalization.md) from web text,
   WikiText, fine-tuning data, or another corpus used for next-token prediction.

Applicable experiments may run in parallel after the selected claims and their
baseline causal accounts are stable. Give each its own directory, protocol
documents, run artifacts, and self-contained HTML report. Record why an experiment
type is not meaningful for a selected claim rather than manufacturing a weak
analogue.

## Repeat the core test

Each generalization experiment repeats:

- residual stream patching;
- attention output patching;
- MLP output patching;
- residual stream DAS;
- attention head DBM; and
- MLP neuron DBM.

Use the same definitions, controls, three-seed rule for DAS and DBM, report
metrics, and positive-control checks as hypothesis testing. Adapt the
counterfactual datasets to the new setting and certify that they still
distinguish the target from plausible input, output, and intermediate
alternatives before interpreting neural results.

Do not assume that a token location transfers. Identify critical tokens and spans
again in each setting. New prompt formats, languages, tasks, and natural text may
move the same semantic role to a new position or divide it across a different
span. For multi-token generations, preserve the semantic next-token target from
the child investigation and resolve its position again in every example.

## Update the causal account

The purpose is not only to report whether one score transfers. Each experiment
must revise the layer-by-layer causal account:

- where the relevant inputs enter;
- how attention moves or combines their information;
- how MLP layers preserve or transform it;
- where the intermediate variable becomes causally effective;
- how it reaches the output; and
- which layers remain unresolved.

Keep the original account and the new account side by side. A mechanism may
generalize while changing token location, or remain at the same location while
changing which components implement it. Report both kinds of change.

## Completion and handoff

Generalization is complete when every applicable report for the selected main
claims exists and `REPORT_PLAN.md` records why the other experiment types were
not applicable. For every setting, state the fraction of behavior explained and
the exact boundary of the claim.

A failed generalization is a result. Do not widen the claim to match the strongest
condition and do not hide settings where the positive control failed. If a new
setting suggests a different intermediate variable, create a new roadmap for that
question.

Update `ROADMAP.md` and `REPORT_PLAN.md`, then go to
[`../save-results/save-results.md`](../save-results/save-results.md).
