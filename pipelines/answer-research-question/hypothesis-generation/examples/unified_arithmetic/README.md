# Worked example: a shared calculator across arithmetic domains

A reference (not a scaffold to copy): the concrete causalab worked example for the
ideas in the handbook's "Designing causal-model hypotheses and counterfactual
datasets" section (the causal-analysis handbook,
`plugins/capabilities/research-handbooks/causal-handbook.md`); the
citations below name its subsections.
One causal model covers standard addition (`integer`, `age`) and the natural
domains (`weekdays`, `months`, `hours`, `alphabet`), with `domain` as an input. The
hypothesis under test: a single module computes `raw_sum` — the pre-reduction
integer sum — reused across every domain.

## Run it

> **Execution: stub.** This example was driven by the `develop_hypothesis` Hydra
> analysis, which imported `models` and `counterfactuals` from this directory and
> wrote `distinguishability.json`. That analysis and its runner were retired in
> the protocol refactor. The underlying primitive,
> `causalab.causal.causal_utils.can_distinguish_with_dataset`, still exists and
> the two modules here still import and run — what is missing is the harness that
> sweeps it into a matrix. See the stub in
> [`../../hypothesis-generation.md`](../../hypothesis-generation.md).

The files themselves are the point of the example: `models.py` shows the
family-of-variables factory and the total-encoder pattern, and
`counterfactuals.py` shows the cross-domain pairs the confounding check needs to
sample.

## What it demonstrates

- **Variable decomposition (one value per variable).** The shipped `natural_domains_arithmetic` task
  hides encode + add + modulus + decode inside one `result` mechanism; here each is
  its own variable (`entity_index`, `number_value`, `raw_sum`, `reduced`, `result`),
  so `raw_sum` can be named and localized.
- **Coupled inputs (one value per variable).** `entity` and `domain` are separate variables but a token is
  only valid within its domain, so the encoders are made *total* (global fallback)
  to survive cross-domain interchange.
- **What a pair can deconfound depends on the pair (fixable vs. structural confounding).** `raw_sum` and `reduced`
  are confounded by within-domain pairs but deconfounded by cross-domain ones; the
  large random run reports `{ raw_sum, operands }` and `{ result, all }` as
  always-confounded only because it samples cross-domain pairs.
- **Shared-vs-split is a generalization question (train/eval & generalization).** `raw_sum` being one shared
  module is invisible to the distinguishability matrix; the `integer_only` /
  `weekdays_only` datasets are the train / held-out split for the DAS cross-domain
  generalization test that actually settles it.
- **Cross-task patching.** `run_interchange(base, {"raw_sum": cf})` transplants a
  sum from one domain into another, which re-reduces it under the base domain.
