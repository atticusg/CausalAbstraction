# Worked example: a shared calculator across arithmetic domains

This worked example applies the hypothesis design guidance in
[`../../hypothesis-generation.md`](../../hypothesis-generation.md). It is not a
template.

One causal model covers standard addition (`integer`, `age`) and the natural
domains (`weekdays`, `months`, `hours`, `alphabet`), with `domain` as an input. The
hypothesis under test states that one module computes `raw_sum`, which is the
integer sum before reduction, and that every domain reuses this module.

## Run it

From the CausaLab repository root, run:

```bash
uv run python scripts/run_hypothesis_generation.py \
  research/answer-research-question/hypothesis-generation/examples/unified_arithmetic
```

This writes `distinguishability.json` beside the example modules. The default
run uses 300 pairs per designed dataset and 100,000 broadly sampled pairs for the
check for hypotheses that remain confounded. Use smaller values for a smoke test:

```bash
uv run python scripts/run_hypothesis_generation.py \
  research/answer-research-question/hypothesis-generation/examples/unified_arithmetic \
  --n 10 --random-n 100
```

`models.py` defines the variables and complete encoder. `counterfactuals.py`
defines the pairs from different domains needed to distinguish hypotheses.

## What it demonstrates

- **Store one value in each variable.** The shipped `natural_domains_arithmetic`
  task combines encoding, addition, modulus, and decoding in one `result`
  mechanism. Here, each operation is
  its own variable (`entity_index`, `number_value`, `raw_sum`, `reduced`, `result`),
  so `raw_sum` can be named and localized.
- **Handle every possible combination of related inputs.** `entity` and `domain`
  are separate variables, but a token is normally valid only within its domain.
  The encoders use a global fallback so that they can also handle interchange
  between domains.
- **Whether a pair can distinguish two hypotheses depends on how the pair was
  constructed.** Pairs from the same domain cannot distinguish `raw_sum` from
  `reduced`, but pairs from different domains can. Because the large random sample
  includes pairs from different domains, the remaining groups that it cannot
  distinguish are `{ raw_sum, operands }` and `{ result, all }`.
- **Testing whether conditions share a module requires a generalization
  experiment.** The distinguishability matrix cannot show whether one module
  computes `raw_sum` for every domain. The DAS generalization test trains on the
  `integer_only` dataset and evaluates on the `weekdays_only` dataset, which was
  held out from training.
- **Interchange between tasks.** `run_interchange(base, {"raw_sum": cf})`
  transplants a sum from one domain into another, then reduces it under the base
  domain.
