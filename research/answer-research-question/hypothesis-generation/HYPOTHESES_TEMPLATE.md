# Dataset experiment: {intermediate variable}

_Use this template during hypothesis generation. A later experiment will test
where these variables are represented in the network._

## 1. Behavior and claim to test

- **Task / behavior:** {what the model does, what the output is}
- **Working prompt(s):** {representative input(s); link the working-prompt file if one exists}
- **Claim to test:** {a location in the network, identified by layer, activation
  stream, and token, faithfully represents {variable subset} from {causal model}}

## 2. Candidate causal model(s)

For each model in `code/hypotheses/models.py`:

### {model name}

- **DAG:** {ASCII or prose — inputs → intermediates → raw_output}
- **Variables:** inputs {…}; intermediates {…}; outputs {raw_input, raw_output}
- **Check the five task quality objectives** (granularity / grading totality /
  input determinism / decoding one token / value coverage): {state whether each
  objective passes and describe any limitations}
- **Check that each variable stores one value:** {note any variable that stores a
  list, dictionary, or tuple, and explain why it must combine several values}
- **Hypotheses that the random sample cannot distinguish:** {list hypotheses that
  produce the same outputs for every pair in the large random sample; see section
  5; for example, two variables may carry the same information}

## 3. Target and plausible alternatives

Name the one intermediate variable targeted by this experiment. List only input,
output, and competing intermediate variables that could plausibly be confused
with it. The code adds the null and full mediation references.

| Name | Role (target/alt) | Model | Target variables | Why it's interesting |
|------|-------------------|-------|------------------|----------------------|
| {h1} | target | {model} | {…} | {…} |
| {h2} | alternative | {model} | {…} | {…} |
| null | alternative (ref) | {model} | (none) | doing nothing |
| all | alternative (ref) | {model} | {all variables being tested} | transplant the whole output representation |

- **Architecture pruning:** {which subsets were ruled out by left-to-right flow /
  token layout, and why}
- **Variables excluded from comparison:** {each variable considered, and why it
  cannot plausibly be confused with the target}

## 4. Counterfactual datasets

List the broad, narrow, and any single-token datasets from
`counterfactuals.py`. Broad and narrow coverage are required. Add a single-token
dataset when it helps distinguish or trace the target.

| Dataset | Type | Data split and what it holds out | What it changes | Purpose |
|---------|--------------------------|-------------------------------|---------------------|------|
| {wide_…} | broad | train | {random / systematic} | broad coverage |
| {narrow_…} | narrow | eval (holds out {entities/templates}) | {fix A, flip B} | separate {hA vs hB} |
| {single_…} | single-token | train/eval | {flip one token = one input var} | trace {var}'s path through the network (low separating power) |

- **Generalization plan:** {what train trains on, what eval holds out, how
  overfitting will be checked — per the handbook's "Train/eval splits and generalization"}

## 5. Which hypotheses the datasets can distinguish

For every target and dataset, report how often each alternative produces a
different intervened output from the target. Also report how often the target
differs from the null and from full mediation:

```
dataset                     target              vs_null  vs_all   alternatives (rate vs target)
narrow_flip_carry_fix_ones  carry                1.00     0.00     ones 1.00
narrow_flip_ones_fix_carry  carry                0.00     1.00     ones 1.00   (no power for carry)
...
```

**Groups that the large random sample never distinguishes (N={random_n}):**

```
{ ... }   # no sampled pair deconfounds these — pick one representative
```

- **Interpretation:** {which datasets test each target; which alternatives remain
  ambiguous; how that affects later experiments; and which hypotheses no sampled
  pair distinguishes, including from the null. This is evidence, not proof.}

## 6. Conclusions and next step

- **Hypotheses worth carrying forward:** {…}
- **Questions to revisit:** {where the causal model is likely wrong, which result
  from a wide counterfactual dataset was surprising, and what to revise}
- **Next step:** {implement the chosen model + generators via the setup-task guide
  (`../../implementation/setup-task/setup-task.md`), design the later experiment
  that will test where the variables are represented in the network, or do both}
- **Valid hypothesis-testing comparisons:** {target-versus-alternative
  comparisons supported by the datasets}
- **Required outputs:** `distinguishability.json` and `report.html`
