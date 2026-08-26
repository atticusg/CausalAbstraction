# Hypotheses: {task / behavior}

_Produced by hypothesis generation (`hypothesis-generation.md`). Causal-level
design; neural localization is the downstream experiment._

## 1. Behavior and anchor claim

- **Task / behavior:** {what the model does, what the output is}
- **Working prompt(s):** {representative input(s); link the working-prompt file if one exists}
- **Anchor claim being pursued:** {causal model} + {variable subset} located at
  {intended neural location — layer / stream / token, stated as the eventual target}

## 2. Candidate causal model(s)

For each model in `code/hypotheses/models.py`:

### {model name}

- **DAG:** {ASCII or prose — inputs → intermediates → raw_output}
- **Variables:** inputs {…}; intermediates {…}; outputs {raw_input, raw_output}
- **Quality-objective check** (granularity / grading totality / input determinism
  / single-token decoding / value coverage): {pass/with-caveats per objective}
- **One-value-per-variable check:** {does each variable hold a single value? note
  any list/dict/tuple-valued variable and why the bundling is justified}
- **Always-confounded notes:** {hypotheses the large random run leaves mutually
  confounded — see §5 — e.g. two variables that carry the same information}

## 3. Hypotheses: targets and alternatives

The focal **target hypotheses** (a group is fine), and the **alternatives** they
are scored against. The null and full-mediation slices are injected automatically.

| Name | Role (target/alt) | Model | Target variables | Why it's interesting |
|------|-------------------|-------|------------------|----------------------|
| {h1} | target | {model} | {…} | {…} |
| {h2} | alternative | {model} | {…} | {…} |
| null | alternative (ref) | {model} | (none) | doing nothing |
| all | alternative (ref) | {model} | {mediating slice} | whole output transplant |

- **Architecture pruning:** {which subsets were ruled out by left-to-right flow /
  token layout, and why}

## 4. Counterfactual datasets

From `code/hypotheses/counterfactuals.py`. Keep all altitudes.

| Dataset | Wide/narrow/single-token | Split (train/eval, holds out) | What it manipulates | Role |
|---------|--------------------------|-------------------------------|---------------------|------|
| {wide_…} | wide | train | {random / systematic} | broad coverage |
| {narrow_…} | narrow | eval (holds out {entities/templates}) | {fix A, flip B} | separate {hA vs hB} |
| {single_…} | single-token | train/eval | {flip one token = one input var} | trace {var}'s path through the network (low separating power) |

- **Generalization plan:** {what train trains on, what eval holds out, how
  overfitting will be checked — per the handbook's "Train/eval splits and generalization"}

## 5. Distinguishability (target-centric baselines + always-confounded groups)

From the distinguishability check (`hypothesis-generation.md` Step 3d). Note that
the harness that produced `distinguishability.json` was retired in the protocol
refactor; see that document's execution stub.

**Per target, per dataset** — each alternative's rate of differing from the
target, plus the target's vs-null (does the counterfactual dataset move it at all) and vs-all:

```
dataset                     target              vs_null  vs_all   alternatives (rate vs target)
narrow_flip_carry_fix_ones  carry                1.00     0.00     ones 1.00
narrow_flip_ones_fix_carry  carry                0.00     1.00     ones 1.00   (no power for carry)
...
```

**Always-confounded groups (large random run, N={random_n}):**

```
{ ... }   # no sampled pair deconfounds these — pick one representative
```

- **Reading:** {which counterfactual datasets give each target power; which alternatives stay
  confounded with a target on which counterfactual datasets, and what that means for interpreting a
  later neural result; which hypotheses no pair deconfounds (incl. any confounded
  with the null = inert under the sampler). Note this is empirical, not proof.}

## 6. Conclusions and next step

- **Hypotheses worth carrying forward:** {…}
- **Open / iterate:** {where the causal model is likely still wrong; what wide counterfactual dataset
  surprised us; what to revise}
- **Next step:** {implement the chosen model + generators via the setup-task guide
  (`../../implementation/setup-task/setup-task.md`), and/or design the downstream neural
  localization test}
