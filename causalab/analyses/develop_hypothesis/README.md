# develop_hypothesis

develop_hypothesis answers: *do a session's counterfactual datasets actually distinguish its competing causal-model hypotheses?* It loads the hand-authored `models.py` / `counterfactuals.py` scaffolds named by `develop_hypothesis.hypotheses_dir`, runs the CPU distinguishability engine in `causalab.causal.causal_utils` (`distinguishability_report`, beside the `can_distinguish_with_dataset` primitive it generalizes), and writes the target-centric rate matrix plus the always-confounded groups. It is a **task-less** analysis (like `characterize_subspace`): its inputs are the causal model + counterfactual generators, not a runner-generated task dataset, and it loads no neural model. It runs at the causal-model level on CPU, before any neural localization.

## Configuration

**Root config** (`causalab/configs/config.yaml`)
- `experiment_root` — output root; the report lands under `${experiment_root}/develop_hypothesis/`.
- `seed` — seed passed to the counterfactual generators (`make_datasets`, `random_pairs`).

**Module config** (`causalab/configs/analysis/develop_hypothesis.yaml`)
```yaml
develop_hypothesis:
  _name_: develop_hypothesis
  _subdir: n${.n}                    # output subdir discriminator
  _output_dir: ${experiment_root}/develop_hypothesis/${._subdir}
  hypotheses_dir: ???                # dir with the session's models.py + counterfactuals.py
  n: 300                             # examples per design dataset (target-centric baselines)
  random_n: 100000                   # pairs for the large always-confounded run
  confound_threshold: 0.2            # alternatives below this vs a target are flagged (display only)
```

The session modules must expose: `models.MODELS` (name → `CausalModel`), `models.DEFAULT_MODEL`, `models.HYPOTHESES` (name → `(model name, [target variables])`), optional `models.TARGETS`; and `counterfactuals.make_datasets(model, n=, seed=)`, `counterfactuals.random_pairs(model, n, seed=)`, optional `counterfactuals.DATASET_ROLES`. The `null` / `all` reference hypotheses are injected automatically.

## Outputs

### Interpretation
- **`distinguishability.json`** — the certification. Per design dataset, a target-centric table: for each focal target hypothesis, the rate at which each alternative's intervened output differs from the target's (plus `vs_null` / `vs_all`). Read as interpretive baselines, not pass/fail: ~0.50 on a wide dataset is information, not failure; a per-dataset 0.00 vs a target is *fixable* confounding (design a sharper narrow dataset). The `always_confounded` groups (from the large random run) are hypotheses no sampled pair deconfounds — confounded everywhere, so pick one representative. `singletons` are everything else. A good result cleanly separates each target from its alternatives on at least one dataset; a bad result leaves a contest you care about confounded on every dataset.

### Saved artifacts

| File | Shape / Format | Used by |
|---|---|---|
| `distinguishability.json` | `{n, random_n, seed, hypotheses, targets, dataset_roles, datasets, always_confounded, singletons}` | the causal-model design report (Step 6); provenance |

The same payload is printed as a table (target-centric rates + always-confounded groups) for quick reading.
