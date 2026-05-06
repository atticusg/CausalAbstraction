# Natural Domains Arithmetic

A factory task for arithmetic over a finite domain. The same causal DAG `(entity, number) → result → raw_output` is parametrized across six domains (weekdays, months, hours, age, alphabet, integer) so a single implementation drives every variant — replacing the deprecated standalone `weekdays/` and `months/` tasks.

A prompt looks like `"Q: What day is three days after Thursday?\nA:"` and the model is expected to produce `" Sunday"`. Swap the domain and the same shape of question covers month arithmetic, hour-on-a-clock arithmetic, integer addition, age arithmetic, and alphabet shifts.

## Domain Matrix

Six presets are bundled in `config.py::DOMAIN_PRESETS`. Each one has a matching Hydra task config under `causalab/configs/task/natural_domains_arithmetic_<domain>.yaml`.

| Domain | Cyclic? | Modulus | Entity vocab | Number vocab | Template | Task config |
|---|---|---|---|---|---|---|
| `weekdays` | yes | 7 | `Monday`…`Sunday` | `one`…`seven` | `Q: What day is {number} days after {entity}?\nA:` | `natural_domains_arithmetic_weekdays.yaml` |
| `months` | yes | 12 | `January`…`December` | `one`…`twelve` | `Q: What month is {number} months after {entity}?\nA:` | `natural_domains_arithmetic_months.yaml` |
| `hours` | yes | 24 | `1`…`24` | `one`…`twenty-four` | `Q: What hour comes {number} hours after {entity} on a clock?\nA: ` | `natural_domains_arithmetic_hours.yaml` |
| `integer` | no | — | word-form `one`…`fifteen` | word-form `one`…`nine` | `Q: What is {number} added to {entity}?\nA:` | `natural_domains_arithmetic_integer.yaml` |
| `age` | no | — | `1`…`99` | `1`…`10` | `Alice is {entity} years old. Bob is {number} years older than Alice. Q: How old is Bob?\nA: Bob is ` | `natural_domains_arithmetic_age.yaml` |
| `alphabet` | no | — | `A`…`Y` | `one`…`three` | `The letter {number} after {entity} in the alphabet is the letter` | `natural_domains_arithmetic_alphabet.yaml` |

For non-cyclic domains an `input_filter` (set in `causal_models.py`) drops `(entity, number)` pairs whose result would fall outside `result_entities` (e.g. `alphabet: Z + two` is excluded). Cyclic domains wrap with the modulus and require no filtering.

## Causal Model

Five variables. The DAG is identical for every domain; only the `compute_result` function differs:

```
number ──┐
         ├──> result ──> raw_output
entity ──┤
         └──> raw_input
```

| Variable | Role | Notes |
|---|---|---|
| `entity` | input — the starting domain element | E.g. `"Thursday"`, `"July"`, `"M"` |
| `number` | input — the offset to add | Word-form for cyclic domains, digit for `age` |
| `result` | computed answer | Cyclic: `(entity_idx + number) % modulus`. Non-cyclic: a custom `compute_result` from the preset. With `number_groups`, becomes a tuple `(entity_result, group_index)`. |
| `raw_input` | rendered prompt string | `template.format(entity=…, number=…)` |
| `raw_output` | expected continuation | `output_prefix + result` |

A multi-template variant is supported: pass `template=[...]` to `NaturalDomainConfig` and a `template` input variable is added so each example samples one of the templates. `token_positions.py` builds a per-template dispatcher automatically.

The model is built by `create_causal_model(config: NaturalDomainConfig)` in `causal_models.py`. A random-word baseline (entities replaced by random English words, cyclic mod arithmetic) is available via `create_random_causal_model` for sanity-checking that performance is driven by domain knowledge rather than surface form.

### Embeddings & periods

`create_causal_model` registers value embeddings used by downstream geometry analyses:

- `entity` and `result` are embedded by their domain index (or by `entity_embedding` from the preset, e.g. integer-valued for `age`).
- `number` is embedded by its integer value via `number_to_int`.
- For cyclic domains, `periods["entity"]` and `periods["result"]` are set to `modulus` so isometry analyses know the geometry is circular. `number` is also marked cyclic for `weekdays` (where `number_is_cyclic=True`).

## Token Positions

`token_positions.py::create_token_positions(pipeline, template=...)` returns:

| Name | Description |
|---|---|
| `last_token` | The final prompt token (index `-1`). |
| `entity` | The last token spanning the `{entity}` slot. |
| `number` | The last token spanning the `{number}` slot. |

Each position is built by `causalab.neural.token_positions.build_token_position_factories` from a declarative spec (no per-model hardcoding). For multi-template configs, pass `templates=[...]` instead and the returned `TokenPosition` objects dispatch on `input_sample["template"]` at index time.

## Counterfactuals

`counterfactuals.py::generate_dataset(model, n, seed)` returns `n` examples of shape `{"input": ..., "counterfactual_inputs": [...]}` where both base and counterfactual are independent samples — every input variable may differ.

Single-variable counterfactuals (only one variable resampled) are configured via the runner config rather than the task module: set `task.resample_variable: <var>` and `runner/helpers.py::generate_datasets` re-derives the counterfactuals. This is required when running `analysis/locate` in `pairwise` mode (see `ARCHITECTURE.md` §5) — pairwise patching is only meaningful when exactly one input variable changes.

## How to Run

Each domain has at least one runner preset already wired up:

```bash
# Single-step runs
./scripts/run_exp.sh weekdays_8b_baseline       # baseline only
./scripts/run_exp.sh weekdays_8b                # subspace at layer 28
./scripts/run_exp.sh age_8b_pullback            # geodesic pullback
./scripts/run_exp.sh integer_dual_manifold      # activation+output manifolds

# Multi-step pipeline
./scripts/run_exp.sh weekdays_8b_pipeline
```

Available analyses for this task: `baseline`, `locate`, `subspace`, `activation_manifold`, `output_manifold`, `path_steering`, `pullback` (see the project root `ARCHITECTURE.md` and per-analysis READMEs for what each one answers).

Outputs land under `artifacts/natural_domains_arithmetic/<model>/<analysis>/...` per `ARCHITECTURE.md` invariant 7.

## Files

| File | Role |
|---|---|
| `config.py` | `NaturalDomainConfig` dataclass + the `DOMAIN_PRESETS` table |
| `causal_models.py` | `create_causal_model`, `create_random_causal_model`, plus the `GET_*` accessors used by `tasks/loader.py` |
| `counterfactuals.py` | `generate_dataset` |
| `token_positions.py` | `create_token_positions` (single- or multi-template) |
| `demo.ipynb` | Runnable walkthrough of the causal model, tokenization, and counterfactuals |
