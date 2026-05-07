# Task Definitions for Causal Abstraction Experiments

Each task is a self-contained package under `causalab/tasks/<name>/` that
defines a causal model, counterfactual generation, and tokenization helpers.
Tasks are loaded at runtime via `load_task()` in `causalab.tasks.loader`.

## Creating a new task

Create a directory `causalab/tasks/<name>/` with these modules:

### causal_models.py (required)

Defines the causal model and exports that `load_task()` reads by convention.

**Singleton tasks** (fixed structure, e.g. weekdays, months, years):

| Export | Type | Required |
|--------|------|----------|
| `CAUSAL_MODEL` | `CausalModel` | yes |
| `VARIABLE_VALUES` | `dict[str, list[str]]` — var name → values | yes |
| `CYCLIC_VARIABLES` | `set[str]` — which variables wrap cyclically | yes (may be empty) |
| `EMBEDDINGS` | `dict[str, Callable]` — var name → embedding fn | yes |
| `PERIODIC_INFO` | `dict[str, int]` — var name → period length | no |
| `TEMPLATE` | `str` — prompt template | no |
| `RESULT_TOKEN_PATTERN` | `Callable[[str], str]` — format output token | no |
| `TARGET_VARIABLE` | `str` — variable being steered | no |
| `RANDOM_CAUSAL_MODEL` | `CausalModel` — random baseline model | no |
| `RANDOM_VARIABLE_VALUES` | `dict[str, list[str]]` — values for random baseline | no |

**Factory tasks** (parameterized, e.g. graph_walk, natural_domains_arithmetic):

| Export | Type | Required |
|--------|------|----------|
| `CREATE_CAUSAL_MODEL` | `Callable[[config], CausalModel]` | yes |
| `GET_VARIABLE_VALUES` | `Callable[[CausalModel], dict[str, list[str]]]` | yes |
| `CYCLIC_VARIABLES` | `set[str]` | yes (may be empty) |
| `EMBEDDINGS` | `dict[str, Callable]` | yes |
| `GET_CYCLIC_VARIABLES` | `Callable[[CausalModel], set[str]]` | no (overrides `CYCLIC_VARIABLES`) |
| `GET_EMBEDDINGS` | `Callable[[CausalModel], dict[str, Callable]]` | no (overrides `EMBEDDINGS`) |
| `GET_PERIODIC_INFO` | `Callable[[CausalModel], dict[str, int] \| None]` | no |
| `CREATE_RANDOM_CAUSAL_MODEL` | `Callable[[config], CausalModel]` | no |
| `TARGET_VARIABLE` | `str` | no |

### counterfactuals.py (required)

```python
generate_dataset(causal_model, n_examples, seed) → list[dict]
```

Each dict has `"input"` (a causal trace) and `"counterfactual_inputs"`
(list of counterfactual traces).

### token_positions.py (required for intervention experiments)

```python
create_token_positions(pipeline, ...) → dict[str, TokenPosition]
```

Maps position names to `TokenPosition` objects that locate where in
the token sequence to intervene.

## Active tasks

| Task | Description | Dimensionality |
|------|-------------|----------------|
| `natural_domains_arithmetic` | Unified weekdays/months/age/alphabet | factory, 1D (cyclic or linear) |
| `graph_walk` | Next-node prediction on graphs | factory, 1D or 2D |
