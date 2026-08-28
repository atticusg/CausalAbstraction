# Add or update a task

A task package defines a high-level causal model and counterfactual generators. A
protocol never imports that package. Before a protocol runs, the task is converted
into a deterministic JSON table whose columns contain prompts, answers, causal
labels, answer forms, and variable values.

The authoritative package contract is
[`../../../causalab/tasks/README.md`](../../../causalab/tasks/README.md). Read it
before editing a task.

## 1. Write the specification

Fill [`TASK_TEMPLATE.md`](TASK_TEMPLATE.md). Apply the quality objectives in
[`instructions/task_quality_objectives.md`](instructions/task_quality_objectives.md).
Name the competing causal variables and the counterfactual pairs needed to
distinguish them before writing code.

## 2. Start from a shipped task

Copy the closest package under `causalab/tasks/`:

- `hierarchical_equality` for a fixed singleton task;
- `natural_domains_arithmetic` for a dataclass-configured factory;
- `entity_binding` for generated families of scalar variables;
- `MCQA` for custom answer semantics.

Do not copy task scaffolds from before the protocol runner. Start from a package
that is exercised by the current serializer tests.

## 3. Implement the package

The normal modules are:

| File | Responsibility |
|---|---|
| `causal_models.py` | Export `CAUSAL_MODEL` or `CREATE_CAUSAL_MODEL`, plus `TARGET_VARIABLE` |
| `counterfactuals.py` | Export `generate_dataset(model, n, seed)` and any named alternatives |
| `config.py` | Define a dataclass for a factory task when needed |
| `token_positions.py` | Optional compatibility helpers for task-aware callers |
| `checker.py` | Optional custom output matcher |
| `__init__.py` | Re-export the public task surface |

Declare answer forms on the `CausalModel` with `output_tokens`. The loader derives
the normal string checker from that declaration. Add `checker.py` only for
semantics that cannot be expressed by those forms and the exact or prefix match
mode.

Every counterfactual example used by the protocol serializer must contain exactly
one counterfactual input. Generators must restore random state or use a local
generator so the same task, configuration, sample count, and seed produce identical
bytes.

## 4. Validate the causal model

Test that:

- every mechanism is defined for all inputs an intervention can create;
- hypothesized intermediates lie on a path to `raw_output`;
- `raw_input` and `raw_output` are deterministic strings;
- every declared answer value has surface forms;
- generators create valid pairs and expose the intended distinctions;
- the same seed produces the same examples.

Use `scripts/run_hypothesis_generation.py` when the task is being developed from a
set of competing models and datasets.

## 5. Serialize protocol data

```bash
uv run python scripts/build_task_dataset.py \
  --task <task_name> \
  --n 300 --seed 0 \
  --target-variable <variable> \
  --out data/<task_name>/train.json
```

Factory tasks accept repeated `--set key=value` arguments. Use `--generator` for a
named generator and `--answer-variable` when the model declares forms for several
variables.

The builder writes the JSON table and a provenance manifest. Re-run with `--check`
to assert byte-for-byte determinism.

## 6. Validate a protocol against the table

```bash
uv run causalab validate experiment.json --data-root data --data
uv run causalab explain experiment.json --data-root data
```

Inspect rows directly. Confirm that `label` is the causal model's output after the
declared interchange, not merely the counterfactual prompt's own answer. Confirm
that every metric and variable or column position references a serialized column.

## 7. Test and ship

Add task tests beside the package and serializer tests when the shared row contract
changes. At minimum run:

```bash
uv run pytest -q tests/tasks/test_serialize.py
uv run python scripts/build_task_dataset.py ... --check
```

Commit the package, tests, and any dataset tables that shipped protocol documents
need. Generated tables are build products; their manifests record how to reproduce
them.
