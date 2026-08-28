# Task conventions

- The package name under `causalab/tasks/` is the task loader key.
- Export either a singleton `CAUSAL_MODEL` or a factory
  `CREATE_CAUSAL_MODEL(config)`, not both.
- Export `TARGET_VARIABLE` when the task has a normal intervention target.
- Put one conceptual value in each causal variable. Generate variable families
  instead of storing lists or dictionaries when the elements need independent
  interventions.
- Make every mechanism total over combinations that interchange can construct.
- Declare answer surface forms with `CausalModel.output_tokens`. Use a custom
  checker only when those forms and match modes are insufficient.
- A protocol consumes serialized tables, never task code.
- Put all row-specific task semantics into table columns.
- Each serialized example has one base and exactly one counterfactual input.
- Generators are deterministic for a fixed configuration, `n`, and seed.
- Do not add task YAML, Hydra defaults, analysis configuration, scheduler options,
  or model loading to a task package.

See [`../../../causalab/tasks/README.md`](../../../causalab/tasks/README.md) for
the loader's exact exports and serialization contract.
