"""Shared helpers for experiment analyses.

Utility functions for task resolution, dataset generation, pipeline loading,
intervention metrics, and discovery of prior analysis outputs.
"""

from __future__ import annotations

import logging
from typing import Any

from causalab.tasks.loader import (
    Task,
    load_task,
    load_task_counterfactuals,
)
from causalab.io.pipelines import (  # noqa: F401  re-exported for back-compat
    DTYPE_MAP,
    load_pipeline,
    find_subspace_dirs,
    find_activation_manifold_dirs,
    load_locate_result,
    load_subspace_metadata,
    load_activation_manifold_metadata,
)

logger = logging.getLogger(__name__)


_TASK_CONFIG_EXCLUDE = {"isometry"}


def _task_config_for_metadata(task_config: dict[str, Any]) -> dict:
    """Filter task config dict for metadata, excluding evaluation-specific keys.

    Excludes evaluation-specific nested configs (isometry) which are analysis
    config, not task identity.
    """
    return {k: v for k, v in task_config.items() if k not in _TASK_CONFIG_EXCLUDE}


# ───────────────────────────────────────────────────────────────────────
# Task / data / model loading
# ───────────────────────────────────────────────────────────────────────


def resolve_task(
    task_name: str,
    task_config: dict[str, Any],
    target_variable: str,
    seed: int | None = None,
) -> tuple[Task, Any]:
    """Build a Task from explicit parameters.

    ``task_config`` contains task-specific fields (domain_type, graph_type, etc.).
    ``target_variable`` sets the intervention variable on the returned task and
    overrides the module's TARGET_VARIABLE export.  It is required — passing
    ``None`` raises ``ValueError`` to prevent silent use of a wrong default.
    """
    if target_variable is None:
        raise ValueError(
            "resolve_task() requires an explicit target_variable. "
            "Pass the variable name you want to localize (e.g. 'answer', 'color'). "
            "Omitting it silently uses the module default, which is often wrong."
        )
    task_cfg_raw = None

    if task_name == "graph_walk":
        from causalab.tasks.graph_walk.config import GraphWalkConfig

        task_cfg_raw = GraphWalkConfig(
            graph_type=task_config["graph_type"],
            graph_size=task_config["graph_size"],
            graph_size_2=task_config["graph_size_2"],
            context_length=task_config["context_length"],
            separator=task_config["separator"],
            no_backtrack=task_config["no_backtrack"],
            seed=seed,
        )
    elif task_name == "natural_domains_arithmetic":
        from causalab.tasks.natural_domains_arithmetic.config import NaturalDomainConfig

        task_cfg_raw = NaturalDomainConfig(
            domain_type=task_config["domain_type"],
            number_range=task_config.get("number_range", None),
            number_groups=task_config.get("number_groups", None),
            result_entities=task_config.get("result_entities", None),
        )

    task = load_task(task_name, task_cfg=task_cfg_raw)
    task.intervention_variable = target_variable
    return task, task_cfg_raw


def _deduplicate_by_input(examples: list) -> list:
    """Remove examples with duplicate raw_input prompts, keeping the first."""
    seen: set[str] = set()
    unique = []
    for ex in examples:
        key = ex["input"]["raw_input"]
        if key not in seen:
            seen.add(key)
            unique.append(ex)
    return unique


def _sample_single_variable_counterfactual(model, base, variable: str):
    """Build a counterfactual trace that differs from ``base`` only in ``variable``.

    The new value is drawn uniformly from ``model.values[variable]`` excluding
    ``base[variable]``, so the pair is guaranteed to differ on exactly one
    input variable.
    """
    import random as _rng

    if variable not in model.inputs:
        raise ValueError(
            f"resample_variable={variable!r} is not an input variable of "
            f"task {model.id!r}. Available inputs: {list(model.inputs)}."
        )
    choices = [v for v in model.values[variable] if v != base[variable]]
    if not choices:
        raise ValueError(
            f"Cannot resample variable {variable!r}: only one possible value."
        )
    cf_inputs = {var: base[var] for var in model.inputs}
    cf_inputs[variable] = _rng.choice(choices)
    return model.new_trace(cf_inputs)


def _generate_single_variable_dataset(
    model,
    n: int,
    seed: int,
    variable: str,
) -> list:
    """Generate ``n`` pairs where each CF resamples only ``variable``."""
    import random as _rng

    state = _rng.getstate()
    _rng.seed(seed)
    try:
        examples = []
        for _ in range(n):
            base = model.sample_input()
            cf = _sample_single_variable_counterfactual(model, base, variable)
            examples.append({"input": base, "counterfactual_inputs": [cf]})
        return examples
    finally:
        _rng.setstate(state)


def _generate_balanced(
    model,
    n: int,
    seed: int,
    iv: str,
    values: list,
) -> list:
    """Generate n examples balanced across intervention variable values."""
    import random as _rng

    state = _rng.getstate()
    _rng.seed(seed)
    examples = []
    for i in range(n):
        val = values[i % len(values)]
        base = model.sample_input(filter_func=lambda t, v=val: t[iv] == v)
        cf_val = values[(i + len(values) // 2) % len(values)]
        cf = model.sample_input(filter_func=lambda t, v=cf_val: t[iv] == v)
        examples.append({"input": base, "counterfactual_inputs": [cf]})
    _rng.setstate(state)
    return examples


def generate_datasets(
    task: Task,
    n_train: int,
    n_test: int,
    seed: int,
    deduplicate: bool = True,
    balanced: bool = False,
    enumerate_all: bool = False,
    resample_variable: str = "all",
) -> tuple[list, list]:
    """Generate train and test counterfactual datasets.

    When enumerate_all=True and n_unique_inputs <= n_train, enumerates
    all unique input combinations instead of sampling.  This avoids
    wasted sampling + deduplication for tasks with small fixed input sets.
    In this mode train and test are the same enumerated set (there is no
    held-out split — every possible input is used).

    When deduplicate=True, removes examples with duplicate raw_input
    prompts (since the model forward pass is deterministic, duplicate
    prompts produce identical activations and are wasted compute).

    Set deduplicate=False when regenerating a dataset that must align
    row-by-row with previously saved features.

    When balanced=True, cycles through intervention variable values
    to ensure equal per-class counts.

    ``resample_variable`` controls which input variable(s) the counterfactual
    resamples. ``"all"`` (default) delegates to the task's hand-written
    generator, which typically resamples every input independently. A single
    variable name (e.g. ``"entity"``) bypasses the task's generator and
    produces pairs that differ from the original only in that one variable —
    required for ``locate`` pairwise mode and any analysis that scores a
    single variable via interchange patching. ``balanced=True`` takes
    precedence when both are set.
    """
    model = task.causal_model

    if enumerate_all and model.n_unique_inputs <= n_train:
        import random as _rng

        _rng_state = _rng.getstate()
        _rng.seed(seed)
        traces = model.enumerate_inputs()
        if resample_variable == "all":
            train = [
                {"input": t, "counterfactual_inputs": [model.sample_input()]}
                for t in traces
            ]
        else:
            train = [
                {
                    "input": t,
                    "counterfactual_inputs": [
                        _sample_single_variable_counterfactual(
                            model,
                            t,
                            resample_variable,
                        )
                    ],
                }
                for t in traces
            ]
        _rng.setstate(_rng_state)
        logger.info(
            "Exhaustive enumeration: %d unique input combinations "
            "(resample_variable=%s)",
            len(train),
            resample_variable,
        )
        return train, list(train)

    if balanced and task.intervention_variable:
        iv = task.intervention_variable
        values = list(task.intervention_values)
        train = _generate_balanced(model, n_train, seed, iv, values)
        if deduplicate:
            before = len(train)
            train = _deduplicate_by_input(train)
            if len(train) < before:
                logger.info(
                    "Deduplicated train: %d -> %d unique prompts", before, len(train)
                )
        test = (
            _generate_balanced(model, n_test, seed + 1, iv, values)
            if n_test > 0
            else []
        )
        if n_test > 0 and deduplicate:
            before = len(test)
            test = _deduplicate_by_input(test)
            if len(test) < before:
                logger.info(
                    "Deduplicated test: %d -> %d unique prompts", before, len(test)
                )
        logger.info("Dataset (balanced): %d train, %d test", len(train), len(test))
        return train, test

    if resample_variable != "all":
        train = _generate_single_variable_dataset(
            model,
            n_train,
            seed,
            resample_variable,
        )
        if deduplicate:
            before = len(train)
            train = _deduplicate_by_input(train)
            if len(train) < before:
                logger.info(
                    "Deduplicated train: %d -> %d unique prompts",
                    before,
                    len(train),
                )
        if n_test > 0:
            test = _generate_single_variable_dataset(
                model,
                n_test,
                seed + 1,
                resample_variable,
            )
            if deduplicate:
                before = len(test)
                test = _deduplicate_by_input(test)
                if len(test) < before:
                    logger.info(
                        "Deduplicated test: %d -> %d unique prompts",
                        before,
                        len(test),
                    )
        else:
            test = []
        logger.info(
            "Dataset (resample_variable=%s): %d train, %d test",
            resample_variable,
            len(train),
            len(test),
        )
        return train, test

    cf_mod = load_task_counterfactuals(task.name)

    if deduplicate:
        train = _deduplicate_by_input(cf_mod.generate_dataset(model, n_train, seed))
        if len(train) < n_train:
            logger.info(
                "Deduplicated train: %d -> %d unique prompts", n_train, len(train)
            )
    else:
        train = cf_mod.generate_dataset(model, n_train, seed)

    if n_test > 0:
        if deduplicate:
            test = _deduplicate_by_input(
                cf_mod.generate_dataset(model, n_test, seed + 1)
            )
            if len(test) < n_test:
                logger.info(
                    "Deduplicated test: %d -> %d unique prompts", n_test, len(test)
                )
        else:
            test = cf_mod.generate_dataset(model, n_test, seed + 1)
    else:
        test = []

    logger.info("Dataset: %d train, %d test", len(train), len(test))
    return train, test


def build_targets_for_grid(
    pipeline,
    task: Task,
    layers: list[int],
    position_names: list[str] | None = None,
):
    """Build residual stream interchange targets for a (layer × token_position) grid.

    ``position_names=None`` uses all positions declared by the task; otherwise the
    names are looked up in ``task.create_token_positions(pipeline)``. Returns the
    targets dict (keys ``(layer, pos_id)``) and the ordered list of TokenPositions
    that the caller can use for plotting axes.
    """
    from causalab.neural.activations.targets import build_residual_stream_targets

    token_position_lookup = task.create_token_positions(pipeline)
    if position_names is None:
        token_positions = list(token_position_lookup.values())
    else:
        missing = [n for n in position_names if n not in token_position_lookup]
        if missing:
            raise ValueError(
                f"Unknown token positions {missing} for task {task.name!r}. "
                f"Available: {sorted(token_position_lookup)}"
            )
        token_positions = [token_position_lookup[n] for n in position_names]

    targets = build_residual_stream_targets(
        pipeline=pipeline,
        layers=layers,
        token_positions=token_positions,
        mode="one_target_per_unit",
    )
    return targets, token_positions


def build_targets_for_layers(pipeline, task: Task, layers: list[int]):
    """Back-compat wrapper: single-position targets keyed (layer, pos_id).

    Picks the first token position declared by the task. Prefer
    :func:`build_targets_for_grid` for new code.
    """
    token_position_lookup = task.create_token_positions(pipeline)
    pos_name = next(iter(token_position_lookup))
    targets, positions = build_targets_for_grid(
        pipeline, task, layers, position_names=[pos_name]
    )
    return targets, positions[0]


def get_output_token_ids(task: Task, pipeline):
    """Tokenize the task's output token values (or causal values if no deduplication).

    Returns (token_ids, n_tokens) or (None, None) if no target variable.
    """
    from causalab.methods.metric import tokenize_variable_values

    token_values = task.output_token_values or task.intervention_values
    if not token_values:
        return None, None
    token_ids = tokenize_variable_values(
        pipeline.tokenizer, token_values, task.result_token_pattern
    )
    n_tokens = len(token_ids) if isinstance(token_ids, list) else token_ids.shape[0]
    return token_ids, n_tokens


def _string_match_metric(neural_output: dict, causal_output: str) -> bool:
    """String containment metric for intervention success."""
    neural_str = neural_output["string"].strip().lower()
    causal_str = causal_output.strip().lower()
    return causal_str in neural_str or neural_str in causal_str


_STRING_METRICS = {
    "string_match": _string_match_metric,
}


def _argmax_accuracy(reference, predicted):
    """Fraction of examples where the top predicted token matches the reference.

    Higher is better.  Signature: ``(N, C), (N, C) -> (N,)``.
    """
    return (reference.argmax(dim=-1) == predicted.argmax(dim=-1)).float()


def resolve_intervention_metric(intervention_metric: str):
    """Resolve intervention metric by name.

    Returns ``(string_metric_fn, comparison_fn)``.  All comparison functions
    follow the **higher-is-better** convention.  Divergence metrics (KL,
    Hellinger) are negated so that values closer to zero (= less divergence)
    rank higher.
    """
    from causalab.methods.metric import DISTRIBUTION_COMPARISONS

    name = intervention_metric

    if name in _STRING_METRICS:
        return _STRING_METRICS[name], _argmax_accuracy
    if name in DISTRIBUTION_COMPARISONS:
        raw_fn = DISTRIBUTION_COMPARISONS[name]

        def _negated(ref, pred, _fn=raw_fn):
            return -_fn(ref, pred)

        return _string_match_metric, _negated
    raise ValueError(
        f"Unknown intervention_metric: {name!r}. "
        f"Available: {', '.join(sorted(set(_STRING_METRICS) | set(DISTRIBUTION_COMPARISONS)))}"
    )
