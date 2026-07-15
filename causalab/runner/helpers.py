"""Shared helpers for experiment analyses.

Utility functions for task resolution, dataset generation, pipeline loading,
intervention metrics, and discovery of prior analysis outputs.
"""

from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from causalab.methods.metric import InterchangeMetric

from causalab.tasks.loader import (
    Task,
    _has_model_export,  # pyright: ignore[reportPrivateUsage]
    _import_task_module,  # pyright: ignore[reportPrivateUsage]
    load_task,
    load_task_counterfactuals,
)
from causalab.io.pipelines import (
    DTYPE_MAP as DTYPE_MAP,
    load_pipeline as load_pipeline,
    find_subspace_dirs as find_subspace_dirs,
    find_activation_manifold_dirs as find_activation_manifold_dirs,
    load_locate_result as load_locate_result,
    load_subspace_metadata as load_subspace_metadata,
    load_activation_manifold_metadata as load_activation_manifold_metadata,
)

logger = logging.getLogger(__name__)


_TASK_CONFIG_EXCLUDE = {"isometry"}


def _task_config_for_metadata(task_config: dict[str, Any]) -> dict:  # pyright: ignore[reportUnusedFunction]
    """Filter task config dict for metadata, excluding evaluation-specific keys.

    Excludes evaluation-specific nested configs (isometry) which are analysis
    config, not task identity.
    """
    return {k: v for k, v in task_config.items() if k not in _TASK_CONFIG_EXCLUDE}


# ───────────────────────────────────────────────────────────────────────
# Task config validation
# ───────────────────────────────────────────────────────────────────────


def validate_task_config(
    task_cfg: Any,
    requirements: list[tuple[str, tuple[str, ...]]],
) -> None:
    """Fail fast if ``cfg.task`` lacks keys the requested analyses require.

    Factory tasks resolved by name with no shipped ``configs/task/<name>.yaml``
    can silently omit keys that shipped task configs always set —
    ``intervention_metric`` (read by direct attribute access) and
    ``colormap`` / ``colormap2`` (resolved through ``${task.*}`` interpolations
    baked into the analysis configs). Each otherwise raises one at a time, deep
    in a run after the model is already loaded (#264). #226 fixed factory
    *resolution*, not this required-keys contract.

    ``requirements`` pairs each requested analysis with the ``task.*`` keys it
    declares via its module-level ``REQUIRED_TASK_KEYS``. This collects EVERY
    missing key across all of them and raises a single ``ValueError`` naming
    each one and the analyses that need it, so a task missing several keys is
    fixed in one pass.

    Membership — not truthiness — is the test: an explicitly ``null`` key (e.g.
    ``colormap2: null``, which shipped configs and the task-config template
    of #263 emit) is present and valid; only *absence* breaks the interpolation.
    """
    if not requirements:
        return
    present = set(task_cfg.keys()) if task_cfg is not None else set()
    missing: dict[str, list[str]] = {}
    for analysis_name, keys in requirements:
        for key in keys:
            if key not in present:
                missing.setdefault(key, []).append(analysis_name)
    if not missing:
        return
    detail = "\n".join(
        f"  - task.{key} (required by: {', '.join(sorted(set(analyses)))})"
        for key, analyses in sorted(missing.items())
    )
    raise ValueError(
        "Task config is missing required key(s) for the requested analyses:\n"
        f"{detail}\n\n"
        "Factory tasks (no shipped configs/task/<name>.yaml) must set these "
        "explicitly; shipped task configs already do. Add the missing keys to "
        "your task config — an explicit null is fine (e.g. `colormap2: null`)."
    )


# ───────────────────────────────────────────────────────────────────────
# Task / data / model loading
# ───────────────────────────────────────────────────────────────────────


def resolve_task(
    task_name: str,
    task_config: dict[str, Any],
    target_variable: str | None,
    seed: int | None = None,
) -> tuple[Task, Any]:
    """Build a Task from explicit parameters.

    ``task_config`` contains task-specific fields (domain_type, graph_type, etc.).
    ``target_variable`` sets the intervention variable on the returned task and
    overrides the module's TARGET_VARIABLE export.  It is required — passing
    ``None`` raises ``ValueError`` to prevent silent use of a wrong default.

    The type allows ``None`` because callers commonly pass
    ``cfg.task.get("target_variable")`` from a Hydra config; the runtime
    check converts that into a clear error message at the boundary.
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

        # Omit `seed` when None so GraphWalkConfig's dataclass default applies
        # (single source of truth for the default seed).
        gw_kwargs: dict[str, Any] = {
            "graph_type": task_config["graph_type"],
            "graph_size": task_config["graph_size"],
            "graph_size_2": task_config["graph_size_2"],
            "context_length": task_config["context_length"],
            "separator": task_config["separator"],
            "no_backtrack": task_config["no_backtrack"],
        }
        if seed is not None:
            gw_kwargs["seed"] = seed
        task_cfg_raw = GraphWalkConfig(**gw_kwargs)
    elif task_name == "natural_domains_arithmetic":
        from causalab.tasks.natural_domains_arithmetic.config import NaturalDomainConfig

        task_cfg_raw = NaturalDomainConfig(
            domain_type=task_config["domain_type"],
            number_range=task_config.get("number_range", None),
            number_groups=task_config.get("number_groups", None),
            result_entities=task_config.get("result_entities", None),
        )
    elif task_name == "identity_naming":
        from causalab.tasks.identity_naming.config import IdentityNamingConfig

        task_cfg_raw = IdentityNamingConfig(
            domain_type=task_config["domain_type"],
        )

    # Generic factory support: a task that exports CREATE_CAUSAL_MODEL (and no
    # CAUSAL_MODEL singleton) but is not special-cased above receives the raw
    # Hydra task-config dict, so factory tasks introduced outside this function
    # (e.g. session-local research tasks selecting a `query`) load without a
    # bespoke branch here. Singletons and the special-cased factories above are
    # untouched (task_cfg_raw is already set, or stays None for singletons).
    if task_cfg_raw is None:
        # Resolve through the shared loader helper so the factory probe honours
        # the same shipped-first / session-local fallback as load_task itself.
        _cm = _import_task_module(task_name, "causal_models")
        if _has_model_export(_cm, "CREATE_CAUSAL_MODEL") and not _has_model_export(
            _cm, "CAUSAL_MODEL"
        ):
            task_cfg_raw = dict(task_config)

    # load_task accepts dict | None; the *Config dataclasses above are not dicts,
    # but the underlying loaders accept them at runtime (task-specific shims).
    task = load_task(task_name, task_cfg=task_cfg_raw)  # pyright: ignore[reportArgumentType]
    task.intervention_variable = target_variable
    # Apply the optional scoring-convention override (e.g. MCQA letter→value).
    # ``score_by`` is absent for tasks that don't declare alternatives, in
    # which case this is a no-op.
    task.apply_score_mode(task_config.get("score_by"))
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


def _prompt_input_vars(model) -> set[str]:
    """Input variables that (transitively) feed the rendered ``raw_input``.

    Walks ``model.parents`` back from ``raw_input`` to the parent-less input
    variables, so a prompt that depends on a derived variable still resolves to
    the underlying inputs. An input *not* in this set is invisible to the model:
    changing it cannot change the prompt.
    """
    inputs = set(model.inputs)
    deps: set[str] = set()
    seen: set[str] = set()
    stack = list(model.parents.get("raw_input", []))
    while stack:
        var = stack.pop()
        if var in seen:
            continue
        seen.add(var)
        if var in inputs:
            deps.add(var)
        else:
            stack.extend(model.parents.get(var, []))
    return deps


def _swept_input_vars(examples: list, model) -> set[str]:
    """Input variables taking more than one distinct value across ``examples``."""
    swept: set[str] = set()
    for var in model.inputs:
        values: set[str] = set()
        for ex in examples:
            values.add(str(ex["input"][var]))
            if len(values) > 1:
                swept.add(var)
                break
    return swept


def _conflicting_prompt_groups(examples: list, target: str | None) -> dict[str, set]:
    """Map each base ``raw_input`` to the distinct gold keys it renders to.

    A gold key is the pair ``(raw_output, intervention-target value)``. Only
    prompts that map to more than one gold key are returned: those are the
    prompts whose duplicates dedup would silently collapse to a single
    (arbitrary) label.
    """
    by_prompt: dict[str, set] = {}
    for ex in examples:
        t = ex["input"]
        gold = (str(t["raw_output"]), str(t[target]) if target else "")
        by_prompt.setdefault(t["raw_input"], set()).add(gold)
    return {prompt: golds for prompt, golds in by_prompt.items() if len(golds) > 1}


def _assert_coherent_prompts(examples: list, task: Task, label: str) -> None:
    """Raise if any prompt in ``examples`` maps to more than one gold label.

    This is the dataset-incoherence guard for issue #180: when a variable that
    varies across the dataset is absent from the prompt template, distinct
    causal-model inputs render to the same prompt but carry different gold
    labels. The model cannot solve such a prompt, and dedup would silently keep
    one label and discard the rest, so we fail loud instead.
    """
    conflicts = _conflicting_prompt_groups(examples, task.intervention_variable)
    if not conflicts:
        return
    model = task.causal_model
    bad_vars = sorted(_swept_input_vars(examples, model) - _prompt_input_vars(model))
    prompt, golds = next(iter(conflicts.items()))
    raise ValueError(
        f"Incoherent {label} dataset for task {task.name!r}: "
        f"{len(conflicts)} prompt(s) each map to multiple distinct gold labels. "
        f"E.g. prompt {prompt!r} maps to {len(golds)} gold labels "
        f"{sorted(golds)}. This means a variable that varies across the dataset "
        "is absent from the prompt template, so distinct inputs render to the "
        "same prompt and dedup would silently merge their differing labels. "
        f"Swept but absent from the prompt: {bad_vars or '<none detected>'}. "
        "Fix the task's raw_input template to reference these variables (or stop "
        "sweeping them)."
    )


def _deduplicate_examples(
    examples: list,
    *,
    label: str,
    task: Task,
    warn_threshold: float,
) -> list:
    """Deduplicate by ``raw_input`` after guarding against incoherent collapse.

    Raises if any prompt maps to multiple gold labels (see
    ``_assert_coherent_prompts``). Otherwise dedup is information-preserving:
    collapsed items share an identical gold label. A large *benign* drop still
    often signals a sweep wider than the prompt-relevant input space, so when the
    dropped fraction exceeds ``warn_threshold`` we log a warning naming the
    prompt-dependent and swept variables.
    """
    if not examples:
        return examples
    _assert_coherent_prompts(examples, task, label)
    before = len(examples)
    deduped = _deduplicate_by_input(examples)
    after = len(deduped)
    if after < before:
        logger.info("Deduplicated %s: %d -> %d unique prompts", label, before, after)
        if (before - after) / before > warn_threshold:
            model = task.causal_model
            logger.warning(
                "Deduplicated %s dropped %.0f%% of items (%d -> %d); collapsed "
                "items shared identical gold labels, but a drop this large often "
                "means the sweep is wider than the prompt. Prompt depends on "
                "inputs %s; variables swept: %s.",
                label,
                100 * (before - after) / before,
                before,
                after,
                sorted(_prompt_input_vars(model)),
                sorted(_swept_input_vars(examples, model)),
            )
    return deduped


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
    dedup_warn_threshold: float = 0.2,
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

    Raises ``ValueError`` when the generated dataset is *incoherent* — i.e. a
    variable that varies across the dataset is absent from the prompt template,
    so distinct inputs render to the same prompt but carry different gold labels
    (issue #180). Such a prompt is unsolvable and dedup would silently merge the
    differing labels, so we fail loud rather than degrade silently. A large but
    *benign* dedup drop (collapsed items share an identical gold label) instead
    logs a warning once the dropped fraction exceeds ``dedup_warn_threshold``
    (default 0.2).
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
        # enumerate_all keeps every unique input (no dedup), but the same
        # prompt->multiple-gold incoherence still applies if the template
        # ignores a swept variable, so guard here too.
        _assert_coherent_prompts(train, task, "enumerated")
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
            train = _deduplicate_examples(
                train, label="train", task=task, warn_threshold=dedup_warn_threshold
            )
        test = (
            _generate_balanced(model, n_test, seed + 1, iv, values)
            if n_test > 0
            else []
        )
        if n_test > 0 and deduplicate:
            test = _deduplicate_examples(
                test, label="test", task=task, warn_threshold=dedup_warn_threshold
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
            train = _deduplicate_examples(
                train, label="train", task=task, warn_threshold=dedup_warn_threshold
            )
        if n_test > 0:
            test = _generate_single_variable_dataset(
                model,
                n_test,
                seed + 1,
                resample_variable,
            )
            if deduplicate:
                test = _deduplicate_examples(
                    test, label="test", task=task, warn_threshold=dedup_warn_threshold
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

    train = cf_mod.generate_dataset(model, n_train, seed)
    if deduplicate:
        train = _deduplicate_examples(
            train, label="train", task=task, warn_threshold=dedup_warn_threshold
        )

    if n_test > 0:
        test = cf_mod.generate_dataset(model, n_test, seed + 1)
        if deduplicate:
            test = _deduplicate_examples(
                test, label="test", task=task, warn_threshold=dedup_warn_threshold
            )
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
    """Score-token ids for the intervention variable's distinct output forms.

    Returns ``(token_ids, n_tokens)``, or ``(None, None)`` when the task has no
    intervention variable or declares no ``output_tokens`` for it. The unified
    resolver derives the ids from the causal model's explicit per-value form map
    (``output_tokens``), with dedup falling out of shared form groups (#291).
    """
    var = task.intervention_variable
    output_tokens = task.causal_model.output_tokens
    if not (var and output_tokens and output_tokens.get(var)):
        return None, None

    from causalab.methods.output_tokens import resolve_score_token_ids

    token_ids = resolve_score_token_ids(pipeline.tokenizer, output_tokens[var])
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


def resolve_intervention_metric(intervention_metric: str, *, checker):
    """Resolve intervention metric by name.

    Returns ``(string_metric_fn, comparison_fn)``.  The string metric is the
    task's own ``checker`` (pass ``task.checker``) — the single match authority,
    with no lenient-containment default (#167).  All comparison functions follow
    the **higher-is-better** convention; divergence metrics (KL, Hellinger) are
    negated so values closer to zero (= less divergence) rank higher.
    """
    from causalab.methods.metric import DISTRIBUTION_COMPARISONS, as_label_checker

    name = intervention_metric
    string_metric = as_label_checker(checker)

    if name in _STRING_METRICS:
        return string_metric, _argmax_accuracy
    if name in DISTRIBUTION_COMPARISONS:
        raw_fn = DISTRIBUTION_COMPARISONS[name]

        def _negated(ref, pred, _fn=raw_fn):
            return -_fn(ref, pred)

        return string_metric, _negated
    raise ValueError(
        f"Unknown intervention_metric: {name!r}. "
        f"Available: {', '.join(sorted(set(_STRING_METRICS) | set(DISTRIBUTION_COMPARISONS)))}"
    )


def resolve_interchange_metric(
    name: str, *, score_token_ids, checker
) -> InterchangeMetric:
    """Resolve a config name to an ``InterchangeMetric`` for ``run_layer_scan``.

    Used by ``locate`` pairwise mode.  Two families:

    - **Causal-label** — compare the patched output string to the causal model's
      expected counterfactual label (``needs_causal_expected``).  Higher = more
      interchange-intervention accuracy.  This is the default.  The string match
      is the task's own ``checker`` (pass ``task.checker``) — the single match
      authority, no lenient/strict default (#167).  The legacy ``causal_label`` /
      ``exact`` / ``string_match`` names are now aliases: each task's checker
      decides its own semantics.
    - **Distribution-shift** — compare the patched output distribution to the base
      (pre-intervention) distribution (``needs_original_output``).  Higher = the
      cell moved the output more, i.e. the variable is more strongly encoded there.

    ``score_token_ids`` is required to project logits onto answer classes for the
    distribution-shift family.  Legacy ``"kl"`` / ``"hellinger"`` (which select a
    class-average comparison for ``centroid`` mode) map to the base-vs-patched
    variants here so distribution-metric task configs still run under the default
    ``pairwise`` mode; use ``mode: centroid`` for the class-average comparison.
    """
    from causalab.methods.metric import (
        as_label_checker,
        make_causal_metric,
        make_distribution_shift_metric,
        kl_divergence,
        hellinger_distance,
    )

    # Causal-model label references — scored by the task's own checker.
    if name in ("causal_label", "exact", "string_match"):
        return make_causal_metric(checker=as_label_checker(checker))

    # Base-vs-patched distribution shift.
    if name in ("output_shift", "output_shift_kl", "kl"):
        return make_distribution_shift_metric(score_token_ids, kl_divergence)
    if name in ("output_shift_hellinger", "hellinger"):
        return make_distribution_shift_metric(score_token_ids, hellinger_distance)

    raise ValueError(
        f"Unknown intervention_metric for pairwise mode: {name!r}. Available: "
        "causal_label, exact, string_match, output_shift (= output_shift_kl), "
        "output_shift_hellinger. Legacy 'kl'/'hellinger' map to the base-vs-patched "
        "variants; use mode: centroid for the class-average comparison."
    )
