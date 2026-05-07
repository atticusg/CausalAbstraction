"""Unified task interface and convention-based loader.

All pipeline steps consume the Task dataclass. Tasks are discovered via
importlib from causalab.tasks.<name>.causal_models and validated against
required exports. See .claude/plans/task-convention.md for the full spec.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from functools import cached_property
from types import ModuleType
from typing import Callable

from causalab.causal.causal_model import CausalModel


@dataclass
class Task:
    """Unified task interface. All pipeline steps consume this.

    Variable properties (periods, embeddings, output_token_values) live on the
    CausalModel. The Task adds experiment-specific concerns: which variable to
    intervene on, how to tokenize output values, and the prompt template.
    """

    # --- Required fields ---
    name: str
    causal_model: CausalModel
    intervention_variable: str | None = None

    # --- Experiment-specific ---
    result_token_pattern: Callable[[str], list[str]] = field(
        default=lambda v: [f" {v}"]
    )
    template: str | list[str] | None = None
    validate: Callable | None = None
    predict_class: Callable | None = field(default=None, repr=False)
    class_token_ids: Callable | None = field(default=None, repr=False)
    _example_to_class_override: Callable | None = field(default=None, repr=False)

    # --- Derived from CausalModel ---

    @property
    def intervention_values(self) -> list:
        """Values of the intervention variable (e.g. 28 (weekday, group) tuples)."""
        if self.intervention_variable:
            return self.causal_model.values.get(self.intervention_variable, [])
        return []

    @property
    def output_token_values(self) -> list | None:
        """Deduplicated output token values (e.g. 7 weekday names for weekdays_2d).

        None when every intervention value maps to a unique output token.
        """
        otv = self.causal_model.output_token_values
        if otv and self.intervention_variable and self.intervention_variable in otv:
            return otv[self.intervention_variable]
        return None

    @property
    def is_cyclic(self) -> bool:
        """Whether the intervention variable is cyclic."""
        return bool(
            self.intervention_variable
            and self.intervention_variable in self.causal_model.periods
        )

    def intervention_value_index(self, ex) -> int:
        """Map an example to its index in intervention_values."""
        if self._example_to_class_override is not None:
            return self._example_to_class_override(ex)
        val = ex["input"][self.intervention_variable]
        if isinstance(val, (list, tuple)):
            val = tuple(val)
        return self._value_to_idx[val]

    @cached_property
    def _value_to_idx(self) -> dict:
        return {
            (tuple(v) if isinstance(v, (list, tuple)) else v): i
            for i, v in enumerate(self.intervention_values)
        }

    def create_token_positions(self, pipeline):
        """Create token positions, passing the task's template automatically."""
        tp_mod = load_task_token_positions(self.name)
        if isinstance(self.template, list):
            return tp_mod.create_token_positions(pipeline, templates=self.template)
        return tp_mod.create_token_positions(pipeline, template=self.template)


# ---------------------------------------------------------------------------
# Convention-based loader
# ---------------------------------------------------------------------------


def _require(mod: ModuleType, name: str, task_name: str):
    """Raise clear error if required export is missing."""
    if not hasattr(mod, name):
        raise ValueError(
            f"Task '{task_name}' is missing required export '{name}' "
            f"in causal_models.py. See task-convention.md for required exports."
        )
    return getattr(mod, name)


def _optional(mod: ModuleType, name: str, default=None):
    return getattr(mod, name, default)


def load_task(
    task_name: str,
    task_cfg: dict | None = None,
    random: bool = False,
) -> Task:
    """Load a task by convention from its causal_models module.

    Args:
        task_name: Module name under causalab.tasks (e.g., "weekdays", "graph_walk")
        task_cfg: Config dict for factory tasks (passed to CREATE_CAUSAL_MODEL)
        random: If True, use RANDOM_CAUSAL_MODEL/RANDOM_VARIABLE_VALUES exports
    """
    mod = importlib.import_module(f"causalab.tasks.{task_name}.causal_models")

    # --- Model: singleton or factory ---
    is_factory = hasattr(mod, "CREATE_CAUSAL_MODEL")
    is_singleton = hasattr(mod, "CAUSAL_MODEL")

    if not is_factory and not is_singleton:
        raise ValueError(
            f"Task '{task_name}' must export either CAUSAL_MODEL (singleton) "
            f"or CREATE_CAUSAL_MODEL (factory) in causal_models.py."
        )

    if is_factory:
        if task_cfg is None:
            raise ValueError(
                f"Task '{task_name}' is a factory task — task_cfg is required."
            )
        causal_model = mod.CREATE_CAUSAL_MODEL(task_cfg)
    else:
        causal_model = mod.CAUSAL_MODEL

    # --- Random baseline override ---
    if random:
        if hasattr(mod, "CREATE_RANDOM_CAUSAL_MODEL") and task_cfg is not None:
            causal_model = mod.CREATE_RANDOM_CAUSAL_MODEL(task_cfg)
        elif hasattr(mod, "RANDOM_CAUSAL_MODEL"):
            causal_model = mod.RANDOM_CAUSAL_MODEL
        else:
            raise ValueError(
                f"Task '{task_name}' does not support random baselines "
                f"(no RANDOM_CAUSAL_MODEL or CREATE_RANDOM_CAUSAL_MODEL export)."
            )

    # --- Ensure CausalModel has embeddings ---
    if not causal_model.embeddings:
        if hasattr(mod, "GET_EMBEDDINGS"):
            causal_model.embeddings = mod.GET_EMBEDDINGS(causal_model)
        elif hasattr(mod, "EMBEDDINGS"):
            causal_model.embeddings = mod.EMBEDDINGS

    # --- Ensure CausalModel has periods ---
    if not causal_model.periods:
        periodic_info = _optional(mod, "PERIODIC_INFO", {})
        if hasattr(mod, "GET_PERIODIC_INFO"):
            periodic_info = mod.GET_PERIODIC_INFO(causal_model) or {}
        if periodic_info:
            causal_model.periods = periodic_info

    # --- Ensure CausalModel has output_token_values ---
    if causal_model.output_token_values is None and hasattr(
        mod, "GET_OUTPUT_TOKEN_VALUES"
    ):
        causal_model.output_token_values = mod.GET_OUTPUT_TOKEN_VALUES(causal_model)

    intervention_variable = _optional(mod, "TARGET_VARIABLE")
    example_to_class_override = _optional(mod, "EXAMPLE_TO_CLASS")
    result_token_pattern = _optional(mod, "RESULT_TOKEN_PATTERN", lambda v: [f" {v}"])
    if hasattr(mod, "GET_RESULT_TOKEN_PATTERN"):
        result_token_pattern = mod.GET_RESULT_TOKEN_PATTERN(causal_model)

    return Task(
        name=task_name,
        causal_model=causal_model,
        intervention_variable=intervention_variable,
        template=(
            mod.GET_TEMPLATE(causal_model)
            if hasattr(mod, "GET_TEMPLATE")
            else _optional(mod, "TEMPLATE")
        ),
        result_token_pattern=result_token_pattern,
        validate=_optional(mod, "VALIDATE"),
        predict_class=_optional(mod, "PREDICT_CLASS"),
        class_token_ids=_optional(mod, "CLASS_TOKEN_IDS"),
        _example_to_class_override=example_to_class_override,
    )


# ---------------------------------------------------------------------------
# Module loaders
# ---------------------------------------------------------------------------


def load_task_counterfactuals(task_name: str) -> ModuleType:
    """Import a task's counterfactuals module.

    Returns the module so callers can access generate_dataset(model, n, seed).
    """
    try:
        return importlib.import_module(f"causalab.tasks.{task_name}.counterfactuals")
    except ImportError:
        raise ImportError(f"Task '{task_name}' has no counterfactuals.py module.")


def load_task_token_positions(task_name: str) -> ModuleType:
    """Import a task's token_positions module.

    Returns the module so callers can access create_token_positions(...).
    """
    try:
        return importlib.import_module(f"causalab.tasks.{task_name}.token_positions")
    except ImportError:
        raise ImportError(f"Task '{task_name}' has no token_positions.py module.")
