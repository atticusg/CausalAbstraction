"""Causal models for the natural_domains_arithmetic factory task.

Unified implementation for weekdays, months, and hours domains.
All share the DAG: (entity, number) → result → raw_output.
"""

from __future__ import annotations

from typing import Any, Callable

from causalab.causal.causal_model import CausalModel, build_output_tokens
from causalab.causal.trace import CausalTrace, Mechanism, input_var
from causalab.tasks.random_words import get_random_words

from .config import NaturalDomainConfig


# ---------------------------------------------------------------------------
# Factory: create causal model from config
# ---------------------------------------------------------------------------


def create_causal_model(config: NaturalDomainConfig) -> CausalModel:
    """Create a causal model for a natural-domain arithmetic task.

    Args:
        config: NaturalDomainConfig specifying domain, entities, etc.

    Returns:
        CausalModel with variables: entity, number, result, raw_input, raw_output.
    """
    entities = config.entities
    numbers = config.numbers
    number_to_int = config.number_to_int
    result_entities = (
        config.result_entities if config.result_entities is not None else entities
    )
    template = config.template
    output_prefix = config.output_prefix

    entity_to_index = {e: i for i, e in enumerate(entities)}
    templates = template if isinstance(template, list) else [template]
    multi_template = isinstance(template, list)

    if config.compute_result is not None:
        _custom_compute = config.compute_result
        _cfg = config

        def compute_result(t: CausalTrace) -> str:
            return _custom_compute(t["entity"], t["number"], _cfg)
    else:
        modulus = config.modulus
        assert modulus is not None, "cyclic domains require modulus"

        def compute_result(t: CausalTrace) -> str:
            idx = (entity_to_index[t["entity"]] + number_to_int[t["number"]]) % modulus
            return result_entities[idx]

    if multi_template:

        def fill_template(t: CausalTrace) -> str:
            return t["template"].format(entity=t["entity"], number=t["number"])
    else:

        def fill_template(t: CausalTrace) -> str:
            return templates[0].format(entity=t["entity"], number=t["number"])

    # When number_groups is configured with >1 bin, result becomes a tuple
    # (entity_result, group_index) so centroid computation gets 2D structure.
    has_groups = bool(config.number_groups) and len(config.number_groups or []) > 1
    number_to_group: dict[str, int] = {}
    if has_groups:
        # Narrow the Optional via assert — has_groups already implies non-None.
        assert config.number_groups is not None
        bins = config.number_groups
        for n in numbers:
            n_int = number_to_int[n]
            for i, (lo, hi) in enumerate(bins):
                if lo <= n_int <= hi:
                    number_to_group[n] = i
                    break
        n_groups = len(bins)
        # Result values: all (entity_result, group) combos
        result_values = [(re, g) for re in result_entities for g in range(n_groups)]
    else:
        result_values = list(result_entities)

    values: dict[str, list | None] = {
        "entity": entities,
        "number": numbers,
        "result": result_values,
        "raw_input": None,
        "raw_output": None,
    }
    if multi_template:
        values["template"] = templates

    raw_input_parents = ["entity", "number"]
    if multi_template:
        raw_input_parents.append("template")

    if has_groups:
        _compute_result_base = compute_result  # save the base compute

        def compute_result_grouped(t: CausalTrace) -> tuple:
            if callable(_compute_result_base):
                # Custom compute
                entity_result = _compute_result_base(t)
            else:
                entity_result = _compute_result_base(t)
            group = number_to_group[t["number"]]
            return (entity_result, group)

        mechanisms = {
            "entity": input_var(entities),
            "number": input_var(numbers),
            "result": Mechanism(
                parents=["entity", "number"],
                compute=compute_result_grouped,
            ),
            "raw_input": Mechanism(
                parents=raw_input_parents,
                compute=fill_template,
            ),
            "raw_output": Mechanism(
                parents=["result"],
                compute=lambda t: output_prefix + t["result"][0],
            ),
        }
    else:
        mechanisms = {
            "entity": input_var(entities),
            "number": input_var(numbers),
            "result": Mechanism(
                parents=["entity", "number"],
                compute=compute_result,
            ),
            "raw_input": Mechanism(
                parents=raw_input_parents,
                compute=fill_template,
            ),
            "raw_output": Mechanism(
                parents=["result"],
                compute=lambda t: output_prefix + t["result"],
            ),
        }
    if multi_template:
        mechanisms["template"] = input_var(templates)

    # Build embeddings
    embeddings: dict[str, Callable[[Any], list[float]]] = {}
    if config.entity_embedding is not None:
        embeddings["entity"] = config.entity_embedding
        if has_groups:
            embeddings["result"] = lambda v, _emb=config.entity_embedding: _emb(
                v[0]
            ) + [float(v[1])]
        else:
            embeddings["result"] = config.entity_embedding
    else:
        embeddings["entity"] = lambda v, _m=entity_to_index: [float(_m[v])]
        re_to_idx = {e: i for i, e in enumerate(result_entities)}
        if has_groups:
            embeddings["result"] = lambda v, _m=re_to_idx: [
                float(_m[v[0]]),
                float(v[1]),
            ]
        else:
            embeddings["result"] = lambda v, _m=re_to_idx: [float(_m[v])]

    # Always provide number embedding
    embeddings["number"] = lambda v, _m=number_to_int: [float(_m[v])]

    # Compute periods for cyclic variables
    periods: dict[str, float] = {}
    if config.cyclic and config.modulus is not None:
        periods["entity"] = config.modulus
        has_groups = config.number_groups and len(config.number_groups) > 1
        if has_groups:
            periods["result_0"] = config.modulus
        else:
            periods["result"] = config.modulus
        if config.number_is_cyclic:
            periods["number"] = config.modulus

    # Declare the answer's surface forms per result value (#296). The answer is
    # the result entity, emitted as ``output_prefix + entity``; the case-sensitive
    # ``[" entity", "entity"]`` forms cover both BPE spacings (the grader's
    # lowercase tolerance lives in the probability path, not the declaration).
    # 1D: keyed by the entity. 2D (number_groups): keyed by the (entity, group)
    # tuple — all groups of one entity share its forms, so form-groups collapse
    # the N_entities × N_groups tuples back to N_entities score tokens. That
    # shared-form-group dedup replaces the former output_token_values override.
    if has_groups:
        output_tokens = {
            "result": {
                (re, g): build_output_tokens([re])[re]
                for re in result_entities
                for g in range(n_groups)
            }
        }
    else:
        output_tokens = {"result": build_output_tokens(result_values)}

    # For non-cyclic domains with a custom compute_result, some (entity, number)
    # pairs may produce results outside the configured result_entities (e.g.
    # alphabet "letter+N" overflowing past Z). Filter those out at the input
    # level so dataset enumeration respects the boundary.
    input_filter: Callable[[Any], bool] | None = None
    if (
        not config.cyclic
        and config.compute_result is not None
        and config.result_entities is not None
    ):
        valid_results = set(result_values)

        def _input_filter(trace, _compute=compute_result, _valid=valid_results):
            return _compute(trace) in _valid

        input_filter = _input_filter

    model = CausalModel(
        mechanisms,
        values,
        id=f"natural_domains_arithmetic_{config.domain_type}",
        embeddings=embeddings,
        periods=periods,
        output_tokens=output_tokens,
        input_filter=input_filter,
    )
    model._nda_config = config  # type: ignore[attr-defined]
    return model


# ---------------------------------------------------------------------------
# Random baseline factory
# ---------------------------------------------------------------------------


def create_random_causal_model(config: NaturalDomainConfig) -> CausalModel:
    """Create a random-word baseline model for the domain.

    Replaces entities with random words and uses cyclic modular arithmetic.
    """
    n_random = len(config.entities)
    random_entities = get_random_words(n_random)
    random_entity_to_index = {e: i for i, e in enumerate(random_entities)}

    numbers = config.numbers
    number_to_int = config.number_to_int
    template = config.template
    templates = template if isinstance(template, list) else [template]
    multi_template = isinstance(template, list)
    output_prefix = config.output_prefix

    def compute_result(t: CausalTrace) -> str:
        idx = (
            random_entity_to_index[t["entity"]] + number_to_int[t["number"]]
        ) % n_random
        return random_entities[idx]

    if multi_template:

        def fill_template(t: CausalTrace) -> str:
            return t["template"].format(entity=t["entity"], number=t["number"])
    else:

        def fill_template(t: CausalTrace) -> str:
            return templates[0].format(entity=t["entity"], number=t["number"])

    raw_input_parents = ["entity", "number"]
    if multi_template:
        raw_input_parents.append("template")

    values: dict[str, list[str] | None] = {
        "entity": random_entities,
        "number": numbers,
        "result": list(random_entities),
        "raw_input": None,
        "raw_output": None,
    }
    if multi_template:
        values["template"] = templates

    mechanisms = {
        "entity": input_var(random_entities),
        "number": input_var(numbers),
        "result": Mechanism(
            parents=["entity", "number"],
            compute=compute_result,
        ),
        "raw_input": Mechanism(
            parents=raw_input_parents,
            compute=fill_template,
        ),
        "raw_output": Mechanism(
            parents=["result"],
            compute=lambda t: output_prefix + t["result"],
        ),
    }
    if multi_template:
        mechanisms["template"] = input_var(templates)

    embeddings: dict[str, Callable[[Any], list[float]]] = {
        "entity": lambda v, _m=random_entity_to_index: [float(_m[v])],
        "result": lambda v, _m=random_entity_to_index: [float(_m[v])],
        "number": lambda v, _m=number_to_int: [float(_m[v])],
    }

    model = CausalModel(
        mechanisms,
        values,
        id=f"natural_domains_arithmetic_{config.domain_type}_random",
        embeddings=embeddings,
        # Random baseline is always 1D (no number_groups): one form group per
        # random entity, matching the migrated real model (#296).
        output_tokens={"result": build_output_tokens(list(random_entities))},
    )
    model._nda_config = config  # type: ignore[attr-defined]
    return model


# ---------------------------------------------------------------------------
# Standard exports for load_task()
# ---------------------------------------------------------------------------

CREATE_CAUSAL_MODEL = create_causal_model
CREATE_RANDOM_CAUSAL_MODEL = create_random_causal_model
TARGET_VARIABLE = "result"

# Static stubs — dynamic getters below override these in the loader
CYCLIC_VARIABLES: set[str] = set()
EMBEDDINGS: dict[str, Callable] = {}


def GET_VARIABLE_VALUES(model: CausalModel) -> dict[str, list]:
    """Derive variable values from the model."""
    return {
        "entity": model.values["entity"],
        "number": model.values["number"],
        "result": model.values["result"],
    }


def GET_CYCLIC_VARIABLES(model: CausalModel) -> set[str]:
    """Derive cyclic variables from the stored config."""
    config: NaturalDomainConfig = model._nda_config  # type: ignore[attr-defined]
    cyclic: set[str] = set()
    if config.cyclic:
        cyclic.add("entity")
        cyclic.add("result")
    if config.number_is_cyclic:
        cyclic.add("number")
    return cyclic


def GET_EMBEDDINGS(model: CausalModel) -> dict[str, Callable]:
    """Return the embeddings dict stored on the model."""
    return model.embeddings


def GET_PERIODIC_INFO(model: CausalModel) -> dict[str, int] | None:
    """Derive period info from the stored config."""
    config: NaturalDomainConfig = model._nda_config  # type: ignore[attr-defined]
    if not config.cyclic:
        return None
    info: dict[str, int] = {}
    modulus = config.modulus
    assert modulus is not None
    info["entity"] = modulus
    # When result is a tuple, extract_parameters_from_dataset expands to
    # result_0 (entity index, cyclic) and result_1 (group index, linear).
    has_groups = config.number_groups and len(config.number_groups) > 1
    if has_groups:
        info["result_0"] = modulus
        # result_1 is linear (group index) — not in periodic_info
    else:
        info["result"] = modulus
    if config.number_is_cyclic:
        info["number"] = modulus
    return info


def GET_TEMPLATE(model: CausalModel) -> str | list[str]:
    """Return the prompt template(s) from the stored config."""
    config: NaturalDomainConfig = model._nda_config  # type: ignore[attr-defined]
    return config.template
