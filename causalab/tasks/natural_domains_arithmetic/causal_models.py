"""Causal models for the natural_domains_arithmetic factory task.

Unified implementation for weekdays, months, and hours domains.
All share the DAG: (entity, number) → result → raw_output.
"""

from __future__ import annotations

from typing import Any, Callable

from causalab.causal.causal_model import CausalModel
from causalab.causal.trace import CausalTrace, Mechanism, input_var

from .config import NaturalDomainConfig


_RANDOM_WORD_POOL: list[str] = [
    "apple", "stone", "blade", "crown", "pearl", "flame",
    "river", "storm", "chair", "glass", "cloud", "dream",
    "tiger", "maple", "frost", "piano", "lemon", "coral",
    "steel", "patch", "torch", "wheat", "brick", "lodge",
]


def get_random_words(n: int) -> list[str]:
    if n > len(_RANDOM_WORD_POOL):
        raise ValueError(
            f"Requested {n} random words but pool only has {len(_RANDOM_WORD_POOL)}"
        )
    return _RANDOM_WORD_POOL[:n]


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
    has_groups = config.number_groups and len(config.number_groups) > 1
    if has_groups:
        bins = config.number_groups
        number_to_group: dict[str, int] = {}
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

    # Compute output token values for 2D variants
    otv: dict[str, list] | None = None
    has_groups = config.number_groups and len(config.number_groups) > 1
    if has_groups:
        result_entities = (
            config.result_entities
            if config.result_entities is not None
            else config.entities
        )
        otv = {"result": list(result_entities)}

    # For non-cyclic domains with a custom compute_result, some (entity, number)
    # pairs may produce results outside the configured result_entities (e.g.
    # alphabet "letter+N" overflowing past Z). Filter those out at the input
    # level so dataset enumeration respects the boundary.
    input_filter = None
    if (
        not config.cyclic
        and config.compute_result is not None
        and config.result_entities is not None
    ):
        valid_results = set(result_values)

        def input_filter(trace, _compute=compute_result, _valid=valid_results):
            return _compute(trace) in _valid

    model = CausalModel(
        mechanisms,
        values,
        id=f"natural_domains_arithmetic_{config.domain_type}",
        embeddings=embeddings,
        periods=periods,
        output_token_values=otv,
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


def GET_RESULT_TOKEN_PATTERN(model: CausalModel):
    """Build result_token_pattern from config.

    Returns a function that maps a concept value to a list of tokenizable
    string variants (e.g., [" Monday", "Monday", " monday"]) so that
    downstream scoring can sum probability mass across all variants.
    """
    config: NaturalDomainConfig = model._nda_config  # type: ignore[attr-defined]
    prefix = config.output_prefix
    has_groups = config.number_groups and len(config.number_groups) > 1

    def _variants(v) -> list[str]:
        s = str(v[0] if isinstance(v, (tuple, list)) and has_groups else v)
        canonical = prefix + s
        candidates = [canonical]
        # Add common variants: with/without space, lowercase
        if canonical.startswith(" "):
            candidates.append(s)  # no space
            candidates.append(" " + s.lower())  # space + lowercase
        else:
            candidates.append(" " + s)  # add space
        if s.lower() != s:
            candidates.append(s.lower())  # no space + lowercase
        # Deduplicate while preserving order
        seen = set()
        return [c for c in candidates if not (c in seen or seen.add(c))]

    return _variants


def GET_OUTPUT_TOKEN_VALUES(model: CausalModel) -> dict[str, list] | None:
    """Return deduplicated score-level variable values for 2D variants.

    For grouped (2D) tasks, the causal variable_values has N_entities * N_groups
    tuples, but only N_entities unique output tokens. This returns the unique
    entity values so scoring collects one probability per unique token.

    Returns None for 1D tasks (no deduplication needed).
    """
    config: NaturalDomainConfig = model._nda_config  # type: ignore[attr-defined]
    has_groups = config.number_groups and len(config.number_groups) > 1
    if not has_groups:
        return None
    result_entities = (
        config.result_entities
        if config.result_entities is not None
        else config.entities
    )
    return {"result": list(result_entities)}
