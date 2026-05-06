"""Token position functions for the natural_domains_arithmetic task.

Provides positions for: last_token, number, entity.
Supports multi-template tasks: when multiple templates are provided,
token positions are resolved per-example based on trace["template"].
"""

from causalab.neural.token_positions import (
    build_token_position_factories,
    TokenPosition,
)
from causalab.neural.pipeline import LMPipeline

from typing import Any

TokenPositionSpec = dict[str, Any]


def _build_specs(template: str) -> dict[str, TokenPositionSpec]:
    """Build token position specs for a single template."""
    specs: dict[str, TokenPositionSpec] = {
        "last_token": {"type": "index", "position": -1},
        "entity": {"type": "index", "position": -1, "scope": {"variable": "entity"}},
    }
    if "{number}" in template:
        specs["number"] = {
            "type": "index",
            "position": -1,
            "scope": {"variable": "number"},
        }
    return specs


def create_token_positions(
    pipeline: LMPipeline,
    template: str | None = None,
    templates: list[str] | None = None,
) -> dict[str, TokenPosition]:
    """Create all token positions for the natural_domains_arithmetic task.

    Args:
        pipeline: The tokenizer pipeline.
        template: Single template string.
        templates: List of template strings (for multi-template tasks).

    Returns:
        Dict mapping position names to TokenPosition objects.
    """
    if templates is not None:
        return _create_multi_template_positions(pipeline, templates)

    if template is None:
        raise ValueError(
            "template is required for natural_domains_arithmetic — "
            "use task.create_token_positions(pipeline) instead of calling directly"
        )

    specs = _build_specs(template)
    factories = build_token_position_factories(specs, template)

    token_positions = {}
    for name, factory in factories.items():
        token_positions[name] = factory(pipeline)

    return token_positions


def _create_multi_template_positions(
    pipeline: LMPipeline, templates: list[str]
) -> dict[str, TokenPosition]:
    """Build dispatching token positions for multi-template tasks.

    For each position name, builds a TokenPosition per template and wraps
    them in a dispatcher that reads trace["template"] to pick the right one.
    """
    # Build per-template position objects
    per_template: dict[str, dict[str, TokenPosition]] = {}
    for tmpl in templates:
        specs = _build_specs(tmpl)
        factories = build_token_position_factories(specs, tmpl)
        per_template[tmpl] = {
            name: factory(pipeline) for name, factory in factories.items()
        }

    # Collect all position names across templates
    all_names: set[str] = set()
    for positions in per_template.values():
        all_names.update(positions.keys())

    # Build dispatching positions
    result: dict[str, TokenPosition] = {}
    for name in all_names:
        template_positions = {
            tmpl: positions[name]
            for tmpl, positions in per_template.items()
            if name in positions
        }

        def make_indexer(tp_map):
            def indexer(input_sample):
                tmpl = input_sample["template"]
                return tp_map[tmpl].index(input_sample)

            return indexer

        result[name] = TokenPosition(
            make_indexer(template_positions), pipeline, id=name
        )

    return result
