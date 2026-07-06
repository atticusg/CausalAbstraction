"""Capability checking, capture provenance, and pyvene coverage.

causalab's contract is that a model works iff its family is in pyvene's
mapping — every intervention routes through ``IntervenableModel``. This
method keeps that contract exactly: there are **no raw torch hooks anywhere**
(grep-checkable), and every capture point is either a pyvene *named*
component or a pyvene *dotted module path* (pyvene's own fallback resolution;
still an ``IntervenableModel`` intervention, not a hook of ours).

Three surfaces:

* :func:`check_capability` — at engine construction, verify that every
  pyvene unit the requested operation needs exists in the family's mapping;
  otherwise raise :class:`UnsupportedArchitectureError` naming the missing
  units and the operation that needs them. Unsupported means unsupported,
  same as the rest of the library.
* :func:`capture_provenance` — document which pyvene units an engine uses
  (named vs dotted-path); exposed as ``engine.provenance`` and written into
  validation results JSONs.
* :func:`coverage_table` — supported/unsupported per family × component,
  generated from the installed pyvene. Regenerated in tests and diffed
  against the committed artifact, so a pyvene pin bump surfaces as a table
  diff instead of a silent behavior change.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from typing import Literal

logger = logging.getLogger(__name__)

Mechanism = Literal["pyvene-named", "pyvene-path", "direct-module-call"]

__all__ = [
    "CapturePoint",
    "UnsupportedArchitectureError",
    "capture_provenance",
    "check_capability",
    "coverage_table",
    "pyvene_pin",
]


class UnsupportedArchitectureError(NotImplementedError):
    """The model family lacks pyvene units this operation needs."""


# pyvene named units each operation requires, beyond family membership.
REQUIRED_UNITS: dict[str, tuple[str, ...]] = {
    "path-patching cache collection": (
        "attention_value_output",
        "mlp_output",
        "block_input",
        "block_output",
    ),
    "reference-twin interventions": (
        "attention_value_output",
        "head_attention_value_output",
        "mlp_output",
        "block_input",
    ),
    "K/V attention-detail collection": (
        "query_output",
        "key_output",
        "value_output",
    ),
}


def _family_mapping(model) -> dict | None:
    """The installed pyvene's component mapping for this model's class."""
    from pyvene.models.intervenable_modelcard import type_to_module_mapping

    return type_to_module_mapping.get(type(model))


def check_capability(model, operations: list[str]) -> dict[str, list[str]]:
    """Verify pyvene covers ``operations`` for this model; raise otherwise.

    Returns {operation: [units used]} for provenance on success.
    """
    mapping = _family_mapping(model)
    if mapping is None:
        raise UnsupportedArchitectureError(
            f"pyvene has no component mapping for model class "
            f"{type(model).__name__} (pyvene pin {pyvene_pin()}); causalab "
            f"methods require the model family to be in pyvene's mapping. "
            f"Operations requested: {operations}."
        )
    used: dict[str, list[str]] = {}
    problems: list[str] = []
    for op in operations:
        required = REQUIRED_UNITS[op]
        missing = [u for u in required if u not in mapping]
        if missing:
            problems.append(
                f"{op!r} needs pyvene unit(s) {missing} that "
                f"{type(model).__name__}'s mapping does not define"
            )
        used[op] = list(required)
    if problems:
        raise UnsupportedArchitectureError(
            "unsupported architecture for path patching (pyvene pin "
            f"{pyvene_pin()}):\n- " + "\n- ".join(problems)
        )
    return used


def pyvene_pin() -> str:
    """The installed pyvene's VCS commit (from installation metadata)."""
    try:
        from importlib.metadata import distribution

        dist = distribution("pyvene")
        direct = dist.read_text("direct_url.json")
        if direct:
            info = json.loads(direct)
            commit = info.get("vcs_info", {}).get("commit_id", "")
            if commit:
                return commit[:12]
        return f"pypi:{dist.version}"
    except Exception:  # noqa: BLE001 - purely diagnostic
        return "unknown"


@dataclass(frozen=True)
class CapturePoint:
    name: str
    component: str
    mechanism: Mechanism
    note: str = ""


def capture_provenance(desc) -> list[CapturePoint]:
    """The pyvene units and module calls an engine for this family uses."""
    points = [
        CapturePoint("head values (z)", desc.component_head_values(), "pyvene-named"),
        CapturePoint("mlp branch output", desc.component_mlp_branch(), "pyvene-named"),
        CapturePoint("block output", desc.component_block_output(), "pyvene-named"),
        CapturePoint(
            "embed contribution", desc.component_block_input(), "pyvene-named"
        ),
        CapturePoint(
            "neuron values",
            desc.component_neuron_values(0).replace("[0]", "[i]"),
            "pyvene-path",
            "pyvene's mlp_activation is the gate activation; a neuron sender "
            "needs the down-projection input (the gated product on "
            "Llama/Gemma), addressed by module path",
        ),
        CapturePoint(
            "receiver MLP re-evaluation",
            "mlp branch modules invoked on cached tensors",
            "direct-module-call",
        ),
        CapturePoint(
            "tail (final norm + LM head + softcapping)",
            "final norm / LM head modules invoked on cached tensors",
            "direct-module-call",
        ),
        CapturePoint(
            "twin: sender/freeze substitution",
            "head_attention_value_output / attention_value_output / mlp_output",
            "pyvene-named",
        ),
        CapturePoint(
            "twin: pre-norm input cancellation",
            desc.component_mlp_pre_norm_input(0).replace("[0]", "[i]"),
            "pyvene-path",
            "norm inputs have no named unit",
        ),
        CapturePoint(
            "twin: receiver trunk-output recording",
            desc.component_mlp_trunk_output(0).replace("[0]", "[i]"),
            "pyvene-path",
            "the trunk contribution is the branch post-norm's output on "
            "Gemma-2; the MLP module's output elsewhere",
        ),
        CapturePoint(
            "twin: final-norm input cancellation",
            desc.component_final_norm_input(),
            "pyvene-path",
            "norm inputs have no named unit",
        ),
    ]
    if desc.attention_style == "fused-qkv-absolute":
        points.append(
            CapturePoint(
                "K/V detail (gated module)",
                "(head_)query/key/value_output",
                "pyvene-named",
                "pre-softmax scores are derived arithmetic on the captured "
                "q/k (no module boundary exists for scores; none is needed)",
            )
        )
    return points


def log_provenance(desc) -> list[dict]:
    points = [asdict(p) for p in capture_provenance(desc)]
    logger.info(
        "path_patching capture provenance (%s, pyvene %s): %s",
        desc.model_type,
        pyvene_pin(),
        json.dumps(points, indent=None),
    )
    return points


# ---------------------------------------------------------------------------
# Coverage table
# ---------------------------------------------------------------------------

_COVERAGE_FAMILIES = ("gpt2", "gpt_neox", "llama", "gemma2")
_NAMED_COMPONENTS = (
    "attention_value_output",
    "head_attention_value_output",
    "mlp_output",
    "mlp_activation",
    "block_input",
    "block_output",
    "query_output",
    "key_output",
    "value_output",
    "head_query_output",
    "head_key_output",
    "head_value_output",
)


def coverage_table() -> dict:
    """family × pyvene component → supported/unsupported, from the installed
    pyvene. The drift-detection artifact for pin bumps."""
    import importlib

    table: dict[str, dict[str, str]] = {}
    for family in _COVERAGE_FAMILIES:
        mod = importlib.import_module(
            f"pyvene.models.{family}.modelings_intervenable_{family}"
        )
        mapping = getattr(mod, f"{family}_lm_type_to_module_mapping", None) or getattr(
            mod, f"{family}_type_to_module_mapping"
        )
        table[family] = {
            comp: "supported" if comp in mapping else "unsupported"
            for comp in _NAMED_COMPONENTS
        }
    return {"pyvene_pin": pyvene_pin(), "families": table}
