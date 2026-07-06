"""Capture-point provenance: gap registry, runtime report, coverage table.

Every activation capture or intervention point this package uses is labeled
with its *mechanism*:

* ``pyvene-named``    a component in pyvene's per-family mapping
* ``pyvene-path``     pyvene's dotted module-path fallback (still pyvene —
                      no hooks of ours — but not covered by the named
                      vocabulary, so it carries a reason code)
* ``raw-hook``        a torch hook of our own (K/V module only); requires a
                      gap-registry entry, enforced by the hygiene test
* ``direct-module-call``  the engine invoking a module on cached tensors
                      (receiver re-evaluation, tail); no hook involved
* ``unsupported``     not available for this family

Reason codes for anything not pyvene-named:

* ``mapping-lacks-entry``    pyvene's mapping has no unit for this point
* ``unit-is-wrong-quantity`` pyvene has a similarly-named unit but it
                             captures the wrong tensor (e.g. ``mlp_activation``
                             is the gate activation, not the gated product a
                             neuron sender needs)
* ``family-unmapped``        pyvene does not map this model family at all
* ``no-module-boundary``     the quantity is not any module's input/output
                             (e.g. pre-softmax attention scores)

The registry exists for drift detection as much as for review: the coverage
table is regenerated in tests against the installed pyvene, so a pin bump
that adds or moves units surfaces as a table diff instead of a silent
behavior change.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from typing import Literal

logger = logging.getLogger(__name__)

Mechanism = Literal[
    "pyvene-named", "pyvene-path", "raw-hook", "direct-module-call", "unsupported"
]
ReasonCode = Literal[
    "mapping-lacks-entry",
    "unit-is-wrong-quantity",
    "family-unmapped",
    "no-module-boundary",
]

__all__ = [
    "GAP_REGISTRY",
    "GapEntry",
    "CapturePoint",
    "capture_provenance",
    "coverage_table",
    "pyvene_pin",
    "register_gap",
]


@dataclass(frozen=True)
class GapEntry:
    site: str  # "<module>:<symbol>" of the code site
    component: str
    model_families: tuple[str, ...]
    reason: ReasonCode
    note: str = ""


GAP_REGISTRY: dict[str, GapEntry] = {}


def register_gap(entry: GapEntry) -> None:
    GAP_REGISTRY[entry.site] = entry


# ---------------------------------------------------------------------------
# Registrations. Import-time, so the hygiene test can cross-check code sites
# against the registry without executing any model code.
# ---------------------------------------------------------------------------

# The package currently has ZERO raw hooks (pyvene's gpt2 mapping splits the
# fused c_attn itself, so even K/V capture is pyvene-named). The registry
# still records the one no-module-boundary quantity, which is *derived*
# arithmetic rather than a capture:
register_gap(
    GapEntry(
        site="kv:scores",
        component="pre-softmax masked attention scores",
        model_families=("gpt2",),
        reason="no-module-boundary",
        note=(
            "scores are computed inline in HF attention (no module I/O); "
            "derived analytically from captured q/k, never hooked or "
            "intervened on"
        ),
    )
)

# pyvene-path (dotted fallback) captures — pyvene-native, registered for
# visibility and for the coverage table:
register_gap(
    GapEntry(
        site="cache:component_neuron_values",
        component="mlp down-projection input (neuron values)",
        model_families=("gpt2", "gpt_neox", "llama", "gemma2"),
        reason="unit-is-wrong-quantity",
        note=(
            "pyvene's mlp_activation captures act_fn output, which for gated "
            "MLPs (Llama/Gemma) is the gate activation alone, not the "
            "act(gate)*up product a neuron sender needs; the down-projection "
            "input is the right quantity on every family."
        ),
    )
)
register_gap(
    GapEntry(
        site="reference:component_mlp_pre_norm_input",
        component="MLP branch pre-norm input",
        model_families=("gpt2", "gpt_neox", "llama", "gemma2"),
        reason="mapping-lacks-entry",
        note="no named unit for norm inputs; used for excluded-edge cancellation",
    )
)
register_gap(
    GapEntry(
        site="reference:component_mlp_trunk_output",
        component="MLP branch trunk contribution (post-norm output where present)",
        model_families=("gemma2",),
        reason="mapping-lacks-entry",
        note=(
            "pyvene's mlp_output is the MLP module output, which on Gemma-2 "
            "is pre-post-norm; the trunk contribution is the post-norm output"
        ),
    )
)
register_gap(
    GapEntry(
        site="reference:component_final_norm_input",
        component="final norm input",
        model_families=("gpt2", "gpt_neox", "llama", "gemma2"),
        reason="mapping-lacks-entry",
        note="no named unit; used for direct-edge cancellation at the logits",
    )
)


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
    reason: ReasonCode | None = None
    gap_site: str | None = None


def capture_provenance(desc) -> list[CapturePoint]:
    """Provenance of every capture/intervention point the engine, cache, and
    reference twin use for this descriptor's family."""
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
            "unit-is-wrong-quantity",
            "cache:component_neuron_values",
        ),
        CapturePoint(
            "receiver MLP re-evaluation",
            "mlp branch modules on cached tensors",
            "direct-module-call",
        ),
        CapturePoint(
            "tail (final norm + LM head + softcapping)",
            "final norm / LM head modules on cached tensors",
            "direct-module-call",
        ),
        # reference twin intervention points
        CapturePoint(
            "twin: sender/freeze substitution",
            "head_attention_value_output / attention_value_output / mlp_output",
            "pyvene-named",
        ),
        CapturePoint(
            "twin: pre-norm input cancellation",
            desc.component_mlp_pre_norm_input(0).replace("[0]", "[i]"),
            "pyvene-path",
            "mapping-lacks-entry",
            "reference:component_mlp_pre_norm_input",
        ),
        CapturePoint(
            "twin: receiver trunk-output recording",
            desc.component_mlp_trunk_output(0).replace("[0]", "[i]"),
            "pyvene-path",
            "mapping-lacks-entry" if desc.spec.mlp_post_norm else None,
            "reference:component_mlp_trunk_output" if desc.spec.mlp_post_norm else None,
        ),
        CapturePoint(
            "twin: final-norm input cancellation",
            desc.component_final_norm_input(),
            "pyvene-path",
            "mapping-lacks-entry",
            "reference:component_final_norm_input",
        ),
    ]
    if desc.attention_style == "fused-qkv-absolute":
        points.append(
            CapturePoint(
                "K/V detail (gated module)",
                "(head_)query/key/value_output; scores derived from cached q/k",
                "pyvene-named",
            )
        )
        points.append(
            CapturePoint(
                "K/V detail: pre-softmax scores",
                "derived arithmetic on captured q/k (never hooked)",
                "direct-module-call",
                "no-module-boundary",
                "kv:scores",
            )
        )
    else:
        points.append(
            CapturePoint(
                "K/V detail (gated module)",
                "n/a for this attention style",
                "unsupported",
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
    "head_key_output",
    "head_query_output",
    "head_value_output",
)
_LOGICAL_POINTS = (
    ("neuron values (down-proj input)", "pyvene-path", "unit-is-wrong-quantity"),
    ("MLP pre-norm input", "pyvene-path", "mapping-lacks-entry"),
    ("final norm input", "pyvene-path", "mapping-lacks-entry"),
    ("pre-softmax attention scores", "derived-from-cache", "no-module-boundary"),
)


def coverage_table() -> dict:
    """family × component → mechanism, generated from the installed pyvene.

    Regenerated in tests and diffed against the committed artifact so a
    pyvene pin bump surfaces as a table diff.
    """
    import importlib

    table: dict[str, dict[str, str]] = {}
    for family in _COVERAGE_FAMILIES:
        mod = importlib.import_module(f"pyvene.models.{family}.modelings_intervenable_{family}")
        mapping = getattr(mod, f"{family}_lm_type_to_module_mapping", None) or getattr(
            mod, f"{family}_type_to_module_mapping"
        )
        row: dict[str, str] = {}
        for comp in _NAMED_COMPONENTS:
            row[comp] = "pyvene-named" if comp in mapping else "unsupported"
        for name, mech, reason in _LOGICAL_POINTS:
            if name == "pre-softmax attention scores" and family != "gpt2":
                row[name] = "unsupported (gated)"
            else:
                row[name] = f"{mech} ({reason})"
        table[family] = row
    return {"pyvene_pin": pyvene_pin(), "families": table}
