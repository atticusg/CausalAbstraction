"""Canonical form and digests (spec §7).

The authored file is for humans; the canonical form is the record. It
materializes every default (optimizer betas, dtypes, the implicit
``revision``), every resolved reference (dataset content digests, artifact
file hashes), every derived width, expands sugar (int positions, the
``neural_model`` alias), sorts unordered lists (IM edit lists), and rejects
out-of-range addresses against the model's static config.

``digest = sha256(canonical bytes)`` with sorted keys and canonical floats —
:func:`canonical_bytes` is byte-stable across platforms because JSON floats
serialize via ``repr`` (shortest round-trip for IEEE doubles) and NaN/Inf
are rejected at load.

Two granularities share one implementation:

* :func:`canonicalize` on a *concrete* raw tree (a point protocol, or an
  un-swept document) materializes everything — the **point digest** is the
  provenance unit stamped on artifacts as ``produced_by``.
* On a swept document the sweep wrappers stay in place (they are the
  campaign's identity) and any derived value that depends on a swept field
  is left unmaterialized — the **document digest** names the campaign.

One deliberate interpretation, surfaced in the PR: a point protocol's
canonical bytes contain no campaign metadata (no coordinates, no parent
digest), so a point re-authored standalone digests identically to the same
point reached by expansion. Campaign linkage is recorded in run outputs,
never in the canonical bytes.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

from causalab.protocol.errors import ValidationError
from causalab.protocol.registry import ModelInfo, component_width
from causalab.protocol.resolve import ResolutionEnv
from causalab.protocol.schema import (
    FEATURIZER_SLOTS,
    LAYERLESS_COMPONENTS,
    OPTIMIZER_DEFAULTS,
    SECTION_ORDER,
    parse_document,
)

__all__ = ["canonical_bytes", "canonicalize", "digest"]


def canonical_bytes(canonical: Mapping[str, Any]) -> bytes:
    """Serialize a canonical form to its digestable bytes: sorted keys,
    minimal separators, UTF-8, no NaN/Inf, numbers normalized (an integral
    float digests as its integer — ``1`` and ``1.0`` are one value across
    the JSON and YAML surfaces; ``-0.0`` digests as ``0``)."""
    try:
        return json.dumps(
            _normalize_numbers(canonical),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except ValueError as err:
        raise ValidationError(
            15, f"canonical form holds a non-finite number: {err}"
        ) from err


def _normalize_numbers(node: Any) -> Any:
    if isinstance(node, bool):
        return node
    if isinstance(node, float):
        if node != node or node in (float("inf"), float("-inf")):
            raise ValidationError(15, "canonical form holds a non-finite number")
        if node.is_integer() and abs(node) <= 2**53:
            return int(node)
        return node
    if isinstance(node, Mapping):
        return {key: _normalize_numbers(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_normalize_numbers(item) for item in node]
    return node


def digest(canonical: Mapping[str, Any]) -> str:
    """``sha256`` of the canonical bytes, hex."""
    return hashlib.sha256(canonical_bytes(canonical)).hexdigest()


def _is_sweep(node: Any) -> bool:
    return isinstance(node, Mapping) and set(node) == {"sweep"}


def canonicalize(raw: Mapping[str, Any], env: ResolutionEnv) -> dict[str, Any]:
    """The canonical form of one raw document tree (artifact fields already
    resolved). Concrete documents materialize fully; swept documents keep
    their wrappers and skip sweep-dependent derivations."""
    doc = parse_document(raw)  # shape-checks the tree we are about to walk
    del doc  # only the raw tree is transformed; parse is the gate

    normalized: dict[str, Any] = {}
    for key, value in raw.items():
        normalized["model" if key == "neural_model" else key] = value

    model_raw = normalized["model"]
    model_key = model_raw.get("key")
    info: ModelInfo | None = None
    if isinstance(model_key, str):
        info = env.model_info(model_key)

    out: dict[str, Any] = {}
    for section in SECTION_ORDER:
        if section not in normalized:
            continue
        value = normalized[section]
        if section == "model":
            out["model"] = {
                "key": value["key"],
                "revision": value.get("revision", "main"),
            }
        elif section == "data":
            out["data"] = _canon_data(value, env)
        elif section == "positions":
            out["positions"] = {
                name: _canon_position_entry(entry) for name, entry in value.items()
            }
        elif section == "sites":
            out["sites"] = {
                name: _canon_site(name, entry, info) for name, entry in value.items()
            }
        elif section == "featurizers":
            out["featurizers"] = {
                name: _canon_featurizer(name, entry, normalized, info, env)
                for name, entry in value.items()
            }
        elif section == "reads":
            out["reads"] = {
                name: _canon_read_or_edit(entry) for name, entry in value.items()
            }
        elif section == "edits":
            out["edits"] = {
                name: _canon_read_or_edit(entry) for name, entry in value.items()
            }
        elif section == "intervened_models":
            out["intervened_models"] = {
                name: {
                    "input": entry["input"],
                    "edits": _canon_edit_list(entry["edits"]),
                }
                for name, entry in value.items()
            }
        elif section == "params":
            out["params"] = {
                name: _canon_param(entry, env) for name, entry in value.items()
            }
        elif section == "train":
            out["train"] = _canon_train(value, info, env)
        else:
            out[section] = value
    return out


def _canon_param(entry: Mapping[str, Any], env: ResolutionEnv) -> dict[str, Any]:
    """§7: "each param replaced by its content hash" — a loaded constant's
    bytes are its identity (this also makes a missing file a load error)."""
    out = dict(entry)
    file_path = out.get("file_path")
    if isinstance(file_path, str):
        out["content_digest"] = env.artifacts.file_digest(file_path)
    return out


# --------------------------------------------------------------------------- #
# per-section transforms
# --------------------------------------------------------------------------- #


def _canon_data(data: Mapping[str, Any], env: ResolutionEnv) -> dict[str, Any]:
    def one(role: Mapping[str, Any]) -> dict[str, Any]:
        ref = role["dataset"]
        stamped = dict(role)
        if isinstance(ref, str):
            stamped["digest"] = env.datasets.digest(ref)
        return stamped

    out: dict[str, Any] = {"base": one(data["base"])}
    if "source" in data:
        src = data["source"]
        out["source"] = [one(s) for s in src] if isinstance(src, list) else one(src)
    return out


def _canon_edit_list(edits: Any) -> Any:
    """IM edit lists are unordered (§6.8) — sorted concrete, and sorted
    per-value inside a sweep wrapper, so one campaign has one spelling."""
    if isinstance(edits, list):
        return sorted(edits)
    if _is_sweep(edits) and isinstance(edits["sweep"], list):
        return {
            "sweep": [sorted(v) if isinstance(v, list) else v for v in edits["sweep"]]
        }
    return edits


def _canon_position_spec(value: Any) -> Any:
    if isinstance(value, int) and not isinstance(value, bool):
        return {"index": value}  # §6.1 sugar
    return value


def _canon_position_entry(entry: Any) -> Any:
    if _is_sweep(entry):
        spec = entry["sweep"]
        if isinstance(spec, list):
            return {"sweep": [_canon_position_spec(v) for v in spec]}
        return entry
    return _canon_position_spec(entry)


def _canon_site(
    name: str, entry: Mapping[str, Any], info: ModelInfo | None
) -> dict[str, Any]:
    component = entry.get("component")
    layer = entry.get("layer")
    if (
        info is not None
        and isinstance(component, str)
        and component not in LAYERLESS_COMPONENTS
    ):
        if isinstance(layer, int) and not 0 <= layer < info.num_layers:
            raise ValidationError(
                4,
                f"site {name!r}: layer {layer} out of range for the "
                f"{info.num_layers}-layer model {info.key!r}",
                path=f"sites.{name}.layer",
            )
    head = entry.get("head")
    if info is not None and isinstance(head, int):
        # attention_value is the per-head o-projection input — query-head space
        space = info.num_heads
        if not 0 <= head < space:
            raise ValidationError(
                4,
                f"site {name!r}: head {head} out of range ({space} heads in this "
                "component's head space)",
                path=f"sites.{name}.head",
            )
    return dict(entry)


def _canon_read_or_edit(entry: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(entry)
    if "pos" in out:
        out["pos"] = (
            out["pos"]
            if isinstance(out["pos"], str)
            else _canon_position_entry(out["pos"])  # sugar expands inside sweeps too
        )
    return out


def _featurizer_sites_raw(name: str, normalized: Mapping[str, Any]) -> list[Any]:
    used: list[Any] = []
    for section in ("reads", "edits"):
        for entry in normalized.get(section, {}).values():
            ref = entry.get("featurizer")
            chain = [ref] if isinstance(ref, str) else (ref or [])
            if name in chain:
                site = entry.get("site")
                if site not in used:
                    used.append(site)
    return used


def _canon_featurizer(
    name: str,
    entry: Mapping[str, Any],
    normalized: Mapping[str, Any],
    info: ModelInfo | None,
    env: ResolutionEnv,
) -> dict[str, Any]:
    out = dict(entry)
    kind = out.setdefault("kind", "identity")
    out.setdefault("dtype", "fp32")
    if not isinstance(kind, str):
        return out  # swept kind: nothing derivable
    if isinstance(out.get("file_path"), str):
        # a loaded bundle: its params are its bytes — hash them (§7)
        out["content_digest"] = env.artifacts.file_digest(out["file_path"])
        return out
    if info is None:
        return out
    width = _derived_width(name, normalized, info)
    if width is None:
        return out
    out["width"] = width
    k = out.get("k")
    shapes: dict[str, list[int]] = {}
    if kind == "subspace" and isinstance(k, int):
        if not 0 < k <= width:
            raise ValidationError(
                4,
                f"featurizer {name!r}: k={k} exceeds the width {width}",
                path=f"featurizers.{name}.k",
            )
        shapes["weight"] = [width, k]
    elif kind == "pca" and isinstance(k, int):
        if not 0 < k <= width:
            raise ValidationError(
                4,
                f"featurizer {name!r}: k={k} exceeds the width {width}",
                path=f"featurizers.{name}.k",
            )
        shapes["weight"] = [width, k]
    elif kind == "gate":
        shapes["theta"] = [width]
    elif kind == "standardize":
        shapes["mu"] = [width]
        shapes["sigma"] = [width]
    if shapes and set(shapes) <= set(FEATURIZER_SLOTS.get(kind, ())):
        out["params"] = shapes
    return out


def _derived_width(
    name: str, normalized: Mapping[str, Any], info: ModelInfo
) -> int | None:
    """The feature width of one featurizer, from the sites its reads/edits
    use (§2.5). Unmaterializable (None) when a needed field is swept;
    ambiguous multi-width use is an error."""
    widths: set[int] = set()
    for site_name in _featurizer_sites_raw(name, normalized):
        if not isinstance(site_name, str):
            return None
        site = normalized.get("sites", {}).get(site_name)
        if site is None:
            return None
        component = site.get("component")
        head = site.get("head")
        if _is_sweep(component) or _is_sweep(head):
            return None
        if not isinstance(component, str):
            return None
        widths.add(
            component_width(
                info, component, head=head if isinstance(head, int) else None
            )
        )
    if not widths:
        return None
    if len(widths) > 1:
        raise ValidationError(
            4,
            f"featurizer {name!r} is used at sites of different widths "
            f"{sorted(widths)} — one featurizer, one width",
            path=f"featurizers.{name}",
        )
    return widths.pop()


def _canon_train(
    train: Mapping[str, Any], info: ModelInfo | None, env: ResolutionEnv
) -> dict[str, Any]:
    out = dict(train)
    optimizer = dict(out["optimizer"])
    name = optimizer.get("name")
    if isinstance(name, str):
        for field, default in OPTIMIZER_DEFAULTS[name].items():
            optimizer.setdefault(field, default)
    out["optimizer"] = optimizer
    precision = dict(out.get("precision", {}))
    precision.setdefault("feature", "fp32")
    precision.setdefault("loss", "fp32")
    precision.setdefault("model", info.native_dtype if info is not None else "fp32")
    out["precision"] = precision
    out.setdefault("seed", 0)
    if "eval" in out and isinstance(out["eval"], Mapping):
        eval_spec = dict(out["eval"])
        split = eval_spec.get("split")
        if isinstance(split, str):
            eval_spec["digest"] = env.datasets.digest(split)  # a dataset ref too (§2.2)
        out["eval"] = eval_spec
    return out
