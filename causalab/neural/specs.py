"""The spec vocabulary: frozen where-values over the engine — WU1 (#503).

The one declarative where-surface (design-of-record:
``docs/REBASE_CAUSALAB_ON_NNTERP.md`` Part 6; the pyvene-era unit vocabulary
it replaced was deleted by the WU6 sweep, #508). Two frozen dataclasses
*compose* engine values instead of paralleling them:

* :class:`SiteSpec` — a :class:`~causalab.neural.featurized_site.FeaturizedSite`
  (real :class:`~causalab.neural.site.Site` / :class:`~causalab.neural.
  head_view.HeadSite` + featurizer + ``feature_ids``) plus a declarative
  position spec, an explicit result ``key``, and the raw feature ``width``.
  There is nothing to map: a spec is born holding a real site, so the
  pyvene component-string table survives only inside this module's *legacy
  loader branch* (see :func:`load_site_specs`).
* :class:`EditSpec` — a :class:`SiteSpec` plus a named intervention mode and
  its declarative params: the five-mode vocabulary
  (``interchange``/``interpolate``/``replace``/``add``/``noise``) with the
  legacy surface's construction-time validation.

Both are frozen values: feature-space attachment is functional
(:meth:`SiteSpec.with_featurizer` / :meth:`~SiteSpec.with_feature_ids` /
:meth:`~SiteSpec.with_positions` return new specs), so *which* feature space
a spec carries is a fact about the value, never about execution history.
Specs are model-free — width is bound by the builders (WU2) and every
model-conditional concern stays in the dataset layer.

Persistence (the legacy target-save/load successor) is
deliberately constructive and named-only:

* :func:`save_site_specs` writes one ``sites.json`` (per spec: ``key``,
  structured site record, ``feature_ids``, position **name**, ``width``, a
  ``featurizer`` presence flag, format version) plus non-trivial featurizers
  through :mod:`causalab.io.nested_artifacts` (safetensors + JSON meta).
  The featurizer payload is written **before** ``sites.json`` — the JSON is
  the bundle's commit point, so an interrupted save never leaves a
  ``sites.json`` whose payload is missing. JSON + safetensors only — this
  path adds **no** ``torch.save``/``torch.load``/pickle reader or writer.
* :func:`load_site_specs` *returns* specs (the legacy load mutated
  caller-prebuilt units). Site, featurizer, ``feature_ids``, ``key`` and
  ``width`` restore fully from bytes; a *named* position rebinds when the
  caller passes its ``token_positions`` mapping, else stays ``None`` with
  the name kept in the on-disk record; *literal* positions are plain data
  and always restore.
* The loader keeps a **legacy branch** for ``units_metadata.json`` bundles
  (version ``"2.0"``, written by the retired legacy target save), translating
  ``component_type`` through the retired adapter's table — the table's one
  surviving home after the where-unification sweep (#508).

What round-trips is the named subset, by design: sites, featurizers with a
``to_dict`` form, feature ids, position *names*, widths. Callables never do
(``interpolate_fn``, hand-built ``Edit.g``) — *named modes are specs;
arbitrary ``g`` is code.* A non-trivial featurizer whose ``to_dict()``
returns ``None`` fails the save **loudly**: the legacy path silently skipped
it and survived only because its mutating load left the caller's in-memory
featurizer in place — a constructive load cannot.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
from typing import Any, Callable, Mapping, Sequence

import torch

from causalab.neural.featurized_site import FeaturizedSite
from causalab.neural.featurizer import Featurizer
from causalab.neural.head_view import HeadSite
from causalab.neural.positions import PositionResolver
from causalab.neural.site import Site

__all__ = [
    "EditSpec",
    "SITE_SPECS_FORMAT_VERSION",
    "SiteSpec",
    "load_site_specs",
    "save_site_specs",
]

#: Format version written into (and required from) ``sites.json``.
SITE_SPECS_FORMAT_VERSION = "1.0"

#: The new bundle's JSON file (one record per spec, in save order).
_SITES_JSON = "sites.json"
#: Shared stem for the featurizer payload (``<stem>.safetensors`` +
#: ``<stem>.meta.json``) — same stem the legacy bundle used, so both formats
#: keep featurizers in the same pair of files.
_FEATURIZERS_STEM = "featurizers"
#: The legacy bundle's metadata file (written by the retired target save).
_LEGACY_METADATA_JSON = "units_metadata.json"
#: The legacy metadata version this loader's legacy branch reads.
_LEGACY_VERSION = "2.0"

#: What :attr:`SiteSpec.positions` accepts: a declarative resolver (any
#: ``PositionResolver`` — ``TokenPosition`` and its combinators, any
#: ``ComponentIndexer``), literal token positions (a flat row broadcast to
#: every example, like the legacy unit surface's static index list), or
#: ``None`` (an *unbound* spec — bind before use; the dataset entry points
#: refuse to resolve ``None``, it is never read as "all positions").
Positionish = PositionResolver | Sequence[int] | torch.Tensor | None


def _normalize_positions(positions: Positionish) -> Any:
    """Validate + normalize ``positions`` at construction time.

    Literal rows (list/tuple/range of ints, or a 1-D integer tensor) become a
    plain ``tuple[int, ...]`` — hashable, JSON-serializable, and accepted
    verbatim by :func:`causalab.neural.positions.resolve_positions_batched`.
    Resolvers (anything with an ``index`` method — the
    :class:`~causalab.neural.positions.PositionResolver` protocol) pass
    through untouched. Anything else is refused here, not at resolve time.
    """
    if positions is None:
        return None
    if isinstance(positions, torch.Tensor):
        if positions.dim() != 1:
            raise ValueError(
                f"literal positions tensor must be 1-D, got {positions.dim()}-D"
            )
        return tuple(int(p) for p in positions.tolist())
    if isinstance(positions, (list, tuple, range)):
        return tuple(int(p) for p in positions)
    if isinstance(positions, (str, bytes, bytearray)):
        # str/bytes carry an `index` METHOD, so the resolver duck-type below
        # would accept them and fail confusingly at resolve time.
        raise TypeError(
            f"positions must be a PositionResolver, a literal row of ints, or "
            f"None; got {type(positions).__name__!r} (a position *name* only "
            "binds at load time, via load_site_specs's token_positions)."
        )
    if callable(getattr(positions, "index", None)):
        return positions
    raise TypeError(
        f"positions must be a PositionResolver, a literal row of ints, or None; "
        f"got {type(positions).__name__!r}"
    )


@dataclasses.dataclass(frozen=True)
class SiteSpec:
    """One intervention/collection site as a frozen value — the legacy
    model-unit successor (WU1, #503).

    Parameters
    ----------
    fsite :
        The engine's :class:`~causalab.neural.featurized_site.FeaturizedSite`
        — a real :class:`~causalab.neural.site.Site` or
        :class:`~causalab.neural.head_view.HeadSite` plus featurizer and
        optional ``feature_ids``. Composing it (instead of re-declaring
        location fields) means ``FeaturizedSite.__post_init__``'s
        ``feature_ids`` validation (non-empty, unique, non-negative, bounded
        by ``n_features``) applies to every spec for free. An *empty*
        selection is deliberately not constructible — an all-features-off
        edit is a no-op, expressed by omitting the edit (the dataset layer's
        empty-selection contract), not in the engine.
    positions :
        Declarative position spec: any
        :class:`~causalab.neural.positions.PositionResolver` (e.g. a
        ``TokenPosition``), a literal row of token indices broadcast to every
        example (normalized to ``tuple[int, ...]``), or ``None``.
    key :
        The explicit result/bundle key (collect outputs, saved records) —
        the legacy unit-id successor. Unique per run/bundle and **opaque**:
        nothing may parse it.
    width :
        The raw feature width at the site (hidden/intermediate/head dim) —
        the legacy ``shape`` field's one real use (DAS/DBM rotation sizing).
        Bound from model config by the builders (WU2); ``None`` when unbound.
    """

    fsite: FeaturizedSite
    positions: Positionish
    key: str
    width: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.fsite, FeaturizedSite):
            raise TypeError(
                f"fsite must be a FeaturizedSite, got {type(self.fsite).__name__!r}"
            )
        if not isinstance(self.key, str) or not self.key:
            raise ValueError(f"key must be a non-empty string, got {self.key!r}")
        if self.width is not None:
            if isinstance(self.width, bool) or not isinstance(self.width, int):
                raise TypeError(f"width must be an int or None, got {self.width!r}")
            if self.width < 1:
                raise ValueError(f"width must be positive, got {self.width}")
        object.__setattr__(self, "positions", _normalize_positions(self.positions))

    # ------------------------ functional updates ------------------------- #
    def with_featurizer(self, featurizer: Featurizer) -> SiteSpec:
        """A new spec reading/writing through ``featurizer`` (site, positions,
        ``feature_ids``, key, width unchanged). The ``set_featurizer``
        successor: ``FeaturizedSite`` re-validates ``feature_ids`` against the
        new featurizer's ``n_features``, so a stale subselection fails here,
        at attach time."""
        return dataclasses.replace(
            self, fsite=dataclasses.replace(self.fsite, featurizer=featurizer)
        )

    def with_feature_ids(self, feature_ids: Sequence[int] | None) -> SiteSpec:
        """A new spec addressing ``feature_ids`` inside the (unchanged)
        feature space — the ``set_feature_indices`` successor, with
        ``FeaturizedSite``'s full validation (non-empty, unique,
        non-negative, bounded)."""
        ids = None if feature_ids is None else tuple(int(i) for i in feature_ids)
        return dataclasses.replace(
            self, fsite=dataclasses.replace(self.fsite, feature_ids=ids)
        )

    def with_positions(self, positions: Positionish) -> SiteSpec:
        """A new spec that reads/writes at ``positions`` — the legacy
        position-resolver-swap successor, same
        shallow-view semantics: the returned spec *shares* this spec's
        ``fsite`` (featurizer and site) and keeps its ``key`` (so rows
        collected through the view accumulate under the real spec) and
        ``width``; only the position spec is swapped."""
        return dataclasses.replace(self, positions=positions)


@dataclasses.dataclass(frozen=True)
class EditSpec:
    """One site's intervention as a frozen value — the legacy unit-edit
    successor (WU1, #503), same five-mode vocabulary and construction-time
    validation.

    ``mode`` picks the feature-space operation:

    * ``"interchange"`` / ``"interpolate"`` — the source is read from the
      spec's positions on the *counterfactual* input of the edit's group.
    * ``"replace"`` / ``"add"`` — ``vector`` supplies the source: a
      feature-space ``(n_features,)`` broadcast to every example, or
      ``(n_examples, n_features)`` per-example rows. ``scale`` multiplies the
      vector (``add``: ``f + scale·v``; ``replace``: ``scale·v``).
    * ``"noise"`` — ``f + scale·randn`` drawn from a stream seeded ``seed``
      (defaults to ``0`` — the legacy private-stream default). When
      ``vector`` is set it supplies a per-feature noise scale (times
      ``scale``) — the causal-tracing corruption contract.

    Round-trip boundary (Part 6): an ``EditSpec`` is serializable iff its
    params are data — ``vector`` is a tensor, ``seed`` an int.
    ``interpolate_fn`` is a callable and never serializes; neither does any
    hand-built :class:`~causalab.neural.edit.Edit`.

    Value-equality and hashability are **undefined** once ``vector`` or
    ``interpolate_params`` is set (tensor ``==`` is elementwise; a mapping is
    unhashable) — exact parity with the legacy unit-edit's latent behavior.
    Do not compare or hash ``EditSpec``\\s; key on ``site.key`` instead.
    """

    site: SiteSpec
    mode: str = "interchange"
    vector: torch.Tensor | None = None
    scale: float = 1.0
    seed: int | None = None
    interpolate_fn: Callable[..., torch.Tensor] | None = None
    interpolate_params: Mapping[str, Any] = dataclasses.field(default_factory=dict)

    _MODES = ("interchange", "interpolate", "replace", "add", "noise")

    def __post_init__(self) -> None:
        if not isinstance(self.site, SiteSpec):
            raise TypeError(
                f"site must be a SiteSpec, got {type(self.site).__name__!r}"
            )
        if self.mode not in self._MODES:
            raise ValueError(
                f"unknown mode {self.mode!r}; expected one of {self._MODES}"
            )
        if self.mode in ("replace", "add") and self.vector is None:
            raise ValueError(f"mode {self.mode!r} needs a vector")
        if self.mode == "interpolate" and self.interpolate_fn is None:
            raise ValueError("mode 'interpolate' needs interpolate_fn")
        if self.mode == "noise" and self.seed is None:
            object.__setattr__(self, "seed", 0)

    @property
    def needs_source(self) -> bool:
        return self.mode in ("interchange", "interpolate")


# --------------------------------------------------------------------------- #
#  serialization — sites.json + featurizers via nested_artifacts               #
# --------------------------------------------------------------------------- #
def _site_record(site: Any, key: str) -> dict[str, Any]:
    """The structured JSON record for a spec's site — component/layer for a
    :class:`Site`, kind/layer/head for a :class:`HeadSite`. Any other
    ``WritableSite`` implementation has no named record and is refused."""
    if isinstance(site, HeadSite):
        return {
            "type": "head_site",
            "kind": site.kind,
            "layer": site.layer,
            "head": site.head,
        }
    if isinstance(site, Site):
        return {"type": "site", "component": site.component, "layer": site.layer}
    raise ValueError(
        f"spec {key!r}: cannot serialize site of type {type(site).__name__!r}; "
        "only Site and HeadSite have a structured record."
    )


def _site_from_record(record: Mapping[str, Any]) -> Site | HeadSite:
    site_type = record.get("type")
    if site_type == "head_site":
        return HeadSite(
            kind=record["kind"], layer=int(record["layer"]), head=int(record["head"])
        )
    if site_type == "site":
        return Site(record["component"], int(record["layer"]))
    raise ValueError(f"unknown site record type {site_type!r} in {_SITES_JSON}")


def _positions_record(positions: Any, key: str) -> dict[str, Any] | None:
    """The JSON record for a spec's positions: ``None``, a literal row (plain
    data — always round-trips), or a resolver's **name** (rebound at load
    time via the caller's ``token_positions`` mapping). A nameless resolver
    cannot be re-bound by anything, so it is refused here rather than
    silently dropped."""
    if positions is None:
        return None
    if isinstance(positions, tuple):  # normalized literal row
        return {"kind": "literal", "positions": [int(p) for p in positions]}
    name = getattr(positions, "id", None)
    if not isinstance(name, str) or not name:
        raise ValueError(
            f"spec {key!r}: position resolver {positions!r} has no `id` name to "
            "serialize; a saved position rebinds by name (give the resolver an "
            "id, use literal positions, or set positions=None)."
        )
    return {"kind": "named", "name": name}


def _bind_positions(
    record: Mapping[str, Any] | None,
    token_positions: Mapping[str, PositionResolver] | None,
    key: str,
) -> Any:
    """Restore a spec's positions from its record: literal rows restore
    directly; a named position binds through ``token_positions`` when
    supplied (a missing name is an error — the legacy loaders' rebinding
    contract) and stays ``None`` otherwise (the name remains in
    the on-disk record)."""
    if record is None:
        return None
    kind = record.get("kind")
    if kind == "literal":
        return tuple(int(p) for p in record["positions"])
    if kind == "named":
        if token_positions is None:
            return None
        name = record["name"]
        if name not in token_positions:
            raise ValueError(
                f"spec {key!r}: position named {name!r} not found in the provided "
                f"token_positions (available: {sorted(token_positions)})."
            )
        return token_positions[name]
    raise ValueError(f"unknown positions record kind {kind!r} in {_SITES_JSON}")


def _featurizer_payload(spec: SiteSpec) -> dict[str, Any] | None:
    """The spec's featurizer as a serializable dict, or ``None`` for a
    trivial (identity) featurizer that needs no bytes. A **non-trivial**
    featurizer without a ``to_dict`` form fails loudly: the legacy save
    silently skipped it (``ComposedFeaturizer.to_dict`` returns ``None``
    whenever any stage's does), which only worked because the mutating
    legacy load left the caller's in-memory featurizer in place — a
    constructive load would silently come back with the identity."""
    featurizer = spec.fsite.featurizer
    payload = featurizer.to_dict()
    if payload is None and not featurizer.is_trivial():
        raise ValueError(
            f"spec {spec.key!r}: featurizer {featurizer.id!r} "
            f"({type(featurizer).__name__}) is non-trivial but its to_dict() "
            "returned None, so it cannot be saved. Saving would silently drop "
            "the feature space and a load would reconstruct the identity "
            "featurizer instead. Make every composed stage serializable, or "
            "attach the featurizer after loading."
        )
    return payload


def save_site_specs(specs: Sequence[SiteSpec], dir: str) -> str:
    """Save ``specs`` to ``dir`` as the new spec bundle — the constructive
    successor of the retired legacy target save.

    Writes ``sites.json`` (one record per spec, in order: ``key``, structured
    site record, ``feature_ids``, position record, ``width``, a ``featurizer``
    presence flag, plus the bundle format version) and, when any spec carries
    a non-trivial featurizer, ``featurizers.safetensors`` +
    ``featurizers.meta.json`` via
    :func:`causalab.io.nested_artifacts.save_nested` (keyed by spec ``key``).
    The featurizer payload is written **first**; ``sites.json`` is the
    bundle's commit point, and its per-record ``featurizer`` flag lets the
    loader refuse a bundle whose payload went missing instead of silently
    reconstructing identity featurizers. JSON + safetensors only — no
    ``torch.save``/pickle. Returns ``dir``.

    Raises
    ------
    ValueError
        On duplicate keys, a non-trivial featurizer whose ``to_dict()``
        returns ``None`` (the legacy silent-drop hazard, now loud), a
        nameless position resolver, or a site type without a record form.
    """
    keys = [spec.key for spec in specs]
    if len(set(keys)) != len(keys):
        dupes = sorted({k for k in keys if keys.count(k) > 1})
        raise ValueError(f"duplicate spec keys: {dupes}")

    records: list[dict[str, Any]] = []
    featurizers: dict[str, dict[str, Any]] = {}
    for spec in specs:
        payload = _featurizer_payload(spec)
        if payload is not None:
            featurizers[spec.key] = payload
        records.append(
            {
                "key": spec.key,
                "site": _site_record(spec.fsite.site, spec.key),
                "feature_ids": (
                    None
                    if spec.fsite.feature_ids is None
                    else [int(i) for i in spec.fsite.feature_ids]
                ),
                "positions": _positions_record(spec.positions, spec.key),
                "width": spec.width,
                "featurizer": payload is not None,
            }
        )

    os.makedirs(dir, exist_ok=True)
    # Payload first, sites.json last: the JSON is the commit point. A save
    # that dies mid-payload leaves no sites.json, so the half-bundle cannot
    # be loaded at all — and the per-record "featurizer" flag lets the loader
    # refuse a bundle whose payload was deleted later.
    if featurizers:
        # Lazy import: `io/` sits above `neural/` in the layering (io imports
        # neural), so the module scope must not close the cycle — the same
        # pattern the retired legacy target save used.
        from causalab.io.nested_artifacts import save_nested

        save_nested(featurizers, dir, _FEATURIZERS_STEM)
    with open(os.path.join(dir, _SITES_JSON), "w") as f:
        json.dump(
            {"format_version": SITE_SPECS_FORMAT_VERSION, "specs": records},
            f,
            indent=2,
        )
    return dir


def load_site_specs(
    dir: str,
    token_positions: Mapping[str, PositionResolver] | None = None,
) -> list[SiteSpec]:
    """Load a spec bundle from ``dir`` — **constructive**: returns the specs
    (the retired legacy load mutated units the caller had to
    pre-build with matching ids).

    Site, featurizer, ``feature_ids``, ``key`` and ``width`` restore fully
    from bytes. Positions restore by *name*: pass the task's
    ``token_positions`` mapping (``create_token_positions(pipeline)``) to
    rebind — a recorded name missing from the mapping is an error — or pass
    ``None`` to get ``positions=None`` specs (the names stay in the on-disk
    record). Literal position rows are plain data and always restore.

    Both formats carry a featurizer-presence signal (the new record's
    ``featurizer`` flag; the legacy ``featurizer_info.is_trivial``): a bundle
    that claims a non-trivial featurizer whose payload entry is absent is
    refused loudly instead of silently loading an identity featurizer. When
    a directory holds both formats, the new ``sites.json`` wins.

    Reads two formats:

    * the new bundle (``sites.json`` + optional featurizer payload), format
      version ``"1.0"``;
    * the **legacy branch**: a ``units_metadata.json`` bundle (version
      ``"2.0"``, written by the retired legacy target save).
      ``component_type`` translates through the retired adapter's per-head
      table (whole-sublayer strings pass
      1:1 — the Site vocabulary *is* those strings); the legacy ``index_id``
      is the position name; ``width`` is the legacy ``shape[0]`` (the
      field's one real use — feature-width). A per-head unit's head index
      lives only in its legacy id string (``...,Head-N,...``) and is parsed
      from there — a legacy-format-only concession; the new format stores
      ``head`` structurally and its keys stay opaque.
    """
    sites_path = os.path.join(dir, _SITES_JSON)
    legacy_path = os.path.join(dir, _LEGACY_METADATA_JSON)
    if os.path.exists(sites_path):
        return _load_new_format(dir, sites_path, token_positions)
    if os.path.exists(legacy_path):
        return _load_legacy_format(dir, legacy_path, token_positions)
    raise FileNotFoundError(
        f"no spec bundle in {dir}: expected {_SITES_JSON} (new format) or "
        f"{_LEGACY_METADATA_JSON} (legacy units-metadata bundle)."
    )


def _load_featurizer_payloads(dir: str) -> dict[str, Any]:
    """The bundle's featurizer dicts keyed by spec key / legacy unit id
    (empty when the bundle has no featurizer payload)."""
    if not os.path.exists(os.path.join(dir, f"{_FEATURIZERS_STEM}.safetensors")):
        return {}
    from causalab.io.nested_artifacts import load_nested  # see save_site_specs

    payloads, _meta = load_nested(dir, _FEATURIZERS_STEM)
    return payloads


def _load_new_format(
    dir: str,
    sites_path: str,
    token_positions: Mapping[str, PositionResolver] | None,
) -> list[SiteSpec]:
    with open(sites_path) as f:
        payload = json.load(f)
    version = payload.get("format_version")
    if version != SITE_SPECS_FORMAT_VERSION:
        raise ValueError(
            f"{_SITES_JSON} format version {version!r} is not the supported "
            f"{SITE_SPECS_FORMAT_VERSION!r}."
        )
    featurizers = _load_featurizer_payloads(dir)

    specs: list[SiteSpec] = []
    for record in payload["specs"]:
        key = record["key"]
        if record["featurizer"] and key not in featurizers:
            raise ValueError(
                f"spec {key!r}: {_SITES_JSON} records a non-trivial featurizer "
                f"but the featurizer payload ({_FEATURIZERS_STEM}.safetensors/"
                f".meta.json) is missing it — a truncated or partially deleted "
                "bundle. Refusing to silently reconstruct an identity "
                "featurizer; restore the payload files or regenerate the "
                "bundle."
            )
        featurizer = (
            Featurizer.from_dict(featurizers[key])
            if record["featurizer"]
            else Featurizer()
        )
        feature_ids = (
            None
            if record["feature_ids"] is None
            else tuple(int(i) for i in record["feature_ids"])
        )
        specs.append(
            SiteSpec(
                fsite=FeaturizedSite(
                    _site_from_record(record["site"]), featurizer, feature_ids
                ),
                positions=_bind_positions(record["positions"], token_positions, key),
                key=key,
                width=record["width"],
            )
        )
    return specs


#: Head index inside a legacy unit id — ``AttentionHead(Layer-0,Head-2,...)``.
_LEGACY_HEAD_RE = re.compile(r"[(,]Head-(\d+)[,)]")

#: pyvene-era per-head component strings → HeadSite kinds. Whole-sublayer
#: components pass through 1:1 (the Site vocabulary IS those strings). The
#: retired legacy adapter's component table — its one surviving home is this
#: module's legacy loader branch (#508).
_HEAD_COMPONENTS: dict[str, str] = {
    "head_value_output": "value",
    "head_query_output": "query",
    "head_attention_value_output": "attention_value",
}


def _legacy_site(metadata: Mapping[str, Any], uid: str) -> Site | HeadSite:
    """A legacy record's engine site — the retired legacy adapter's component
    mapping, reproduced exactly (per-head strings through the table above,
    whole-sublayer strings 1:1), with the head index recovered from the one
    place the legacy format kept it: the unit id string."""
    component = metadata["component_type"]
    layer = int(metadata["layer"])
    if component in _HEAD_COMPONENTS:
        match = _LEGACY_HEAD_RE.search(uid)
        if match is None:
            raise ValueError(
                f"legacy unit {uid!r} has per-head component {component!r} but "
                "its id does not encode a head index (expected ',Head-N,'); "
                "the legacy metadata stores the head nowhere else."
            )
        return HeadSite(
            kind=_HEAD_COMPONENTS[component], layer=layer, head=int(match.group(1))
        )
    if component.startswith("head_"):
        raise ValueError(
            f"per-head component {component!r} has no HeadSite mapping "
            f"(supported: {sorted(_HEAD_COMPONENTS)})."
        )
    return Site(component, layer)


def _load_legacy_format(
    dir: str,
    metadata_path: str,
    token_positions: Mapping[str, PositionResolver] | None,
) -> list[SiteSpec]:
    with open(metadata_path) as f:
        all_metadata = json.load(f)
    featurizers = _load_featurizer_payloads(dir)

    specs: list[SiteSpec] = []
    for uid, metadata in all_metadata.items():
        version = metadata.get("version")
        if version != _LEGACY_VERSION:
            raise ValueError(
                f"legacy unit {uid!r} has version {version!r}; this loader "
                f"reads version {_LEGACY_VERSION!r} bundles."
            )
        # The legacy metadata's own presence signal (featurizer_info.is_trivial,
        # always written by the retired legacy save): a non-trivial featurizer
        # whose payload entry is absent means either a truncated bundle or a
        # historic silent-drop bundle (the legacy save skipped any featurizer
        # whose to_dict() returned None) — both unloadable constructively.
        is_trivial = bool(
            (metadata.get("featurizer_info") or {}).get("is_trivial", True)
        )
        if not is_trivial and uid not in featurizers:
            raise ValueError(
                f"legacy unit {uid!r}: units_metadata.json records a "
                "non-trivial featurizer (featurizer_info.is_trivial=false) but "
                f"the featurizer payload ({_FEATURIZERS_STEM}.safetensors/"
                ".meta.json) is missing it — a truncated bundle, or one whose "
                "featurizer the legacy save silently dropped (to_dict() "
                "returned None). A constructive load cannot substitute the "
                "caller's in-memory featurizer the way the legacy mutating "
                "load did; regenerate the artifact."
            )
        featurizer = (
            Featurizer.from_dict(featurizers[uid]) if not is_trivial else Featurizer()
        )
        feature_indices = metadata.get("feature_indices")
        if feature_indices is not None and len(feature_indices) == 0:
            raise ValueError(
                f"legacy unit {uid!r} has an empty feature_indices selection "
                "([]): an all-features-off selection (e.g. a DBM mask that "
                "switched everything off) is the dataset-layer no-op contract "
                "and has no constructive SiteSpec form — the engine refuses "
                "empty feature_ids by design. Drop the unit from the bundle "
                "or regenerate the artifact."
            )
        feature_ids = (
            None if feature_indices is None else tuple(int(i) for i in feature_indices)
        )
        index_id = metadata.get("index_id")
        positions_record = (
            None if index_id is None else {"kind": "named", "name": index_id}
        )
        shape = metadata.get("shape")
        specs.append(
            SiteSpec(
                fsite=FeaturizedSite(
                    _legacy_site(metadata, uid), featurizer, feature_ids
                ),
                positions=_bind_positions(positions_record, token_positions, uid),
                key=uid,
                width=None if shape is None else int(shape[0]),
            )
        )
    return specs
