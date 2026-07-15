"""Core logic for ingesting a manifold-SAE-autointerp bundle community.

A manifold-SAE-autointerp run (e.g. ``semantic_admm3e4_10M/``) is **not** a
single ``.safetensors`` subspace — it is a bundle of per-community records.
Each community lives as one line in ``hypotheses.jsonl`` (keyed by ``comm``)
and carries everything ``characterize_subspace`` needs:

- ``subspace.projection_matrix`` — a ``(n_dims, d_model)`` rotation,
- ``config`` — ``model_id`` / ``layer_index`` / ``hook_site``,
- ``significance_description`` — markdown that embeds the per-manifold
  exemplar spans under a ``## Manifold exemplars`` section,
- ``hypothesis.theme`` + ``topology_characterization`` — the provided
  significance.

:func:`build_characterize_inputs` turns ``(bundle_dir, community_id)`` into the
input set ``characterize_subspace`` consumes — a ``rotation_matrix`` safetensors,
a ``step1_dataset.json`` of exemplar spans, and a ``subspace_manifest.json`` — with
**no hand-editing** (issue #265). The Hydra entry point in :mod:`.main` wraps it.

Robustness notes (the failure modes earlier sessions hit by hand):

- The exemplar parser splits on ``[label]`` entry markers, not on quotes, so
  multi-line spans and spans containing embedded ``"`` quotes survive (a naive
  ``"(.*?)"`` regex dropped 7/16).
- The ``hypotheses.jsonl`` scan is word-boundary-anchored on the ``comm`` value
  (``"comm": 46`` never matches ``466``) and streams line by line, so it does
  not load the ~900 MB file into memory.
- ``picks/comm<ID>_<label>.json`` filenames are advisory and can be wrong
  (``comm396_finance.json`` is actually about *ease*); when a pick filename's
  label is absent from the derived theme and member-feature labels, we warn and
  trust the record content. Member-feature ids/labels come from the pick's JSON
  ``member_features`` — ``communities.pkl`` is deliberately not unpickled.
"""

from __future__ import annotations

import json
import logging
import pathlib
import re
from typing import Any

import torch
from safetensors.torch import save_file

logger = logging.getLogger(__name__)

# The reproduction gate needs ≥8 spans for the margin metric (and ≥4 for TwoNN);
# a full grid is 16. Below this we still write the inputs but warn — the gate
# will then surface a precise, actionable failure rather than us guessing.
_MIN_EXEMPLARS_FOR_GATE = 8

# Isolate the exemplar section (until the next ``## `` header or end of string),
# then split it into entries on their ``[label]`` markers. Splitting on the
# labels — not on the surrounding quotes — is what makes multi-line spans and
# spans with embedded ``"`` survive.
_EXEMPLAR_SECTION_RE = re.compile(r"(?ms)^##\s+Manifold\s+exemplars\b.*?(?=^##\s|\Z)")
_ENTRY_LABEL_RE = re.compile(r"(?m)^\[[^\]\n]+\][ \t]*")
_ELLIPSIS_EDGE_RE = re.compile(r"^\.{2,}|\.{2,}$")


def _clean_exemplar(raw: str) -> str:
    """Normalise one raw ``[label]``-delimited span into natural text.

    Strips a single surrounding quote pair, the ``<|begin_of_text|>`` special
    token, the ``>>>``/``<<<`` activation markers (keeping the peak token), and
    leading/trailing ``...`` truncation. Returns ``""`` for an empty span.
    """
    t = raw.strip()
    if t.startswith('"'):
        t = t[1:]
    if t.endswith('"'):
        t = t[:-1]
    t = t.strip()
    t = t.replace("<|begin_of_text|>", "")
    t = t.replace(">>>", "").replace("<<<", "")
    t = _ELLIPSIS_EDGE_RE.sub("", t)
    return t.strip()


def extract_manifold_exemplars(significance_description: str | None) -> list[str]:
    """Extract the per-manifold exemplar spans from a ``significance_description``.

    Returns the cleaned spans in document order. Empty input, a missing
    ``## Manifold exemplars`` section, or a section with no ``[label]`` entries
    all yield an empty list.
    """
    if not significance_description:
        return []
    section_match = _EXEMPLAR_SECTION_RE.search(significance_description)
    if section_match is None:
        return []
    section = section_match.group(0)
    labels = list(_ENTRY_LABEL_RE.finditer(section))
    spans: list[str] = []
    for i, label in enumerate(labels):
        start = label.end()
        end = labels[i + 1].start() if i + 1 < len(labels) else len(section)
        cleaned = _clean_exemplar(section[start:end])
        if cleaned:
            spans.append(cleaned)
    return spans


def _find_pick_file(
    bundle_dir: pathlib.Path, community_id: int
) -> tuple[str, str | None] | None:
    """Return ``(path, label)`` for a ``picks/comm<ID>_<label>.json`` if one exists.

    The label is the filename segment after ``comm<ID>_``; ``None`` when the pick
    is the unlabelled ``comm<ID>.json``. The ``comm<ID>_`` prefix is exact, so
    community ``396`` never matches ``comm3960_*``.
    """
    picks = bundle_dir / "picks"
    if not picks.is_dir():
        return None
    labelled = sorted(picks.glob(f"comm{community_id}_*.json"))
    if labelled:
        path = labelled[0]
        label = path.stem[len(f"comm{community_id}_") :]
        return str(path), (label or None)
    unlabelled = picks / f"comm{community_id}.json"
    if unlabelled.is_file():
        return str(unlabelled), None
    return None


def _scan_jsonl_for_comm(jsonl_path: str, community_id: int) -> dict[str, Any] | None:
    """Stream ``hypotheses.jsonl`` and return the record whose ``comm`` matches.

    Uses a word-boundary-anchored prefilter so we full-parse only the candidate
    line(s); the boundary stops ``community_id=46`` from matching ``"comm": 466``.
    Streams line by line — the real file is ~900 MB.
    """
    pattern = re.compile(rf'"comm"\s*:\s*{int(community_id)}\b')
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            if not pattern.search(line):
                continue
            record = json.loads(line)
            if int(record.get("comm", -1)) == int(community_id):
                return record
    return None


def _read_member_features(pick_path: str | None) -> list[dict[str, Any]]:
    """The pick's ``member_features`` entries (``[]`` if no pick / none present).

    Pick files are JSON, so member-feature ids and labels come from here — we do
    **not** unpickle ``communities.pkl``. Communities without a pick simply carry
    no member-feature provenance, which is informational only.
    """
    if pick_path is None:
        return []
    with open(pick_path, "r", encoding="utf-8") as fh:
        pick = json.load(fh)
    return [m for m in pick.get("member_features", []) if isinstance(m, dict)]


def _pick_label_warning(
    *,
    community_id: int,
    pick_label: str | None,
    member_labels: list[str],
    theme: str,
    manifold_guess: str,
) -> str | None:
    """Warn when a pick filename's label disagrees with the record content.

    Splits the label on ``_`` and checks each token against the derived theme,
    manifold guess, and member-feature labels (case-insensitive). If *no* token
    appears, the filename is likely mislabelled and the record content should be
    trusted instead. Returns the warning message (also logged) or ``None``.
    """
    if not pick_label:
        return None
    haystack = " ".join([theme, manifold_guess, *member_labels]).lower()
    tokens = [tok for tok in re.split(r"[_\W]+", pick_label) if tok]
    if any(tok.lower() in haystack for tok in tokens):
        return None
    message = (
        f"Pick filename label {pick_label!r} for community {community_id} does not "
        f"appear in the derived theme or member-feature labels; trusting the record "
        f"content over the filename (cf. comm396_finance.json, which is about ease)."
    )
    logger.warning(message)
    return message


def find_community_record(
    bundle_dir: str, community_id: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve a community's record and provenance from a bundle.

    The authoritative record is read from ``hypotheses.jsonl``; a ``picks/``
    entry is used only as a fallback source and to drive the mislabel warning.
    Returns ``(record, provenance)`` where provenance carries the source, any
    pick label/warning, and member-feature ids.
    """
    bundle = pathlib.Path(bundle_dir)
    if not bundle.is_dir():
        raise FileNotFoundError(f"Bundle directory not found: {bundle_dir!r}.")

    jsonl_path = bundle / "hypotheses.jsonl"
    pick = _find_pick_file(bundle, community_id)
    pick_path = pick[0] if pick else None
    pick_label = pick[1] if pick else None

    record: dict[str, Any] | None = None
    source: str | None = None
    if jsonl_path.is_file():
        record = _scan_jsonl_for_comm(str(jsonl_path), community_id)
        if record is not None:
            source = "hypotheses.jsonl"
    if record is None and pick_path is not None:
        with open(pick_path, "r", encoding="utf-8") as fh:
            record = json.load(fh)
        source = "pick"
    if record is None:
        raise KeyError(
            f"Community {community_id} not found in {bundle_dir!r} "
            f"(checked hypotheses.jsonl and picks/)."
        )

    member_features = _read_member_features(pick_path)
    member_labels = [str(m["label"]) for m in member_features if m.get("label")]
    member_ids = [
        int(m["feature"]) for m in member_features if m.get("feature") is not None
    ]

    hypothesis = record.get("hypothesis", {}) or {}
    warning = _pick_label_warning(
        community_id=community_id,
        pick_label=pick_label,
        member_labels=member_labels,
        theme=str(hypothesis.get("theme", "")),
        manifold_guess=str(hypothesis.get("manifold_guess", "")),
    )

    provenance: dict[str, Any] = {
        "bundle_dir": str(bundle),
        "community_id": community_id,
        "source": source,
        "pick_path": pick_path,
        "pick_label": pick_label,
        "member_feature_ids": member_ids or None,
        "warnings": [warning] if warning else [],
    }
    return record, provenance


def _format_topology(topology: dict[str, Any]) -> str | None:
    """Compose ``topology_description`` as ``shape: description (metrics)``."""
    shape = topology.get("shape")
    description = topology.get("description")
    head_parts = [str(p) for p in (shape, description) if p]
    head = ": ".join(head_parts)
    metrics = [
        f"{key}={topology[key]}"
        for key in ("intrinsic_dim", "n_components", "n_loops", "n_voids")
        if topology.get(key) is not None
    ]
    if head and metrics:
        return f"{head} ({', '.join(metrics)})"
    if metrics:
        return ", ".join(metrics)
    return head or None


def build_characterize_inputs(
    *,
    bundle_dir: str,
    community_id: int,
    out_dir: str,
    artifact_name: str = "subspace.safetensors",
    step1_name: str = "step1_dataset.json",
    manifest_name: str = "subspace_manifest.json",
) -> dict[str, Any]:
    """Build a complete ``characterize_subspace`` input set from ``(bundle, comm)``.

    Writes the rotation safetensors, the exemplar ``step1_dataset.json``, and a
    ``subspace_manifest.json`` under ``out_dir``, and returns the manifest dict.
    The manifest matches the shape ``characterize_subspace`` consumes, so it
    needs no hand-editing.
    """
    record, provenance = find_community_record(bundle_dir, community_id)

    subspace = record.get("subspace") or {}
    projection_matrix = subspace.get("projection_matrix")
    if projection_matrix is None:
        raise KeyError(
            f"Community {community_id} record has no subspace.projection_matrix."
        )
    rotation = torch.tensor(projection_matrix, dtype=torch.float32)
    if rotation.ndim != 2:
        raise ValueError(
            f"projection_matrix must be 2-D (n_dims, d_model); got shape "
            f"{tuple(rotation.shape)} for community {community_id}."
        )
    n_dims_axis, d_model = int(rotation.shape[0]), int(rotation.shape[1])
    n_dims = int(subspace.get("n_dims", n_dims_axis))

    config = record.get("config") or {}
    for required in ("model_id", "layer_index", "hook_site"):
        if config.get(required) is None:
            raise KeyError(f"Community {community_id} config is missing {required!r}.")

    exemplars = extract_manifold_exemplars(record.get("significance_description"))
    if not exemplars:
        raise ValueError(
            f"No manifold exemplars extracted for community {community_id}; "
            "the significance_description has no '## Manifold exemplars' section "
            "or no parseable entries."
        )
    if len(exemplars) < _MIN_EXEMPLARS_FOR_GATE:
        logger.warning(
            "Only %d exemplar(s) extracted for community %d (the reproduction "
            "gate prefers >=%d); proceeding, but the gate may fail.",
            len(exemplars),
            community_id,
            _MIN_EXEMPLARS_FOR_GATE,
        )

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Save in the on-disk (n_dims, d_model) orientation under the first key
    # ``loading.load_subspace`` looks for; it auto-transposes to (d_model, k).
    artifact_path = out / artifact_name
    save_file({"rotation_matrix": rotation.contiguous()}, str(artifact_path))

    step1_path = out / step1_name
    with open(step1_path, "w", encoding="utf-8") as fh:
        json.dump(exemplars, fh, indent=2, ensure_ascii=False)

    hypothesis = record.get("hypothesis", {}) or {}
    topology = record.get("topology_characterization", {}) or {}
    manifest: dict[str, Any] = {
        "subspace_artifact": str(artifact_path),
        "model": str(config["model_id"]),
        "layer": int(config["layer_index"]),
        "site": str(config["hook_site"]),
        "k_features_hint": n_dims,
        "step1_dataset": str(step1_path),
        "significance": {
            "hypothesis_text": hypothesis.get("theme"),
            "figure_path": None,
            "topology_description": _format_topology(topology),
        },
        "provenance": {
            **provenance,
            "n_exemplars": len(exemplars),
            "n_dims": n_dims,
            "d_model": d_model,
        },
    }

    manifest_path = out / manifest_name
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    logger.info(
        "Built characterize inputs for community %d: rotation (d_model=%d, k=%d), "
        "%d exemplar span(s), manifest -> %s",
        community_id,
        d_model,
        n_dims,
        len(exemplars),
        manifest_path,
    )
    return manifest
