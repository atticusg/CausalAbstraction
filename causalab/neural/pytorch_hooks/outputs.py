"""Run outputs: the save manifest written to disk, stamped (spec §2.12, §8).

Tensors (reads, featurizer bundles) land in ``.safetensors`` with the
point's ``ArtifactIdentity`` in the header; per-example metric tables land
in ``.parquet``. In swept documents the authored ``file_path`` is
unchanged: axis coordinates become columns of the metric tables and key
suffixes on tensor entries (``rot[k=8,seed=0]``).

**Per-entry provenance.** A swept document writes one file from many points,
so file-level identity can only carry what *every* point agrees on: the
fields that vary (``k``, the point digest, a swept site) live in an
``entries`` table in the safetensors ``__metadata__``, one record per tensor
key. That table is what makes an entry both selectable
(:mod:`causalab.protocol.bundles`) and provable — before it, whichever point
executed last silently stamped the whole file."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import torch

from causalab.protocol.bundles import RAGGED_SUFFIX, entry_key
from causalab.protocol.errors import ProtocolError
from causalab.protocol.resolve import build_artifact_identity
from causalab.protocol.sweep import coordinate_label, short_coords

__all__ = ["MetricTable", "TensorFile", "write_outputs"]


class TensorFile:
    """Accumulates tensor entries for one save file across points."""

    def __init__(self) -> None:
        self.entries: dict[str, torch.Tensor] = {}
        self.metadata: dict[str, str] = {}
        #: key -> {"slot", "coords", …identity}; serialized as ``entries``
        self.entry_meta: dict[str, dict[str, Any]] = {}
        self._common_seen = False

    def add(
        self,
        name: str,
        value: Any,
        coords: Mapping[str, Any],
        *,
        label_entry: str | None = None,
        reduce: str | None = None,
        identity: Mapping[str, Any] | None = None,
    ) -> None:
        """Add one point's value under ``name``.

        ``label_entry`` is the *declared* entity the coordinates belong to,
        which for a featurizer bundle is the featurizer, not the slot: the
        axis ``featurizers.rot.k`` shortens to ``k`` against ``rot`` and to
        ``rot.k`` against ``weight``, and only the former is a name a
        consuming document can write in an ``entry`` selector."""
        entity = label_entry or name
        key = entry_key(name, coordinate_label(coords, entry=entity) if coords else "")
        self.entry_meta[key] = {
            "slot": name,
            "coords": {
                short: _plain(coord)
                for short, coord in short_coords(coords, entry=entity).items()
            },
            **{k: str(v) for k, v in (identity or {}).items()},
        }
        from causalab.neural.pytorch_hooks.executor import RaggedValue

        if reduce is not None:
            self.entries[key] = _reduce_rows(value, reduce)
            return
        if isinstance(value, RaggedValue):
            # ragged reads persist as the flat gather + per-row widths
            self.entries[key] = value.flat.detach().to("cpu").contiguous()
            self.entries[f"{key}{RAGGED_SUFFIX}"] = torch.tensor(
                value.widths, dtype=torch.long
            )
            return
        self.entries[key] = value.detach().to("cpu").contiguous()

    def record_common(self, identity: Mapping[str, Any]) -> None:
        """Fold one point's identity into the file-level stamp, keeping only
        the fields every point so far agrees on.

        A single-point document therefore stamps exactly what it always did;
        a swept one drops the fields that differ (``k``, the point digest)
        rather than letting the last point speak for the file. The dropped
        fields are still provable per entry via the ``entries`` table."""
        stamped = {key: str(value) for key, value in identity.items()}
        if not self._common_seen:
            self.metadata.update(stamped)
            self._common_seen = True
            return
        for key in list(self.metadata):
            if self.metadata[key] != stamped.get(key):
                del self.metadata[key]


def _reduce_rows(value: Any, reduce: str) -> torch.Tensor:
    """§2.12 ``reduce``: a statistic over a read's gathered rows instead of
    the rows themselves — ``(…, width)`` collapses to ``(width,)``, the
    broadcast form a write operand takes.

    Reducing here rather than downstream is the point: the un-reduced
    harvest never reaches disk, which for an ablation grid is the difference
    between gigabytes of activations and kilobytes of means. The
    accumulation is fp32 regardless of the run's dtype — a bf16 sum over
    thousands of rows loses the low bits it is meant to average.
    """
    from causalab.neural.pytorch_hooks.executor import RaggedValue

    rows = value.flat if isinstance(value, RaggedValue) else value
    flat = (
        rows.detach().to(device="cpu", dtype=torch.float32).reshape(-1, rows.shape[-1])
    )
    if reduce == "mean":
        return flat.mean(dim=0).contiguous()
    raise ProtocolError("P2", f"unknown save reduction {reduce!r}")


class MetricTable:
    """Accumulates per-example metric rows for one save file across points."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(
        self, name: str, values: list[Any], coords: Mapping[str, Any], point_digest: str
    ) -> None:
        for example, value in enumerate(values):
            self.rows.append(self._row(name, example, value, coords, point_digest))

    def add_windowed(
        self,
        name: str,
        values: list[list[Any]],
        coords: Mapping[str, Any],
        point_digest: str,
        *,
        steps: list[list[int]] | None,
        matched: list[bool],
    ) -> None:
        """Rows for a metric over a read that addresses several positions.

        One row per (example, position), carrying the ``step`` it scored and
        whether the example addressed anything at all. An example that
        addressed **nothing** — a row that stopped generating, or never said
        the value a ``variable`` anchor looks for — still gets exactly one
        row, with a null value and ``matched=false``: "the model never said
        it" has to be distinguishable from "it said it and scored 0", and a
        missing row would make the two look identical after a group-by.

        ``steps`` is ``None`` for a kind that reduces the whole window to
        one value (``decode``): there is no single step such a value belongs
        to, so the column stays null rather than lying about one.
        """
        for example, row_values in enumerate(values):
            if not row_values:
                self.rows.append(
                    self._row(
                        name,
                        example,
                        None,
                        coords,
                        point_digest,
                        step=None,
                        matched=matched[example],
                    )
                )
                continue
            for offset, value in enumerate(row_values):
                self.rows.append(
                    self._row(
                        name,
                        example,
                        value,
                        coords,
                        point_digest,
                        step=steps[example][offset] if steps is not None else None,
                        matched=matched[example],
                    )
                )

    def _row(
        self,
        name: str,
        example: int,
        value: Any,
        coords: Mapping[str, Any],
        point_digest: str,
        *,
        step: int | None = None,
        matched: bool | None = None,
    ) -> dict[str, Any]:
        row: dict[str, Any] = {"example": example, "metric": name}
        if isinstance(value, dict):
            import json

            row["value"] = json.dumps(value, sort_keys=True)
        else:
            row["value"] = value
        if matched is not None:
            row["step"] = step
            row["matched"] = matched
        row.update({axis: _plain(coord) for axis, coord in coords.items()})
        row["produced_by"] = point_digest
        return row


def _plain(value: Any) -> Any:
    if isinstance(value, (int, float, str, bool)):
        return value
    import json

    return json.dumps(value, sort_keys=True)


def code_commit(repo_root: Path) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def write_outputs(
    output_dir: Path,
    tensor_files: Mapping[str, TensorFile],
    metric_files: Mapping[str, MetricTable],
    *,
    identity_base: Mapping[str, Any],
) -> dict[str, Path]:
    """Write every accumulated save file under ``output_dir``; returns
    manifest path → absolute path."""
    from safetensors.torch import save_file

    written: dict[str, Path] = {}
    for rel, tensors in tensor_files.items():
        target = output_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        metadata = build_artifact_identity(**identity_base)
        metadata.update(tensors.metadata)
        metadata["entries"] = json.dumps(tensors.entry_meta, sort_keys=True)
        save_file(tensors.entries, str(target), metadata=metadata)
        written[rel] = target
    for rel, table in metric_files.items():
        import pandas as pd

        target = output_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(table.rows).to_parquet(target, index=False)
        written[rel] = target
    return written
