"""Run outputs: the save manifest written to disk, stamped (spec §2.12, §8).

Tensors (reads, featurizer bundles) land in ``.safetensors`` with the
point's ``ArtifactIdentity`` in the header; per-example metric tables land
in ``.parquet``. In swept documents the authored ``file_path`` is
unchanged: axis coordinates become columns of the metric tables and key
suffixes on tensor entries (``rot[k=8,seed=0]``)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Mapping

import torch

from causalab.protocol.resolve import build_artifact_identity
from causalab.protocol.sweep import coordinate_label

__all__ = ["MetricTable", "TensorFile", "write_outputs"]


class TensorFile:
    """Accumulates tensor entries for one save file across points."""

    def __init__(self) -> None:
        self.entries: dict[str, torch.Tensor] = {}
        self.metadata: dict[str, str] = {}

    def add(self, name: str, value: Any, coords: Mapping[str, Any]) -> None:
        key = f"{name}{coordinate_label(coords, entry=name)}" if coords else name
        from causalab.neural.pytorch_hooks.executor import RaggedValue

        if isinstance(value, RaggedValue):
            # ragged reads persist as the flat gather + per-row widths
            self.entries[key] = value.flat.detach().to("cpu").contiguous()
            self.entries[f"{key}.widths"] = torch.tensor(value.widths, dtype=torch.long)
            return
        self.entries[key] = value.detach().to("cpu").contiguous()


class MetricTable:
    """Accumulates per-example metric rows for one save file across points."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(
        self, name: str, values: list[Any], coords: Mapping[str, Any], point_digest: str
    ) -> None:
        for example, value in enumerate(values):
            row: dict[str, Any] = {"example": example, "metric": name}
            if isinstance(value, dict):
                import json

                row["value"] = json.dumps(value, sort_keys=True)
            else:
                row["value"] = value
            row.update({axis: _plain(coord) for axis, coord in coords.items()})
            row["produced_by"] = point_digest
            self.rows.append(row)


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
        save_file(tensors.entries, str(target), metadata=metadata)
        written[rel] = target
    for rel, table in metric_files.items():
        import pandas as pd

        target = output_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(table.rows).to_parquet(target, index=False)
        written[rel] = target
    return written
