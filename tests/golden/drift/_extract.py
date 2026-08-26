"""Value extraction for the chat-coherent drift tier.

One code path shared by the capture script (update_drift_goldens.py) and
the replay test, so pins and assertions are guaranteed to reduce the run
outputs identically — the structural idea inherited from the retired
tier's extract_values.

Keys:
- ``interchange.<metric>.mean`` (and ``.std`` for the continuous metric)
  from the point document's parquets;
- ``interchange.acts_mid.<stat>`` — mean/std/first/last/shape of the
  harvested residual tensor;
- ``scan.iia.<axis-label>.mean`` — per-layer IIA means from the swept
  document's single parquet (axis coordinates are row columns).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from causalab.protocol.cli import main

from tests.golden._env import FIXTURES, GOLDEN_PROTOCOLS

DOCS = ("drift_interchange_im.json", "drift_locate_scan_im.json")
ACCURACY_GATE = 0.9  # the old tier's baseline gate, kept verbatim
PINS = Path(__file__).parent / "drift_goldens.json"

_META_COLUMNS = {"value", "example", "point", "produced_by", "metric", "name"}


def run_drift_documents(out_root: Path, device: str) -> dict[str, Path]:
    dirs: dict[str, Path] = {}
    for name in DOCS:
        out = out_root / name.removesuffix("_im.json")
        argv = [
            "run",
            str(GOLDEN_PROTOCOLS / name),
            "--data-root",
            str(FIXTURES / "data"),
            "--artifacts-root",
            str(out / "artifacts"),
            "--out",
            str(out),
            "--device",
            device,
        ]
        code = main(argv)
        if code != 0:
            raise RuntimeError(f"{name} failed (exit {code})")
        dirs[name] = out
    return dirs


def _tensor_stats(path: Path, prefix: str, values: dict[str, Any]) -> None:
    from safetensors.torch import load_file

    for name, tensor in load_file(str(path)).items():
        flat = tensor.float().flatten()
        values[f"{prefix}.{name}.mean"] = float(flat.mean())
        values[f"{prefix}.{name}.std"] = float(flat.std())
        values[f"{prefix}.{name}.first"] = float(flat[0])
        values[f"{prefix}.{name}.last"] = float(flat[-1])
        values[f"{prefix}.{name}.shape"] = list(tensor.shape)


def extract_values(dirs: dict[str, Path]) -> dict[str, Any]:
    values: dict[str, Any] = {}

    point = dirs["drift_interchange_im.json"]
    for metric in ("acc", "iia", "ld"):
        column = pd.read_parquet(point / f"{metric}.parquet")["value"]
        values[f"interchange.{metric}.mean"] = float(column.mean())
    values["interchange.ld.std"] = float(
        pd.read_parquet(point / "ld.parquet")["value"].std()
    )
    _tensor_stats(point / "acts_mid.safetensors", "interchange", values)

    scan = pd.read_parquet(dirs["drift_locate_scan_im.json"] / "iia.parquet")
    axes = [c for c in scan.columns if c not in _META_COLUMNS]
    for coords, group in scan.groupby(axes):
        coords = coords if isinstance(coords, tuple) else (coords,)
        label = ",".join(f"{a}={c}" for a, c in zip(axes, coords))
        values[f"scan.iia.{label}.mean"] = float(group["value"].mean())
    return values


def load_pins(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def compare(
    pinned: dict[str, Any], measured: dict[str, Any], tolerance: dict[str, Any]
) -> list[str]:
    """Human-readable mismatches; empty means the replay is within pins."""
    default = float(tolerance.get("default", 0.0))
    problems = []
    for key in sorted(set(pinned) | set(measured)):
        if key not in pinned or key not in measured:
            problems.append(f"{key}: only in {'pins' if key in pinned else 'run'}")
            continue
        want, got = pinned[key], measured[key]
        if isinstance(want, list):
            if want != got:
                problems.append(f"{key}: shape {got} != pinned {want}")
        else:
            tol = float(tolerance.get(key, default))
            if abs(float(got) - float(want)) > tol:
                problems.append(f"{key}: {got:.6g} != pinned {want:.6g} (tol {tol})")
    return problems
