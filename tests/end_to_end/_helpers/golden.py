"""Helpers for goldens (numerical tier) and determinism (property tier).

Defines:

- :func:`load_golden` — read one JSON spec from
  ``tests/end_to_end/goldens/``.
- :func:`run_baseline_for_golden` — compose the golden's runner config
  via :func:`tests.end_to_end._helpers.enumeration.load_e2e_runner_config`,
  pin seed + experiment_root, run it on CPU into ``tmp_path``, and
  return the dict of captured values.
- :func:`extract_values` — read the numerical artifacts a golden cares
  about across the whole output tree (``baseline/accuracy.json`` plus a
  hash + reductions of every ``*.safetensors`` the run produced, including
  the folded analysis-chain goldens) and return them as a flat dict.
- :func:`enumerate_goldens` — list every golden JSON shipped under
  ``tests/end_to_end/goldens/``.

The goldens themselves are produced by
``tests/end_to_end/update_goldens.py``, which calls into the same
helpers — so the numerical / property tiers exercise the same code path
that generated the locked values.

Hardware contract (Phase 2 of the runner-golden refactor): device is
taken verbatim from the golden's pinned model fixture
(``cfg.model.device``) — no runtime device override. The ``unit`` tier
stays CPU-bound; ``smoke``, ``numerical``, and ``property`` follow
whatever the model fixture pins. ``"deterministic": true`` in a golden
JSON is only meaningful on CPU — GPU runs default to
``"deterministic": false`` (per-key reductions inside ``tolerance``
carry the drift signal). See ``docs/TESTS.md`` "Golden JSON contract" for
the policy. ``run_baseline_for_golden`` seeds ``random`` / ``numpy`` /
``torch`` from the JSON's ``seed`` field before composing.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Repo-relative paths.
#
# GOLDENS_DIR holds the locked-value JSONs
#   (``tests/end_to_end/goldens/*.json``).
# GOLDEN_CONFIGS_DIR holds the Hydra-composable runner configs that
#   produce those locked values
#   (``tests/end_to_end/configs/golden/<runner>.yaml``).
GOLDENS_DIR = Path(__file__).resolve().parents[1] / "goldens"
GOLDEN_CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs" / "golden"


@dataclass(frozen=True)
class Golden:
    """One locked baseline: seed, model, tolerances, expected values."""

    runner: str
    seed: int
    model: str
    tolerance: dict[str, float]
    values: dict[str, Any]
    deterministic: bool
    path: Path

    @classmethod
    def from_path(cls, path: Path) -> "Golden":
        with path.open() as f:
            data = json.load(f)
        return cls(
            runner=data["runner"],
            seed=int(data["seed"]),
            model=data["model"],
            tolerance=dict(data.get("tolerance", {"default": 1e-5})),
            values=dict(data["values"]),
            # Default to True — most baselines are deterministic on CPU.
            # Set to False explicitly in the JSON for runners that aren't.
            deterministic=bool(data.get("deterministic", True)),
            path=path,
        )

    def tol_for(self, key: str) -> float:
        """Return the per-key tolerance, falling back to ``default``."""
        return float(self.tolerance.get(key, self.tolerance.get("default", 1e-5)))


def enumerate_goldens(goldens_dir: Path = GOLDENS_DIR) -> list[Path]:
    """Return every ``*.json`` directly under ``tests/end_to_end/goldens/`` (sorted).

    Top-level JSONs only — these are runner-scope, full-pipeline pins.
    Task-scope symbolic pins are not goldens; they live under
    ``tests/tasks/<task>/pinned_samples.json`` and are consumed by
    ``tests/tasks/<task>/test_<task>_numerical.py``.
    """
    if not goldens_dir.is_dir():
        return []
    return sorted(p for p in goldens_dir.glob("*.json"))


def load_golden(path: Path) -> Golden:
    """Read one golden JSON from disk."""
    return Golden.from_path(path)


def load_golden_runner_config(runner: str):
    """Compose a golden runner cfg from ``tests/end_to_end/configs/golden/``.

    Thin wrapper around
    :func:`tests.end_to_end._helpers.enumeration.load_e2e_runner_config`
    that accepts either a bare runner stem (``"mcqa"``) — assumed to live
    in the ``golden`` group — or a fully-qualified ``"<group>/<name>"``
    (kept for symmetry with the smoke tier, which uses
    ``"baseline/<name>"``).

    Returned ``DictConfig`` is fully composed — callers only need to set
    per-test fields (``experiment_root``, ``seed``) before running.
    """
    from tests.end_to_end._helpers.enumeration import load_e2e_runner_config

    name = runner if "/" in runner else f"golden/{runner}"
    return load_e2e_runner_config(name)


def _seed_everything(seed: int) -> None:
    """Seed Python / numpy / torch from the golden's pinned seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(device: Any) -> str:
    """Return the effective device string for a composed model fixture.

    Resolves ``"auto"`` to ``"cuda"`` when CUDA is available and
    ``"cpu"`` otherwise. Concrete strings (``"cpu"``, ``"cuda"``,
    ``"cuda:0"``, …) pass through unchanged.
    """
    s = str(device)
    if s == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return s


def is_gpu_device(device: str) -> bool:
    """True when ``device`` names a GPU (``cuda``, ``cuda:0``, …)."""
    return device.startswith("cuda")


def _hash_safetensors(path: Path) -> str:
    """Return a SHA-256 of the raw safetensors bytes.

    Hashing the file (not the loaded tensor) is intentional — equality
    on the file means equality on every bit including the header dtype
    string, so two runs that drift in dtype despite agreeing on values
    still diverge. For determinism (property tier) that's exactly what
    we want; for numerical-tier within-tolerance comparison we fall
    back to a separate value field if the byte-hash diverges.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _add_tensor_reductions(values: dict[str, Any], key_prefix: str, t: Any) -> None:
    """Emit ``shape`` (always) plus mean/std/first/last reductions for ``t``.

    Reductions are emitted **only when finite** — an empty tensor (e.g.
    ``manifold._sphere_base`` with shape ``(0,)``) or one containing
    NaN/Inf would otherwise pin a ``NaN`` reduction, and the numerical
    gate's ``abs(got - expected) <= tol`` is ``False`` for ``NaN`` even
    across byte-identical reruns. Shape is always safe to pin (it's an
    exact-compared int list), so it still anchors empty/degenerate tensors.
    """
    tf = t.float()
    values[f"{key_prefix}.shape"] = list(t.shape)
    if t.numel() == 0:
        return
    flat = tf.flatten()
    candidates = {
        "mean": float(tf.mean().item()),
        "first": float(flat[0].item()),
        "last": float(flat[-1].item()),
    }
    # std of a single element is NaN (and torch warns about dof<=0); only
    # pin it when there are >=2 elements. Any remaining non-finite reduction
    # (NaN/Inf in the tensor itself) is dropped by the finite check below.
    if t.numel() >= 2:
        candidates["std"] = float(tf.std().item())
    for name, val in candidates.items():
        if math.isfinite(val):
            values[f"{key_prefix}.{name}"] = val


def _add_json_floats(values: dict[str, Any], key_prefix: str, obj: Any) -> None:
    """Recursively pin every finite ``float`` leaf of a JSON-loaded object.

    Walks dicts (by key) and lists (by index), recording each finite float leaf
    as ``key_prefix.<dotted path>``. **Only floats are pinned** — they are the
    continuous metrics an analysis writes to ``results.json`` (e.g. locate's
    ``scores_per_cell`` / ``scores_per_layer`` / ``base_accuracy``) and are
    within-tolerance comparable on GPU ``deterministic: false`` runs. Ints
    (argmax indices like ``best_layer``), strings (token-position ids, variable
    labels), bools, and ``None`` are skipped: they are derived/categorical and
    would be brittle under the exact comparison the numeric gate applies to
    non-floats (a near-tie argmax can flip across same-seed GPU reruns).
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            _add_json_floats(values, f"{key_prefix}.{k}", v)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _add_json_floats(values, f"{key_prefix}.{i}", v)
    elif isinstance(obj, float) and math.isfinite(obj):
        values[key_prefix] = obj


def extract_values(experiment_root: Path) -> dict[str, Any]:
    """Read the numerical artifacts a runner produced, across all analyses.

    Returns a flat dict of pinnable values. Three layers:

    * **Legacy baseline keys** (preserved verbatim so the baseline goldens'
      tuned tolerances keep matching): ``baseline/accuracy.json`` fields
      become ``accuracy.<field>``; ``baseline/full_output_dists.safetensors``
      becomes ``full_output_dists.{sha256,shape,mean,std,first,last}`` for
      its ``dists`` tensor.
    * **Multi-analysis tree walk**: every *other* ``*.safetensors`` under
      ``experiment_root`` (e.g. ``output_manifold/per_example_output_dists``,
      the subspace/manifold/pullback chain) is summarised as a file
      ``<relpath>.sha256`` plus per-tensor ``<relpath>.<tensor>.{shape,mean,
      std,first,last}`` reductions. This is what lets the folded analysis
      goldens (``output_manifold``, ``pullback``) pin scalars rather than
      only assert artifact existence.
    * **results.json float scores**: every finite ``float`` leaf of each
      ``results.json`` (e.g. locate's ``scores_per_cell`` / ``scores_per_layer``
      / ``base_accuracy``) becomes ``<relpath>.<dotted-json-path>``. These
      scores are written as JSON, not safetensors, so the tree walk misses
      them — without this layer a pairwise-mode locate golden would pin only
      run completion + baseline numerics, not the scores themselves. See
      :func:`_add_json_floats` for the float-only rationale.

    The SHA-256 keys carry byte-identity for the property (determinism)
    tier; the reductions carry within-tolerance equality for the numerical
    tier on ``"deterministic": false`` (GPU) runners. Non-finite reductions
    are dropped (see :func:`_add_tensor_reductions`).

    Adding a new artifact: it is picked up automatically if it is a
    ``*.safetensors``; the test harness iterates ``golden.values`` so any
    new key flows through after a re-pin.
    """
    from safetensors.torch import load_file

    root = Path(experiment_root)
    values: dict[str, Any] = {}

    # --- Legacy baseline keys (unprefixed, byte-for-byte compatible) ---
    base_dir = root / "baseline"
    acc_path = base_dir / "accuracy.json"
    if acc_path.is_file():
        with acc_path.open() as f:
            acc = json.load(f)
        for k, v in acc.items():
            # Skip metrics the analysis reports as ``None`` — e.g.
            # ``prob_accuracy`` on a multi-token task (max_new_tokens > 1).
            # A ``null`` carries no within-tolerance signal (docs/TESTS.md
            # "Numeric contract"); dropping the key is the sanctioned path.
            if v is None:
                continue
            values[f"accuracy.{k}"] = v

    legacy_dists = base_dir / "full_output_dists.safetensors"
    if legacy_dists.is_file():
        values["full_output_dists.sha256"] = _hash_safetensors(legacy_dists)
        tensors = load_file(str(legacy_dists))
        if "dists" in tensors:
            t = tensors["dists"].float()
            values["full_output_dists.shape"] = list(t.shape)
            values["full_output_dists.mean"] = float(t.mean().item())
            values["full_output_dists.std"] = float(t.std().item())
            # First / last entry for a coarse identity check that catches
            # index-shift bugs without locking the whole tensor.
            flat = t.flatten()
            values["full_output_dists.first"] = float(flat[0].item())
            values["full_output_dists.last"] = float(flat[-1].item())

    # --- Multi-analysis tree walk over every other *.safetensors ---
    # Sorted for deterministic key ordering; the legacy baseline dists file
    # is already captured above under its unprefixed keys, so skip it here.
    for sft_path in sorted(root.rglob("*.safetensors")):
        if sft_path == legacy_dists:
            continue
        prefix = sft_path.relative_to(root).with_suffix("").as_posix().replace("/", ".")
        values[f"{prefix}.sha256"] = _hash_safetensors(sft_path)
        try:
            tensors = load_file(str(sft_path))
        except Exception:
            # A safetensors we can't load (unexpected dtype, partial write)
            # still contributes its file hash above; skip its reductions
            # rather than failing the whole extraction.
            continue
        for name, t in tensors.items():
            _add_tensor_reductions(values, f"{prefix}.{name}", t)

    # --- results.json float scores across every analysis ---
    # Analyses such as ``locate`` write their per-cell / per-layer scores to a
    # ``results.json`` (not a safetensors), so the tree walk above misses them.
    # Pin every finite float leaf of each ``results.json`` so those scores are
    # value-checked (notably the pairwise-mode locate scores). Matched by exact
    # filename, so ``accuracy.json`` (captured above) and ``metadata.json``
    # (config echo) are intentionally not picked up.
    for results_path in sorted(root.rglob("results.json")):
        prefix = (
            results_path.relative_to(root).with_suffix("").as_posix().replace("/", ".")
        )
        try:
            with results_path.open() as f:
                payload = json.load(f)
        except Exception:
            # A malformed/partial results.json contributes nothing rather than
            # failing the whole extraction (mirrors the safetensors guard above).
            continue
        _add_json_floats(values, prefix, payload)

    return values


def run_baseline_for_golden(
    golden: Golden,
    experiment_root: Path,
) -> tuple[dict[str, Any], str]:
    """Run one golden baseline under the pinned seed.

    Composes the golden's runner config from
    ``tests/end_to_end/configs/golden/<runner>.yaml`` (no runtime model
    overrides — the yaml itself pins ``/model: <fixture>`` and the
    dataset/batch_size knobs at golden scale), then sets per-test
    fields (``experiment_root``, ``seed``) and dispatches via
    ``_run_steps``. Device follows ``cfg.model.device`` (the pinned
    fixture); ``"auto"`` resolves to CUDA when available, else CPU.
    Caller is responsible for providing a fresh ``experiment_root``
    (typically ``tmp_path``).

    Returns ``(values, device)`` — the captured artifact dict plus the
    resolved device string. ``update_goldens.py`` uses ``device`` to
    default ``deterministic: false`` when bootstrapping a golden on
    GPU.
    """
    _seed_everything(golden.seed)

    # Imports are deferred so this module is import-safe at collection
    # time even when the runner stack isn't on the path yet.
    from causalab.runner.run_exp import _run_steps

    cfg = load_golden_runner_config(golden.runner)
    # The two per-test fields that can't be pinned in the yaml: the
    # output directory (tmp_path) and the seed (set by the JSON, not
    # the config). Everything else — model, task knobs, batch_size —
    # is already locked by the golden config.
    cfg.experiment_root = str(experiment_root)
    cfg.seed = golden.seed

    _run_steps(cfg)
    device = resolve_device(cfg.model.device)
    return extract_values(experiment_root), device
