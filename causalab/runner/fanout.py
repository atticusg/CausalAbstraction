"""Cluster-agnostic GPU fan-out for arbitrary experiments.

Decompose one experiment into many independent shards along arbitrary Hydra
override axes (e.g. one job per ``layer``, or per ``layer × token`` — the widest,
shortest set of jobs), fan those shards out over many GPUs, then recombine scan
results into a single artifact.

The orchestrator never interprets axis names. It builds the cartesian product of
the declared override axes into a manifest, then runs each shard by shelling out
to ``scripts/run_exp.sh`` — inheriting that script's runner discovery, session
detection, ``experiment_root`` routing, session-local code injection, and config
snapshotting for free. No analysis changes are required: each shard writes under
its own ``--experiment-root <base>/shards/<id>`` so outputs never collide.

Two submission backends sit behind one interface:

* **slurm** — write the manifest, submit ONE array job ``0-(N-1)%CAP`` via
  ``scripts/fanout_array.sbatch`` (each task reads its row and execs
  ``run_exp.sh``). Resources default to the runner config's GPU count / walltime;
  per-site ``--partition``/``--account``/``--qos`` come from config/env and are
  omitted entirely when unset (org policy). Waiting uses ``monitor_jobs``.
* **local** — no scheduler: run shards as subprocesses across visible GPUs,
  one shard per GPU (round-robin ``CUDA_VISIBLE_DEVICES``).

Examples::

    # Param-grid sweep (one shard per k_features value)
    uv run python -m causalab.runner.fanout weekdays/weekdays_8b_subspace \
        --axis 'subspace.k_features=8,16,32,64' --cap 8 --wait --collect

    # Per layer × token scan (32 layers × 3 tokens = 96 one-GPU shards)
    uv run python -m causalab.runner.fanout age/age_8b_pipeline \
        --axis-each 'locate.layers=range(0,32)' \
        --axis 'locate.token_positions=[subject],[verb],[object]' \
        --const 'locate.method=interchange' \
        --gpus 1 --cap 96 --time 00:20:00 --wait --collect
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from queue import Queue
from typing import Any

# fanout.py lives at <repo>/causalab/runner/fanout.py
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_EXP = REPO_ROOT / "scripts" / "run_exp.sh"
ARRAY_SBATCH = REPO_ROOT / "scripts" / "fanout_array.sbatch"
CLUSTER_CONFIG_DIR = REPO_ROOT / "causalab" / "configs" / "cluster"
SHARD_ID_WIDTH = 5

# Underscore-prefixed so artifact viewers / interpret skip it (same convention as
# the runner's other internal dirs).
FANOUT_DIR_NAME = "_fanout"


# --------------------------------------------------------------------------- #
# Spec parsing
# --------------------------------------------------------------------------- #
def _split_top_level(spec: str) -> list[str]:
    """Split on commas that are NOT inside ``[]`` brackets.

    ``'8,16,32'`` -> ``['8','16','32']``;
    ``'[subject],[verb]'`` -> ``['[subject]','[verb]']``;
    ``'[a,b],[c,d]'`` -> ``['[a,b]','[c,d]']`` (the inner commas are preserved).
    """
    parts: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in spec:
        if ch == "[":
            depth += 1
            cur.append(ch)
        elif ch == "]":
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append("".join(cur).strip())
    return [p for p in parts if p != ""]


_RANGE_RE = re.compile(r"^range\(\s*(-?\d+)\s*,\s*(-?\d+)\s*(?:,\s*(-?\d+)\s*)?\)$")


def _expand_scalar_spec(spec: str) -> list[str]:
    """Expand a scalar value spec: ``range(a,b[,step])`` or a comma list."""
    m = _RANGE_RE.match(spec.strip())
    if m:
        start, stop = int(m.group(1)), int(m.group(2))
        step = int(m.group(3)) if m.group(3) is not None else 1
        if step == 0:
            raise ValueError(f"range step must be non-zero: {spec!r}")
        return [str(v) for v in range(start, stop, step)]
    return _split_top_level(spec)


def _parse_kv(arg: str, *, flag: str) -> tuple[str, str]:
    """Split ``KEY=VALUE`` (only on the first ``=``)."""
    if "=" not in arg:
        raise ValueError(f"{flag} expects KEY=VALUE, got {arg!r}")
    key, value = arg.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError(f"{flag} has an empty key: {arg!r}")
    return key, value


def _parse_axis(arg: str) -> tuple[str, list[str]]:
    """``--axis 'K=v1,v2'`` -> ``(K, ['v1','v2'])`` (values used verbatim).

    ``range(a,b[,step])`` is expanded to its scalar values, so e.g.
    ``--axis 'exploration.pair.index=range(0,12)'`` yields the 12 scalar indices
    (use this for fields that want a scalar; use ``--axis-each`` for fields that
    want a singleton list).
    """
    key, value = _parse_kv(arg, flag="--axis")
    values = _expand_scalar_spec(value)
    if not values:
        raise ValueError(f"--axis {arg!r} produced no values")
    return key, values


def _parse_axis_each(arg: str) -> tuple[str, list[str]]:
    """``--axis-each 'K=range(0,4)'`` -> ``(K, ['[0]','[1]','[2]','[3]'])``.

    Each scalar is wrapped in a singleton list so an analysis dimension that is
    normally iterated internally (layers, heads, …) becomes one shard per value.
    """
    key, value = _parse_kv(arg, flag="--axis-each")
    scalars = _expand_scalar_spec(value)
    if not scalars:
        raise ValueError(f"--axis-each {arg!r} produced no values")
    return key, [f"[{s}]" for s in scalars]


def load_spec(
    args: argparse.Namespace,
) -> tuple[str, list[tuple[str, list[str]]], dict[str, str], dict[str, Any]]:
    """Resolve runner, axes, and constants from a YAML ``--spec`` and/or flags.

    CLI flags are layered on top of (and override) the YAML spec. Returns
    ``(runner, axes, consts, spec_meta)`` where ``axes`` is an ordered list of
    ``(key, [values...])`` and ``consts`` is a flat override map.
    """
    runner: str | None = args.runner
    axes: list[tuple[str, list[str]]] = []
    consts: dict[str, str] = {}
    meta: dict[str, Any] = {}

    if args.spec:
        from omegaconf import OmegaConf

        spec = OmegaConf.to_container(OmegaConf.load(args.spec), resolve=True)
        if not isinstance(spec, dict):
            raise ValueError(f"--spec {args.spec} must be a mapping")
        runner = runner or spec.get("runner")
        meta.update({k: spec[k] for k in ("backend", "cap", "base") if k in spec})
        for entry in spec.get("axes", []) or []:
            key = entry["key"]
            if "values" in entry:
                axes.append((key, [str(v) for v in entry["values"]]))
            elif "each" in entry:
                axes.append((key, [f"[{v}]" for v in entry["each"]]))
            else:
                raise ValueError(f"axis {key!r} needs 'values' or 'each'")
        for key, value in (spec.get("constant_overrides", {}) or {}).items():
            consts[key] = str(value)

    for arg in args.axis or []:
        axes.append(_parse_axis(arg))
    for arg in args.axis_each or []:
        axes.append(_parse_axis_each(arg))
    for arg in args.const or []:
        key, value = _parse_kv(arg, flag="--const")
        consts[key] = value

    if not runner:
        raise SystemExit("error: no runner given (positional arg or spec.runner)")
    if not axes:
        raise SystemExit("error: no axes given (--axis / --axis-each / spec.axes)")
    return runner, axes, consts, meta


def build_shards(
    axes: list[tuple[str, list[str]]], consts: dict[str, str]
) -> list[dict[str, str]]:
    """Cartesian product of the axes, merged with the constant overrides.

    Returns one override map per shard, in row-major order over ``axes``.
    """
    shards: list[dict[str, str]] = [dict(consts)]
    for key, values in axes:
        shards = [{**row, key: v} for row in shards for v in values]
    return shards


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
def _shard_id(index: int) -> str:
    return str(index).zfill(SHARD_ID_WIDTH)


def write_manifest(
    base: Path, shards: list[dict[str, str]], spec_meta: dict[str, Any]
) -> Path:
    """Write ``<base>/_fanout/manifest.jsonl`` and ``spec.resolved.yaml``."""
    fanout_dir = base / FANOUT_DIR_NAME
    fanout_dir.mkdir(parents=True, exist_ok=True)
    manifest = fanout_dir / "manifest.jsonl"
    with open(manifest, "w") as f:
        for i, overrides in enumerate(shards):
            f.write(
                json.dumps({"shard_id": _shard_id(i), "overrides": overrides}) + "\n"
            )

    from omegaconf import OmegaConf

    with open(fanout_dir / "spec.resolved.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(OmegaConf.create(spec_meta)))
    return manifest


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def overrides_to_tokens(overrides: dict[str, str]) -> list[str]:
    """``{'a':'1','b':'[x]'}`` -> ``['a=1', 'b=[x]']`` (Hydra override tokens)."""
    return [f"{k}={v}" for k, v in overrides.items()]


def emit_row(manifest_path: str, index: int) -> None:
    """Print a shard's id then one ``key=value`` override per line.

    Consumed by ``scripts/fanout_array.sbatch`` (via ``mapfile``) so JSONL ->
    CLI-override translation and list-literal handling live in one place.
    """
    rows = read_manifest(manifest_path)
    row = rows[index]
    print(row["shard_id"])
    for token in overrides_to_tokens(row["overrides"]):
        print(token)


# --------------------------------------------------------------------------- #
# Runner config / session resolution
# --------------------------------------------------------------------------- #
def _session_code_dir(base: Path) -> Path | None:
    """If ``base`` lives under ``agent_logs/<session>/artifacts``, return the
    session's ``code/`` dir when present (so session-local runners resolve).

    See ``causalab/runner/README.md`` "Session-local code injection".
    """
    parts = base.parts
    if "agent_logs" in parts and "artifacts" in parts:
        idx = parts.index("artifacts")
        session_dir = Path(*parts[:idx])
        code = session_dir / "code"
        if code.is_dir():
            return code
    return None


def compose_runner_cfg(runner: str, extra_config_dirs: list[Path]):
    """Compose a runner's resolved Hydra config (mirrors slurm_args.py).

    Picks whichever config dir actually contains the runner as the primary path
    and puts the rest on ``hydra.searchpath`` so nested defaults still resolve.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    import causalab

    shipped = Path(causalab.__file__).parent / "configs"
    candidates = [d for d in extra_config_dirs] + [shipped]
    primary = next((d for d in candidates if (d / f"{runner}.yaml").is_file()), shipped)
    others = [d for d in candidates if d != primary]
    overrides: list[str] = []
    if others:
        paths = ",".join(f"file://{p}" for p in others)
        overrides.append(f"hydra.searchpath=[{paths}]")

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(primary), version_base=None):
        return compose(config_name=runner, overrides=overrides)


def resolve_runner(runner: str, base: Path) -> str:
    """Normalize a runner basename to the Hydra config name (``runners/...``).

    Mirrors the discovery in ``scripts/run_exp.sh`` so users can pass a bare
    basename. Searches shipped configs then the session's ``code/configs``.
    """
    runner = runner.removeprefix("runners/")
    search_dirs = [REPO_ROOT / "causalab" / "configs"]
    code = _session_code_dir(base)
    if code is not None:
        search_dirs.append(code / "configs")
    for cfg_dir in search_dirs:
        if (cfg_dir / f"{runner}.yaml").is_file():
            return runner
        if (cfg_dir / "runners" / f"{runner}.yaml").is_file():
            return f"runners/{runner}"
        matches = (
            list((cfg_dir / "runners").rglob(f"{runner}.yaml"))
            if (cfg_dir / "runners").is_dir()
            else []
        )
        if len(matches) == 1:
            return (
                f"runners/{matches[0].relative_to(cfg_dir / 'runners').with_suffix('')}"
            )
    # Fall through: let Hydra error with its own message later.
    return runner


# --------------------------------------------------------------------------- #
# Cluster config / sbatch directives
# --------------------------------------------------------------------------- #
def resolve_cluster_directives(args: argparse.Namespace) -> list[str]:
    """Build per-site sbatch directives.

    Resolution order (highest first): env var > CLI flag > ``cluster/<name>.yaml``
    > ``cluster/default.yaml`` (all null). An unset value means OMIT the directive
    entirely — never pass an empty ``--partition`` (org policy: the cluster
    default is correct).
    """
    from omegaconf import OmegaConf

    cfg: dict[str, Any] = {}
    default = CLUSTER_CONFIG_DIR / "default.yaml"
    if default.is_file():
        cfg.update(OmegaConf.to_container(OmegaConf.load(default), resolve=True) or {})
    if args.cluster:
        site = CLUSTER_CONFIG_DIR / f"{args.cluster}.yaml"
        if not site.is_file():
            raise SystemExit(f"error: cluster config not found: {site}")
        cfg.update(OmegaConf.to_container(OmegaConf.load(site), resolve=True) or {})

    def pick(name: str, env: str, cli: str | None) -> str | None:
        return os.environ.get(env) or cli or cfg.get(name)

    directives: list[str] = []
    partition = pick("partition", "CAUSALAB_SLURM_PARTITION", args.partition)
    account = pick("account", "CAUSALAB_SLURM_ACCOUNT", args.account)
    qos = pick("qos", "CAUSALAB_SLURM_QOS", args.qos)
    if partition:
        directives.append(f"--partition={partition}")
    if account:
        directives.append(f"--account={account}")
    if qos:
        directives.append(f"--qos={qos}")
    for extra in cfg.get("extra_sbatch", []) or []:
        directives.append(str(extra))
    return directives


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
def detect_backend(arg: str) -> str:
    if arg == "auto":
        return "slurm" if shutil.which("sbatch") else "local"
    return arg


def _visible_gpu_ids() -> list[str]:
    """Resolve the list of GPU ids to spread local shards across."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        return [g.strip() for g in env.split(",") if g.strip()]
    nvidia = shutil.which("nvidia-smi")
    if nvidia:
        try:
            out = subprocess.run(
                [nvidia, "-L"], capture_output=True, text=True, check=True
            ).stdout
            n = sum(1 for line in out.splitlines() if line.strip())
            if n:
                return [str(i) for i in range(n)]
        except subprocess.CalledProcessError:
            pass
    try:
        import torch

        n = torch.cuda.device_count()
        if n:
            return [str(i) for i in range(n)]
    except Exception:
        pass
    return ["0"]


def _shard_command(runner: str, base: Path, shard_id: str, overrides: dict[str, str]):
    return [
        str(RUN_EXP),
        "--experiment-root",
        str(base / "shards" / shard_id),
        runner,
        *overrides_to_tokens(overrides),
    ]


def submit_local(
    runner: str,
    base: Path,
    shards: list[dict[str, str]],
    manifest: Path,
    cap: int,
) -> int:
    """Run shards as subprocesses, one per visible GPU (round-robin)."""
    gpu_ids = _visible_gpu_ids()
    pool = min(len(gpu_ids), cap)
    print(
        f"+ local backend: {len(shards)} shards across {pool} GPU(s) "
        f"({','.join(gpu_ids[:pool])})",
        file=sys.stderr,
    )

    work: Queue[int] = Queue()
    for i in range(len(shards)):
        work.put(i)
    failures: list[tuple[int, int]] = []
    lock = threading.Lock()

    def worker(gpu: str) -> None:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
        while True:
            try:
                i = work.get_nowait()
            except Exception:
                return
            shard_id = _shard_id(i)
            cmd = _shard_command(runner, base, shard_id, shards[i])
            print(f"  [gpu {gpu}] shard {shard_id}: {' '.join(cmd)}", file=sys.stderr)
            try:
                rc = subprocess.run(cmd, cwd=REPO_ROOT, env=env).returncode
            except Exception as exc:
                # A launch failure (e.g. run_exp.sh missing / not executable) must
                # still be recorded as a shard failure — otherwise the exception
                # kills this worker thread, the shard vanishes from the failure
                # report, and the pool silently shrinks.
                print(f"! shard {shard_id} could not launch: {exc}", file=sys.stderr)
                rc = 127
            if rc != 0:
                with lock:
                    failures.append((i, rc))
            work.task_done()

    threads = [
        threading.Thread(target=worker, args=(gpu_ids[i],), daemon=True)
        for i in range(pool)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if failures:
        for i, rc in sorted(failures):
            print(f"! shard {_shard_id(i)} failed (exit {rc})", file=sys.stderr)
        return 1
    print(f"+ all {len(shards)} shards completed", file=sys.stderr)
    return 0


def submit_slurm(
    runner: str,
    base: Path,
    shards: list[dict[str, str]],
    manifest: Path,
    cap: int,
    gpus: int,
    time: str,
    job_name: str,
    cluster_directives: list[str],
    session_code: Path | None,
    wait: bool,
) -> tuple[int, str | None, bool]:
    """Submit ONE array job covering all shards.

    Returns ``(exit_code, array_id, waited)``. ``waited`` is True only when this
    call actually blocked until the array finished; the caller must not run an
    inline ``--collect`` unless it did, else it collects against shards the job
    has not populated yet.
    """
    n = len(shards)
    (REPO_ROOT / "slurm_logs").mkdir(exist_ok=True)
    array_spec = f"0-{n - 1}%{cap}"

    sb_args = [
        "sbatch",
        f"--array={array_spec}",
        f"--gres=gpu:{gpus}",
        f"--time={time}",
        f"--job-name={job_name}",
        *cluster_directives,
    ]
    if session_code is not None:
        session_dir = session_code.parent
        sb_args.append(f"--export=ALL,CAUSALAB_SESSION_CODE={session_dir}")

    cmd = [
        *sb_args,
        str(ARRAY_SBATCH),
        str(REPO_ROOT),
        str(base),
        runner,
        str(manifest),
    ]
    print(
        f"+ logs: {REPO_ROOT}/slurm_logs/{job_name}_<jobid>_<idx>.{{out,err}}",
        file=sys.stderr,
    )
    print(f"+ {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        return result.returncode, None, False
    # "Submitted batch job 12345"
    m = re.search(r"(\d+)", result.stdout)
    array_id = m.group(1) if m else None
    print(result.stdout.strip(), file=sys.stderr)
    if array_id is None:
        print("! could not parse array job id from sbatch output", file=sys.stderr)
        return 1, None, False

    if wait:
        monitor = shutil.which("monitor_jobs")
        if monitor:
            print(f"+ waiting via monitor_jobs {array_id}", file=sys.stderr)
            rc = subprocess.run([monitor, array_id], cwd=REPO_ROOT).returncode
            return rc, array_id, True
        # Could not actually block. Report waited=False so the caller treats this
        # exactly like a detached submit (skip inline --collect, print the
        # --collect-only recovery) instead of collecting against shards the array
        # job has not populated yet.
        print(
            f"! monitor_jobs not on PATH; cannot --wait. Job {array_id} submitted "
            "detached.",
            file=sys.stderr,
        )
        return 0, array_id, False
    print(f"+ wait with: monitor_jobs {array_id}", file=sys.stderr)
    return 0, array_id, False


# --------------------------------------------------------------------------- #
# Collect / merge
# --------------------------------------------------------------------------- #
def _merge_scores_results(jsons: list[dict[str, Any]]) -> dict[str, Any]:
    """Union the ``scores_per_cell`` of per-shard ``results.json`` files.

    Cell keys are ``"<layer>|<pos_id>"`` and are disjoint across shards by
    construction, so the union is lossless. ``best_cell`` / ``best_layer`` /
    ``scores_per_layer`` / ``token_position_ids`` are recomputed from the union,
    reconstituting exactly what a single full scan would have written.
    """
    merged_cells: dict[str, float] = {}
    token_ids: list[str] = []
    base = dict(jsons[0])
    for data in jsons:
        merged_cells.update(data.get("scores_per_cell", {}))
        for pos in data.get("token_position_ids", []):
            if pos not in token_ids:
                token_ids.append(pos)

    best_key = (
        max(merged_cells, key=lambda k: merged_cells[k]) if merged_cells else None
    )
    best_layer = best_pos = None
    if best_key is not None:
        best_layer_s, best_pos = best_key.split("|", 1)
        best_layer = int(best_layer_s)

    # scores_per_layer = max cell score within each layer.
    per_layer: dict[str, float] = {}
    for key, score in merged_cells.items():
        layer = key.split("|", 1)[0]
        per_layer[layer] = max(score, per_layer.get(layer, float("-inf")))

    base.update(
        {
            "scores_per_cell": merged_cells,
            "scores_per_layer": per_layer,
            "token_position_ids": token_ids,
            "best_cell": (
                {"layer": best_layer, "token_position": best_pos}
                if best_key is not None
                else None
            ),
            "best_layer": best_layer,
        }
    )
    return base


def _render_locate_heatmap(merged: dict[str, Any], dest_dir: Path, title: str) -> None:
    """Best-effort recombined heatmap from merged ``scores_per_cell``."""
    try:
        from causalab.io.plots.score_heatmap import plot_residual_stream_heatmap

        cells = merged.get("scores_per_cell", {})
        if not cells:
            return
        scores = {}
        layers_set = set()
        for key, score in cells.items():
            layer_s, pos = key.split("|", 1)
            scores[(int(layer_s), pos)] = score
            layers_set.add(int(layer_s))
        plot_residual_stream_heatmap(
            scores=scores,
            layers=sorted(layers_set),
            token_position_ids=merged.get("token_position_ids", []),
            title=title,
            save_path=str(dest_dir / "heatmap.png"),
            figure_format="png",
        )
    except Exception as exc:  # mirror locate's own defensive rendering
        print(f"! heatmap render failed for {dest_dir}: {exc}", file=sys.stderr)


def _varied_keys(rows: list[dict[str, Any]]) -> list[str]:
    """Override keys whose value differs across shards (used for suffix names)."""
    keys = {k for r in rows for k in r["overrides"]}
    return sorted(k for k in keys if len({r["overrides"].get(k) for r in rows}) > 1)


def _suffix_for(overrides: dict[str, str], varied: list[str]) -> str:
    """Filename-safe suffix from the axis values that distinguish a shard.

    Path-like values collapse to their basename stem so a long file path doesn't
    bloat the name; an over-long result returns empty so the caller falls back to
    the shard id.
    """

    def clean(v: str) -> str:
        if "/" in v:
            v = os.path.splitext(os.path.basename(v))[0]
        return re.sub(r"[^A-Za-z0-9._-]+", "-", v).strip("-")

    suffix = ".".join(
        f"{k.split('.')[-1]}-{clean(overrides[k])}" for k in varied if k in overrides
    )
    return suffix if 0 < len(suffix) <= 80 else ""


def collect(base: Path) -> int:
    """Recombine ``<base>/shards/*`` into the canonical layout under ``<base>``.

    Files written by a single shard are copied straight over. Files written by
    multiple shards are merged when they carry ``scores_per_cell`` (scan output)
    and otherwise copied side by side with an axis-value suffix so nothing is
    silently overwritten. Only writes under ``<base>``; shard outputs are read-only.
    """
    shards_root = base / "shards"
    if not shards_root.is_dir():
        print(f"! no shards dir at {shards_root}; nothing to collect", file=sys.stderr)
        return 1

    rows_by_id: dict[str, dict[str, Any]] = {}
    manifest = base / FANOUT_DIR_NAME / "manifest.jsonl"
    if manifest.is_file():
        rows_by_id = {r["shard_id"]: r for r in read_manifest(manifest)}
    varied = _varied_keys(list(rows_by_id.values())) if rows_by_id else []

    shard_dirs = sorted(d for d in shards_root.iterdir() if d.is_dir())
    # Group every produced file by its path relative to its shard root.
    groups: dict[Path, list[tuple[str, Path]]] = {}
    for sd in shard_dirs:
        for f in sd.rglob("*"):
            if not f.is_file():
                continue
            rel = f.relative_to(sd)
            if rel.parts and rel.parts[0] == FANOUT_DIR_NAME:
                continue
            groups.setdefault(rel, []).append((sd.name, f))

    # Merge results.json before its directory siblings: the heatmap.png dedup
    # guard below checks for the *written* merged results.json, so it must run
    # after that file exists. Plain filename sort puts 'heatmap.png' < 'results.json'
    # and the guard would never fire, leaking stray per-shard heatmap copies.
    def _merge_first(rel: Path) -> tuple[str, bool, str]:
        return (rel.parent.as_posix(), rel.name != "results.json", rel.name)

    merged_count = collisions = copied = 0
    for rel, items in sorted(groups.items(), key=lambda kv: _merge_first(kv[0])):
        dest = base / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        if len(items) == 1:
            shutil.copy2(items[0][1], dest)
            copied += 1
            continue

        # results.json with scores_per_cell -> merge.
        if rel.name == "results.json":
            jsons = [json.loads(p.read_text()) for _id, p in items]
            if all("scores_per_cell" in d for d in jsons):
                merged = _merge_scores_results(jsons)
                dest.write_text(json.dumps(merged, indent=2))
                _render_locate_heatmap(
                    merged, dest.parent, title=f"Locate: {rel.parent.name}"
                )
                merged_count += 1
                continue

        # metadata.json -> union the layer/token lists, keep the rest.
        if rel.name == "metadata.json":
            metas = [json.loads(p.read_text()) for _id, p in items]
            meta = dict(metas[0])
            layers: list[int] = []
            for m in metas:
                for layer in m.get("layers", []) or []:
                    if layer not in layers:
                        layers.append(layer)
            if layers:
                meta["layers"] = sorted(layers)
            dest.write_text(json.dumps(meta, indent=2))
            merged_count += 1
            continue

        # Heatmap already re-rendered from the merged scores; skip the per-shard ones.
        if rel.name == "heatmap.png" and (dest.parent / "results.json").exists():
            continue

        # Fallback: copy each shard's version with an axis-value suffix.
        for shard_id, p in items:
            suffix = _suffix_for(
                rows_by_id.get(shard_id, {}).get("overrides", {}), varied
            )
            stem, ext = os.path.splitext(rel.name)
            tagged = f"{stem}.{suffix}{ext}" if suffix else f"{stem}.{shard_id}{ext}"
            shutil.copy2(p, dest.parent / tagged)
            collisions += 1

    print(
        f"+ collect: {copied} copied, {merged_count} merged, "
        f"{collisions} suffixed -> {base}",
        file=sys.stderr,
    )
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="causalab.runner.fanout",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("runner", nargs="?", help="Runner config name (basename ok).")
    p.add_argument("--spec", help="YAML sweep spec (axes/constant_overrides).")
    p.add_argument("--axis", action="append", help="Sweep axis: 'KEY=v1,v2,v3'.")
    p.add_argument(
        "--axis-each",
        action="append",
        dest="axis_each",
        help="Shard axis: 'KEY=range(0,32)' -> one shard per value (singleton list).",
    )
    p.add_argument("--const", action="append", help="Override applied to every shard.")
    p.add_argument(
        "--base", help="experiment_root base (default: runner config value)."
    )
    p.add_argument("--backend", choices=["auto", "slurm", "local"], default="auto")
    p.add_argument(
        "--local", action="store_true", help="Shorthand for --backend local."
    )
    p.add_argument(
        "--slurm", action="store_true", help="Shorthand for --backend slurm."
    )
    p.add_argument("--cap", type=int, help="Max concurrent shards (default min(N,32)).")
    p.add_argument("--gpus", type=int, help="GPUs per shard (default: model config).")
    p.add_argument("--time", help="Walltime per shard (default: runner config).")
    p.add_argument("--partition", help="SLURM partition (per-site; omitted if unset).")
    p.add_argument("--account", help="SLURM account (per-site; omitted if unset).")
    p.add_argument("--qos", help="SLURM qos (per-site; omitted if unset).")
    p.add_argument("--cluster", help="Cluster config name under configs/cluster/.")
    p.add_argument("--wait", action="store_true", help="Block until shards finish.")
    p.add_argument("--collect", action="store_true", help="Recombine shards when done.")
    p.add_argument(
        "--collect-only",
        action="store_true",
        help="Just recombine an existing <base>/shards/* (after a detached job).",
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Write manifest + plan; don't submit."
    )
    p.add_argument(
        "--emit-row",
        nargs=2,
        metavar=("MANIFEST", "INDEX"),
        help="Internal: print a shard's id + overrides for the array task.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.emit_row:
        emit_row(args.emit_row[0], int(args.emit_row[1]))
        return 0

    if args.collect_only:
        if not args.base:
            raise SystemExit("error: --collect-only requires --base")
        return collect(Path(args.base).resolve())

    if args.local:
        args.backend = "local"
    if args.slurm:
        args.backend = "slurm"

    runner, axes, consts, spec_meta = load_spec(args)

    # Resolve base (experiment_root). Default to the runner config's value.
    base_str = args.base or spec_meta.get("base")
    code = _session_code_dir(Path(base_str).resolve()) if base_str else None
    extra_dirs = [code / "configs"] if code is not None else []
    cfg = compose_runner_cfg(resolve_runner(runner, Path(base_str or ".")), extra_dirs)
    if not base_str:
        base_str = str(cfg.experiment_root)
    base = Path(base_str).resolve()
    runner = resolve_runner(runner, base)
    code = _session_code_dir(base)

    shards = build_shards(axes, consts)
    cap = args.cap or spec_meta.get("cap") or min(len(shards), 32)
    backend = detect_backend(
        spec_meta.get("backend", args.backend)
        if args.backend == "auto"
        else args.backend
    )

    # Single source of truth for GPU count / walltime, shared with the
    # run_exp.sh --slurm path via slurm_args so the two can't drift. CLI
    # --gpus / --time win; else the runner config; else a safe default.
    from causalab.runner.slurm_args import resolve_gpus_time

    cfg_gpus, cfg_time = resolve_gpus_time(cfg, gpus_default=1, time_default="04:00:00")
    gpus = args.gpus if args.gpus is not None else cfg_gpus
    time = args.time or cfg_time
    job_name = "causalab_fanout_" + os.path.basename(runner)

    base.mkdir(parents=True, exist_ok=True)
    manifest = write_manifest(
        base,
        shards,
        {
            "runner": runner,
            "backend": backend,
            "cap": cap,
            "gpus_per_shard": gpus,
            "time": time,
            "base": str(base),
            "n_shards": len(shards),
            "axes": [{"key": k, "values": v} for k, v in axes],
            "constant_overrides": consts,
        },
    )
    print(
        f"+ {len(shards)} shards | backend={backend} | cap={cap} | "
        f"gpus/shard={gpus} | base={base}",
        file=sys.stderr,
    )
    print(f"+ manifest: {manifest}", file=sys.stderr)

    if args.dry_run:
        for i, ov in enumerate(shards[:8]):
            print(f"    shard {_shard_id(i)}: {' '.join(overrides_to_tokens(ov))}")
        if len(shards) > 8:
            print(f"    ... (+{len(shards) - 8} more)")
        return 0

    if backend == "local":
        rc = submit_local(runner, base, shards, manifest, cap)
        if rc == 0 and args.collect:
            return collect(base)
        return rc

    # slurm
    directives = resolve_cluster_directives(args)
    rc, array_id, waited = submit_slurm(
        runner,
        base,
        shards,
        manifest,
        cap,
        gpus,
        time,
        job_name,
        directives,
        code,
        wait=args.wait,
    )
    if rc != 0:
        return rc
    if args.collect:
        if waited:
            return collect(base)
        print(
            f"! --collect needs the jobs to finish first. After {array_id} "
            f"completes (monitor_jobs {array_id}), run:\n"
            f"    uv run python -m causalab.runner.fanout --base {base} --collect-only",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
