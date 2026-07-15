"""Resolve sbatch resource args from a runner's Hydra config.

Invoked by ``scripts/run_exp.sh --slurm`` to print a single line:

    <gpus> <time> <job_name>

The wrapper script captures these and forwards them to ``sbatch`` so the
runner config remains the single source of truth for GPU count and walltime.
"""

from __future__ import annotations

import argparse
import os
import sys


def resolve_gpus_time(
    cfg, *, gpus_default: int | None = None, time_default: str | None = None
) -> tuple[int, str]:
    """Read ``(gpus, walltime)`` from a composed runner config.

    The one place that knows the config paths (``model.slurm.gpus`` /
    ``slurm.time``), so the ``run_exp.sh --slurm`` path (via this module) and the
    fan-out orchestrator (``causalab.runner.fanout``) cannot drift. With no
    default a missing key raises — a SLURM runner that omits its resources should
    fail loudly. Fan-out passes defaults so its ``--gpus`` / ``--time`` overrides
    and fallbacks apply instead (and a ``slurm`` block without ``gpus`` no longer
    ``KeyError``s).
    """
    model = cfg.get("model")
    slurm = model.get("slurm") if model else None
    gpus = slurm.get("gpus") if slurm else None
    if gpus is None:
        gpus = gpus_default
    if gpus is None:
        raise KeyError("model.slurm.gpus is not set in the runner config")
    time = cfg.get("slurm", {}).get("time", time_default)
    if time is None:
        raise KeyError("slurm.time is not set in the runner config")
    return int(gpus), str(time)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runner", help="Runner config name, e.g. age/age_8b_k64")
    parser.add_argument(
        "--config-dir",
        action="append",
        default=[],
        help=(
            "Extra Hydra config dir, considered alongside the shipped "
            "causalab/configs/. Repeatable. Whichever dir actually contains "
            "the runner becomes the primary path; the rest are appended to "
            "``hydra.searchpath`` so defaults like ``analysis/baseline`` keep "
            "resolving. Mirrors how ``--config-dir`` extends the search path "
            "for ``@hydra.main`` in run_exp.py."
        ),
    )
    args = parser.parse_args()

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    import causalab

    shipped = os.path.join(os.path.dirname(causalab.__file__), "configs")
    extra_dirs = [os.path.abspath(p) for p in args.config_dir]

    # Hydra's primary-config lookup uses only the dir passed to
    # initialize_config_dir — runtime ``hydra.searchpath`` overrides apply to
    # subsequent (defaults) lookups, not the primary. So we pick whichever
    # configured dir actually contains the runner as primary, and put the
    # rest on hydra.searchpath. Extras are searched first so user/session
    # configs win over shipped on name collisions.
    candidates = extra_dirs + [shipped]
    primary = next(
        (
            d
            for d in candidates
            if os.path.isfile(os.path.join(d, f"{args.runner}.yaml"))
        ),
        shipped,
    )
    others = [d for d in candidates if d != primary]

    overrides: list[str] = []
    if others:
        paths = ",".join(f"file://{p}" for p in others)
        overrides.append(f"hydra.searchpath=[{paths}]")

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=primary, version_base=None):
        cfg = compose(config_name=args.runner, overrides=overrides)

    gpus, time = resolve_gpus_time(cfg)
    job_name = "causalab_" + os.path.basename(args.runner)
    print(f"{gpus} {time} {job_name}")


if __name__ == "__main__":
    sys.exit(main())
