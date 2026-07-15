#!/usr/bin/env bash
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/run_exp.sh [opts] <runner> [hydra overrides...]

Opts:
  --slurm                   Dispatch as an sbatch job (else run inline).
  --wait                    With --slurm: block until the job finishes and exit
                            with its status (sbatch --wait). Replaces hand-rolled
                            `while squeue -j <id>` pollers.
  --qos QOS                 normal | opportunistic | scavenge (--slurm only).
  --gpus N                  Override model-config GPU count (--slurm only).
  --time HH:MM:SS           Override runner-config walltime  (--slurm only).
  --config-dir DIR          Hydra --config-dir (out-of-tree configs).
  --experiment-root DIR     Override experiment_root.
  -h, --help                Show this message.

Hydra introspection flags (--cfg job|hydra|all, --resolve, --package PKG) may be
passed anywhere — before or after <runner> — and are forwarded to Hydra.

SLURM logs land in slurm_logs/<job-name>_<jobid>.out (and .err), relative to the
repo root; the resolved path is printed at submit. When --experiment-root is a
session dir, the resolved config is also snapshotted to
<session>/run/<runner>_resolved.yaml before the run starts.

Examples:
  scripts/run_exp.sh age_8b_k64
  scripts/run_exp.sh age_8b_k64 --cfg job          # print resolved config, don't run
  scripts/run_exp.sh --slurm age_8b_k64
  scripts/run_exp.sh --slurm --wait age_8b_k64     # submit and block until done
  scripts/run_exp.sh --slurm --qos=opportunistic --time=08:00:00 alphabet_70b_k128
EOF
    exit 1
}

slurm=0
wait_for_job=0
qos=""
gpus_override=""
time_override=""
config_dir=""
experiment_root=""
# Hydra introspection flags (--cfg/-c, --resolve, --package/-p) collected from
# anywhere on the command line and forwarded to Hydra. Without this, a leading
# `--cfg job` is caught by the `*) break` arm below and mistaken for the runner
# name (#269 I4).
hydra_passthrough=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --slurm)               slurm=1;                shift ;;
        --wait)                wait_for_job=1;         shift ;;
        --qos)                 qos="$2";               shift 2 ;;
        --qos=*)               qos="${1#*=}";          shift ;;
        --gpus)                gpus_override="$2";     shift 2 ;;
        --gpus=*)              gpus_override="${1#*=}"; shift ;;
        --time)                time_override="$2";     shift 2 ;;
        --time=*)              time_override="${1#*=}"; shift ;;
        --config-dir)          config_dir="$2";        shift 2 ;;
        --config-dir=*)        config_dir="${1#*=}";   shift ;;
        --experiment-root)     experiment_root="$2";   shift 2 ;;
        --experiment-root=*)   experiment_root="${1#*=}"; shift ;;
        --cfg|-c)              hydra_passthrough+=("$1" "$2");          shift 2 ;;
        --cfg=*)              hydra_passthrough+=("--cfg" "${1#*=}");   shift ;;
        --package|-p)         hydra_passthrough+=("$1" "$2");          shift 2 ;;
        --package=*)          hydra_passthrough+=("--package" "${1#*=}"); shift ;;
        --resolve)            hydra_passthrough+=("$1");               shift ;;
        -h|--help)             usage ;;
        *) break ;;
    esac
done
[ $# -lt 1 ] && usage

runner="$1"
shift
# Strip leading runners/ prefix if present. The wrapper itself adds this prefix
# during discovery so Hydra resolves the runner correctly, but on SLURM re-exec
# the prefixed value gets passed back in as $1 — double-prefixing then breaks
# the discovery check.
runner="${runner#runners/}"

# Auto-discover the runner config by basename if it wasn't passed as an
# explicit relative path. Runners live under causalab/configs/runners/<group>/
# but Hydra needs the path relative to causalab/configs/.
#
# Under sbatch, SLURM stages the batch script as /var/spool/slurmd/jobN/slurm_script,
# so BASH_SOURCE[0] no longer points to the in-tree script. SLURM_SUBMIT_DIR is set
# to the directory sbatch was invoked from (which the --slurm path below cd's to
# repo_root), so prefer it when present.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    repo_root="$SLURM_SUBMIT_DIR"
else
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    repo_root="$(cd "$script_dir/.." && pwd)"
fi
# --- Session-local code detection (computed before runner resolution so
# session-local runners can be discovered too) ----------------------------------
# When --experiment-root lives under agent_logs/<session>/, expose the session's
# code/ tree to the runner so session-local method/analysis prototypes resolve.
# See causalab/runner/README.md "Session-local code injection" for the feature.
# Channels:
#   1) PYTHONPATH gets ${SESSION_DIR}/code/ prepended so
#      `import analyses.<name>` and `import methods.<name>` work.
#   2) Hydra's searchpath gets ${SESSION_DIR}/code/configs/ appended so
#      `- analysis/<name>` defaults entries find the session-local YAML.
#   3) Runner discovery also looks under ${SESSION_DIR}/code/configs/runners/.
session_dir=""
if [ -n "$experiment_root" ]; then
    case "$experiment_root" in
        *agent_logs/*/artifacts*)
            session_dir="${experiment_root%%/artifacts*}"
            ;;
    esac
fi
if [ -n "${CAUSALAB_SESSION_CODE:-}" ] && [ "$CAUSALAB_SESSION_CODE" != "1" ]; then
    session_dir="$CAUSALAB_SESSION_CODE"
fi
session_code=""
if [ -n "$session_dir" ] && [ -d "$session_dir/code" ]; then
    session_code="$(cd "$session_dir/code" && pwd)"
fi

# --- Runner discovery -------------------------------------------------------
configs_dir="$repo_root/causalab/configs"
session_runners_dir=""
[ -n "$session_code" ] && [ -d "$session_code/configs/runners" ] && \
    session_runners_dir="$session_code/configs/runners"

# Searches first in causalab/configs/runners/, then (if present) in the session's
# code/configs/runners/. Session-local runners require the wrapper to invoke
# Hydra with --config-dir pointing at the session's configs root (added
# unconditionally below whenever session code is present).
if [ ! -f "$configs_dir/$runner.yaml" ]; then
    if [ -f "$configs_dir/runners/$runner.yaml" ]; then
        runner="runners/$runner"
    elif [ -n "$session_runners_dir" ] && [ -f "$session_runners_dir/$runner.yaml" ]; then
        runner="runners/$runner"
    else
        matches=$(find "$configs_dir/runners" -type f -name "$runner.yaml" 2>/dev/null)
        if [ -n "$session_runners_dir" ]; then
            session_matches=$(find "$session_runners_dir" -type f -name "$runner.yaml" 2>/dev/null)
        else
            session_matches=""
        fi
        all_matches=$(printf '%s\n%s\n' "$matches" "$session_matches" | grep -c . || true)
        if [ "$all_matches" -eq 1 ]; then
            if [ -n "$matches" ]; then
                runner=$(printf '%s\n' "$matches" | sed "s|$configs_dir/||;s|\.yaml\$||")
            else
                runner=$(printf '%s\n' "$session_matches" | sed "s|$session_runners_dir/||;s|\.yaml\$||")
                runner="runners/$runner"
            fi
        elif [ "$all_matches" -gt 1 ]; then
            echo "Ambiguous runner name '$runner'; matches:" >&2
            printf '%s\n%s\n' "$matches" "$session_matches" | grep . >&2
            echo "Use the group-qualified form, e.g. <group>/$runner" >&2
            exit 1
        else
            echo "Runner not found: $runner" >&2
            echo "Searched: $configs_dir/runners/" >&2
            [ -n "$session_runners_dir" ] && echo "Also: $session_runners_dir/" >&2
            exit 1
        fi
    fi
fi

# Hydra's CLI is order-sensitive: all --flag-style options (--config-dir,
# --config-name) must precede key=value overrides, otherwise the argparse
# layer rejects the overrides as unrecognized positional args. Keep the two
# kinds in separate arrays and emit flags-first, overrides-last (with user
# "$@" overrides last of all).
hydra_flags=()
hydra_overrides=()
[ -n "$config_dir" ]      && hydra_flags+=("--config-dir" "$(realpath "$config_dir")")
[ -n "$experiment_root" ] && hydra_overrides+=("experiment_root=$experiment_root")

# --- Apply session-local code injection -------------------------------------
if [ -n "$session_code" ]; then
    export PYTHONPATH="${session_code}${PYTHONPATH:+:$PYTHONPATH}"
    export CAUSALAB_SESSION_CODE="$session_dir"
    if [ -d "$session_code/configs" ]; then
        hydra_overrides+=("++hydra.searchpath=[file://$session_code/configs]")
        # ++hydra.searchpath is applied AFTER the primary config's own defaults
        # list is composed, so it can't resolve nested `defaults:` entries
        # (e.g. - /analysis/<name>, - /task: <name>). --config-dir is added to
        # the search path BEFORE composition, so it can. Add it whenever session
        # configs exist — not only for runners discovered session-local (#166
        # sub-bug C: a runner copied in-tree is "qualified", so the old
        # runner_session_local gate skipped this and Hydra failed with
        # "Could not load 'analysis/<name>'"). Mirrors the primary-vs-searchpath
        # handling in causalab/runner/slurm_args.py.
        [ -z "$config_dir" ] && hydra_flags+=("--config-dir" "$session_code/configs")
    fi
    echo "+ session-local code: $session_code (PYTHONPATH + Hydra searchpath)" >&2
fi

# --- Snapshot the resolved config at submit time (#269 I5) -------------------
# the interpret phase treats <session>/run/<runner>_resolved.yaml as the
# canonical record of what ran. Capture it here, before dispatch — for --slurm
# on the submitting host, so it exists pre-run rather than as a separate manual
# `--cfg job | tee` step that can be skipped or run after the fact. Done
# whenever a session dir is detected; a --slurm job re-snapshots on the compute
# node, which is a harmless idempotent overwrite.
if [ -n "$session_dir" ]; then
    cd "$repo_root"
    snapshot_dir="$session_dir/run"
    snapshot_file="$snapshot_dir/$(basename "$runner")_resolved.yaml"
    mkdir -p "$snapshot_dir"
    if uv run python -m causalab.runner.run_exp \
        ${hydra_flags[@]+"${hydra_flags[@]}"} \
        --config-name "$runner" \
        ${hydra_overrides[@]+"${hydra_overrides[@]}"} \
        --cfg job > "$snapshot_file" 2>/dev/null; then
        echo "+ resolved config snapshot: $snapshot_file" >&2
    else
        echo "! could not snapshot resolved config to $snapshot_file" >&2
        rm -f "$snapshot_file"
    fi
fi

# --- Slurm submission path ---------------------------------------------------
# Resolve gpus/time/job_name from the Hydra config (single source of truth)
# and exec sbatch. CLI flags override Hydra-resolved values. The re-exec drops
# the --slurm flag so the job step falls through to the inline path.
if [ "$slurm" -eq 1 ]; then
    cd "$repo_root"
    mkdir -p slurm_logs

    # Mirror the inline path's Hydra searchpath: the shipped configs are the
    # primary path (set by @hydra.main in run_exp.py and by slurm_args.py),
    # and any user --config-dir or session-local code/configs/ are added on
    # top. Without this, a session-local runner can't be found, or it's found
    # but its references to shipped defaults (analysis/baseline, …) can't.
    slurm_args_cmd=(uv run python -m causalab.runner.slurm_args "$runner")
    [ -n "$config_dir" ] && slurm_args_cmd+=(--config-dir "$(realpath "$config_dir")")
    [ -n "$session_code" ] && [ -d "$session_code/configs" ] && \
        slurm_args_cmd+=(--config-dir "$session_code/configs")
    read -r r_gpus r_time r_name < <("${slurm_args_cmd[@]}")

    gpus="${gpus_override:-$r_gpus}"
    time="${time_override:-$r_time}"
    name="$r_name"

    sb_args=(
        --gres="gpu:${gpus}"
        --time="${time}"
        --job-name="${name}"
    )
    [ -n "$qos" ] && sb_args+=(--qos="$qos")

    # --wait: block until the job completes and exit with its status. This is the
    # native "wait for a SLURM job" primitive — no hand-rolled `while squeue -j
    # <id>` poller needed (#269 I2). exec keeps the clean handoff; the shell only
    # returns once sbatch (and thus the job) is done.
    [ "$wait_for_job" -eq 1 ] && sb_args+=(--wait)

    # CAUSALAB_SESSION_CODE is exported above, but sbatch's env handling does not
    # reliably propagate it into the job step, so the in-job re-exec can't
    # re-detect the session (#166 sub-bug D). Force it through explicitly; the
    # leading "ALL," keeps the normal inherit-submitting-env behaviour on top.
    [ -n "$session_code" ] && sb_args+=(--export="ALL,CAUSALAB_SESSION_CODE=$session_dir")

    # Forward the original invocation, minus the --slurm flag (and its peers).
    # Flags MUST precede the runner positional arg — otherwise the wrapper's
    # argparser breaks at the first positional and leaves --config-dir /
    # --experiment-root in $@, which Hydra then rejects. See HANDOFF_v2.md §8
    # issue #1 (v1 worktree patch). --cfg/--resolve/--package are now recognized
    # before the positional too, so the passthrough goes here as well.
    forward=()
    [ -n "$config_dir" ]      && forward+=(--config-dir "$config_dir")
    [ -n "$experiment_root" ] && forward+=(--experiment-root "$experiment_root")
    forward+=("${hydra_passthrough[@]+"${hydra_passthrough[@]}"}")
    forward+=("$runner")
    forward+=("$@")

    # Log path (#269 I3): SBATCH --output/--error are slurm_logs/%x_%j.{out,err}
    # (%x = job name, %j = job id), repo-root-relative. Surface the resolved
    # pattern so the log is findable without guessing.
    echo "+ logs: $repo_root/slurm_logs/${name}_<jobid>.{out,err}" >&2
    echo "+ sbatch ${sb_args[*]} $(realpath "$0") ${forward[*]}" >&2
    exec sbatch "${sb_args[@]}" "$(realpath "$0")" "${forward[@]}"
fi

# --- Inline path -------------------------------------------------------------
# Works on laptop, dev pod, or inside an sbatch step alike. Hydra introspection
# flags collected before the positional (e.g. `--cfg job`) are forwarded after
# --config-name; trailing "$@" carries any overrides given after the runner.
cd "$repo_root"
uv run python -m causalab.runner.run_exp \
    ${hydra_flags[@]+"${hydra_flags[@]}"} \
    --config-name "$runner" \
    ${hydra_passthrough[@]+"${hydra_passthrough[@]}"} \
    ${hydra_overrides[@]+"${hydra_overrides[@]}"} \
    "$@"
