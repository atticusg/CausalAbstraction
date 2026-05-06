#!/usr/bin/env bash
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: scripts/run_exp.sh [opts] <runner> [hydra overrides...]

Opts:
  --slurm                   Dispatch as an sbatch job (else run inline).
  --qos QOS                 normal | opportunistic | scavenge (--slurm only).
  --gpus N                  Override model-config GPU count (--slurm only).
  --time HH:MM:SS           Override runner-config walltime  (--slurm only).
  --config-dir DIR          Hydra --config-dir (out-of-tree configs).
  --experiment-root DIR     Override experiment_root.
  -h, --help                Show this message.

Examples:
  scripts/run_exp.sh age_8b_k64
  scripts/run_exp.sh --slurm age_8b_k64
  scripts/run_exp.sh --slurm --qos=opportunistic --time=08:00:00 alphabet_70b_k128
EOF
    exit 1
}

slurm=0
qos=""
gpus_override=""
time_override=""
config_dir=""
experiment_root=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --slurm)               slurm=1;                shift ;;
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
        -h|--help)             usage ;;
        *) break ;;
    esac
done
[ $# -lt 1 ] && usage

runner="$1"
shift

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
# code/ tree to the runner so /setup-methods and /setup-analyses prototypes
# resolve. Channels:
#   1) PYTHONPATH gets ${SESSION_DIR}/code/ prepended so
#      `import analyses.<name>` and `import methods.<name>` work.
#   2) Hydra's searchpath gets ${SESSION_DIR}/code/configs/ appended so
#      `- analysis/<name>` defaults entries find the session-local YAML.
#   3) Runner discovery also looks under ${SESSION_DIR}/code/configs/runners/.
# See .claude/skills/research-session/CONVENTIONS.md "What goes in code/".
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
# Hydra with --config-dir pointing at the session's configs root.
runner_session_local=0
if [ ! -f "$configs_dir/$runner.yaml" ]; then
    if [ -f "$configs_dir/runners/$runner.yaml" ]; then
        runner="runners/$runner"
    elif [ -n "$session_runners_dir" ] && [ -f "$session_runners_dir/$runner.yaml" ]; then
        runner="runners/$runner"
        runner_session_local=1
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
                runner_session_local=1
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
    fi
    if [ "$runner_session_local" -eq 1 ] && [ -z "$config_dir" ]; then
        hydra_flags+=("--config-dir" "$session_code/configs")
    fi
    echo "+ session-local code: $session_code (PYTHONPATH + Hydra searchpath)" >&2
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

    # Forward the original invocation, minus the --slurm flag (and its peers).
    forward=("$runner")
    [ -n "$config_dir" ]      && forward+=(--config-dir "$config_dir")
    [ -n "$experiment_root" ] && forward+=(--experiment-root "$experiment_root")
    forward+=("$@")

    echo "+ sbatch ${sb_args[*]} $(realpath "$0") ${forward[*]}" >&2
    exec sbatch "${sb_args[@]}" "$(realpath "$0")" "${forward[@]}"
fi

# --- Inline path -------------------------------------------------------------
# Works on laptop, dev pod, or inside an sbatch step alike.
cd "$repo_root"
uv run python -m causalab.runner.run_exp \
    ${hydra_flags[@]+"${hydra_flags[@]}"} \
    --config-name "$runner" \
    ${hydra_overrides[@]+"${hydra_overrides[@]}"} \
    "$@"
