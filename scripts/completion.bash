# Bash completion for ./scripts/run_exp.sh
#
# Enable for the current shell:
#   source scripts/completion.bash
#
# Enable permanently by adding the line above to ~/.bashrc
# (use an absolute path to the repo).

_causalab_run_exp_complete() {
    local cur="${COMP_WORDS[COMP_CWORD]}"

    # Resolve the configs dir from this script's location (not cwd).
    local script_dir configs_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    configs_dir="$script_dir/../causalab/configs"

    # Only complete the runner name (the first positional arg). After that,
    # arguments are passed through to Hydra and we leave completion alone.
    local i positional=0
    for (( i=1; i < COMP_CWORD; i++ )); do
        case "${COMP_WORDS[i]}" in
            --config-dir|--experiment-root|--qos|--gpus|--time) ((i++)) ;;
            -*) : ;;
            *) ((positional++)) ;;
        esac
    done
    [ "$positional" -ge 1 ] && return 0

    # Runners live under configs/runners/<group>/<name>.yaml. Emit the bare
    # basename when it's unique across groups, or <group>/<name> when the same
    # basename appears in multiple groups (e.g. age/viz vs weekdays/viz).
    local configs
    configs=$(find "$configs_dir/runners" -name '*.yaml' -type f -printf '%P\n' 2>/dev/null \
              | awk -F/ '
                  {
                      p = $0; sub(/\.yaml$/, "", p)
                      n = split(p, parts, "/"); b = parts[n]
                      paths[NR] = p; bases[NR] = b; count[b]++
                  }
                  END {
                      for (i = 1; i <= NR; i++)
                          print (count[bases[i]] > 1 ? paths[i] : bases[i])
                  }')
    COMPREPLY=( $(compgen -W "$configs" -- "$cur") )
}

complete -F _causalab_run_exp_complete ./scripts/run_exp.sh scripts/run_exp.sh run_exp.sh
