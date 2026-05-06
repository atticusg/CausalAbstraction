# Zsh completion for ./scripts/run_exp.sh
#
# Enable for the current shell:
#   source scripts/completion.zsh
#
# Enable permanently by adding the line above to ~/.zshrc
# (use an absolute path to the repo).

_causalab_run_exp_complete() {
    local script_dir configs_dir
    script_dir="${${(%):-%x}:A:h}"
    configs_dir="$script_dir/../causalab/configs"

    # Only complete the runner name (first positional arg). Skip flags and
    # their values; once a positional has been seen, defer to default.
    local -a words
    words=("${(@)words[2,CURRENT-1]}")
    local i=1 positional=0
    while (( i <= ${#words} )); do
        case "${words[i]}" in
            --config-dir|--experiment-root|--qos|--gpus|--time) (( i += 2 )) ;;
            -*) (( i++ )) ;;
            *) (( positional++ )); (( i++ )) ;;
        esac
    done
    (( positional >= 1 )) && return 0

    # Runners live under configs/runners/<group>/<name>.yaml. Emit the bare
    # basename when it's unique across groups, or <group>/<name> when the same
    # basename appears in multiple groups (e.g. age/viz vs weekdays/viz).
    local -a configs
    configs=( $(find "$configs_dir/runners" -name '*.yaml' -type f -printf '%P\n' 2>/dev/null \
                | awk -F/ '
                    {
                        p = $0; sub(/\.yaml$/, "", p)
                        n = split(p, parts, "/"); b = parts[n]
                        paths[NR] = p; bases[NR] = b; count[b]++
                    }
                    END {
                        for (i = 1; i <= NR; i++)
                            print (count[bases[i]] > 1 ? paths[i] : bases[i])
                    }') )
    _describe 'runner config' configs
}

# Make sure compinit has run.
if ! whence -w compdef >/dev/null 2>&1; then
    autoload -Uz compinit && compinit
fi

compdef _causalab_run_exp_complete ./scripts/run_exp.sh scripts/run_exp.sh run_exp.sh
