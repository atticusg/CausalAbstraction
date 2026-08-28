# {{ANALYSIS_TITLE}}

{{ANALYSIS_TITLE}} answers: *{{RESEARCH_QUESTION}}* {{WHAT_IT_DOES_MECHANICALLY}}.

{{POSITION_IN_PIPELINE}}

---

## Configuration

**Root config** (`causalab/configs/config.yaml`) — shared params used by this analysis:
- `experiment_root` — output root (default `artifacts/${task.name}/${model.id}`; pass `--experiment-root $WORKDIR/artifacts/...` to route it to your working directory)
- `seed` — dataset generation seed
- {{ANY_OTHER_ROOT_LEVEL_PARAMS_USED}}

**Module config** (`$WORKDIR/code/configs/analysis/{{ANALYSIS_NAME}}.yaml`):

```yaml
{{ANALYSIS_NAME}}:
  _name_: {{ANALYSIS_NAME}}
  _subdir: {{SUBDIR_PATTERN}}
  _output_dir: ${experiment_root}/{{ANALYSIS_NAME}}/${._subdir}
  # {{KNOB_DOCS_FROM_SPEC_SECTION_4}} — one bullet per knob with inline `# comment`
  visualization:
    figure_format: ${figure_format}
```

---

## Outputs

### Interpretation

{{ONE_BULLET_PER_HUMAN_READABLE_OUTPUT}}

- **`results.json`** — {{WHAT_THE_NUMBER_OR_VISUAL_SHOWS}}. {{GOOD_RESULT}}. {{BAD_RESULT}}.
- **`heatmap.png`** — {{ROW_COL_SEMANTICS}}. Look for {{WHAT_TO_LOOK_FOR}}.

### Saved artifacts

| File | Shape / Format | Used by |
|---|---|---|
| `results.json` | {{SCHEMA}} | {{CONSUMER}} |
| `heatmap.png` | matplotlib heatmap | human reference |
| `metadata.json` | run config snapshot | provenance |
