# Attention Pattern

Attention pattern answers: *How do individual attention heads route information between token positions, and which heads attend to task-relevant tokens?* It extracts attention weights at configurable (layer, head) pairs, computes per-head statistics (entropy, max attention, self-attention, previous-token), and optionally aggregates attention by semantic token type. This is a standalone diagnostic — it does not require prior analyses and is not consumed by downstream modules.

---

## Configuration

**Root config** (`causalab/configs/config.yaml`) — shared params used by this analysis:
- `experiment_root` — output root (default: `artifacts/${task.name}/${model.id}`)

**Module config** (`causalab/configs/analysis/attention_pattern.yaml`):

```yaml
analysis:
  _name_: attention_pattern
  _output_dir: ${experiment_root}/attention_pattern

  seed: 42                        # RNG seed for example sampling
  n_examples: 8                   # number of correct examples to generate
  max_attempts: 100               # max sampling attempts to find n_examples correct ones

  layers: null                    # list[int] or null = scan all layers
  heads: null                     # list[int] or null = scan all heads

  source_token_types: null        # list of task token position names (FROM), or null to skip
  target_token_types: null        # list of task token position names (TO), or null to skip

  ignore_first_token: true        # exclude attention sink from heatmaps
  max_tokens_display: 40          # max tokens shown on comparison grid axes

  visualization:
    figure_format: pdf            # png or pdf
    per_head_heatmaps: true       # single-example attention heatmap per (layer,head)
    average_heatmaps: true        # averaged attention heatmap (same-length examples only)
    statistics_charts: true       # per-head statistics bar chart
    token_type_heatmaps: true     # token-type attention heatmap per (layer,head)
    head_comparison_grid: true    # multi-head comparison grid per layer
```

Setting `layers` and `heads` to specific lists (e.g. `layers: [14]`, `heads: [0, 11, 14, 15]`) is recommended for targeted analysis. Scanning all layers x heads is feasible but slow.

Setting both `source_token_types` and `target_token_types` enables the token-type aggregation. These must be valid token position names from the task (e.g. `last_token`, `correct_symbol`). Setting either to `null` skips token-type analysis entirely.

---

## Outputs

### Interpretation

- **`results.json`** — Per-head statistics (`avg_entropy`, `avg_max_attention`, `avg_diagonal`, `avg_previous`). Compare entropy across heads: low entropy = focused head (attends to few positions), high entropy = diffuse. High `avg_diagonal` = self-attention head. High `avg_previous` = induction-like previous-token head.

- **`token_type_results.json`** — Per-head source x target attention matrices with mean and std. High values reveal which semantic token types a head routes between (e.g. "last_token attends strongly to correct_symbol"). Only produced when `source_token_types` and `target_token_types` are configured.

- **`visualization/L*_H*/heatmap.pdf`** — Raw attention pattern for one example. Rows are query positions (FROM), columns are key positions (TO). Useful for spotting specific routing patterns.

- **`visualization/L*_H*/avg_heatmap.pdf`** — Attention averaged across examples with matching sequence length. Reduces per-example noise; persistent patterns indicate structural head behavior.

- **`visualization/L*_H*/statistics.pdf`** — Bar chart of the four statistics metrics for that head.

- **`visualization/L*_H*/token_type_heatmap.pdf`** — Semantic token-type attention matrix. Compact view of which token types a head routes between, averaged over examples.

- **`visualization/layer_*_grid.pdf`** — Side-by-side comparison of all scanned heads within one layer. Quick visual scan to identify which heads have distinctive patterns.

### Saved artifacts

| File | Format | Used by |
|---|---|---|
| `results.json` | `{statistics_per_head: {L*_H*: {avg_entropy, ...}}, n_examples, pairs_scanned}` | human reference |
| `token_type_results.json` | `{L*_H*: {attention_matrix, std_matrix, source_ids, target_ids}}` | human reference |
| `metadata.json` | run config snapshot | provenance |
