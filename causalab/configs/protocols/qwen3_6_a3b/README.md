# qwen3.6-35b-a3b — a runnable protocol per method

Twelve documents plus one workflow, one per method the Silico causalab
documents describe, all pointed at
[`Qwen/Qwen3.6-35B-A3B`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) and the
`weekdays` tables in [`../../data/weekdays`](../../data/weekdays).

They exist because the six-step protocol in silico's
`answer-research-question/` marks each method **Execution: stub** — "the
science is kept, the invocations aren't". These are the invocations.

```bash
causalab validate causalab/configs/protocols/qwen3_6_a3b/interchange.json --data \
    --data-root causalab/configs/data
causalab run      causalab/configs/protocols/qwen3_6_a3b/interchange.json \
    --data-root causalab/configs/data --out runs/interchange --device cuda
causalab run      causalab/configs/workflows/qwen3_6_a3b_weekdays.json \
    --data-root causalab/configs/data --out runs --device cuda
```

| document | method it implements | points |
|---|---|---|
| `probe.json` | step 2 · probe — greedy decode, no writes | 1 |
| `logit_lens.json` | step 2 · logit lens (layer × position) | 77 |
| `locate_scan.json` | step 2 · paired examples / step 4 · locate | 40 |
| `knockout_head.json` | step 2 · knockout, attention head grid | 160 |
| `knockout_mlp.json` | step 2 · knockout, MLP bands at width 1 | 40 |
| `knockout_mlp_band3.json` | step 2 · knockout, one width-3 band | 1 |
| `harvest.json` | step 2 · PCA's input + the mean-ablation reference | 1 |
| `head_attribution.json` | round 2's derived per-head `attention_result` | 1 |
| `interchange.json` | step 4 · the core test | 1 |
| `control_positive.json` | step 4 · full mediation, must sit at ceiling | 1 |
| `control_negative.json` | step 4 · a site the variable cannot be at | 1 |
| `das_sweep.json` | step 4 · DAS over k × seed | 6 |
| `das_apply.json` | step 4 · the winner on held-out data (workflow-driven) | 1 |

## What the model's shape costs you

`Qwen3.6-35B-A3B` is a **hybrid** tower: `full_attention_interval` is 4, so of
its 40 layers only **ten** (3, 7, … 39) carry a full-attention mixer and the
other thirty carry a Gated DeltaNet. Three consequences the documents encode:

- **the attention head grid is 10 × 16, not 40 × 16.** Naming an attention
  component at a DeltaNet layer is refused with the architectural reason —
  there is no attention matrix there — not with "not implemented".
- **`mlp_activation` is not addressable at all.** Every layer is a sparse-MoE
  block, which has no `act_fn`, so MLP-family knockout goes through
  `mlp_output`.
- **KV-space components have two heads, not sixteen** (`num_key_value_heads` is
  2 against 16 query heads). `head: 5` on `attention_key`,
  `attention_key_pre_rope` or `attention_value_states` is refused rather than
  silently yielding an empty slice.

`head_dim` is 256 and decoupled, so `attention_premix`, `attention_z` and
`attention_query` are 4096 wide — twice `hidden_size`.

## Retargeting

Every document differs from a Llama-shaped one in `model.key` and site
`layer`s, and nothing else. The one thing that is **not** portable is which
layers may carry an attention component; that is the hybrid tower's own fact,
pinned in `tests/protocol/test_shipped_a3b_configs.py`.

## Two places the document layer does not reach

- **A swept band start.** A sweep axis is a field *value*, not a set of table
  entries, so a width-3 band whose start moves is a dependent axis — spec §3
  assigns those to a generator that emits the JSON.
  `knockout_mlp_band3.json` is the shape such a generator emits.
- **A prompt-frame metric over several positions.** `top_k` reduces exactly one
  position per example outside the continuation frame, so the logit lens's
  (layer × position) grid is two swept axes rather than one `pos: "all"` read.
