# Hook oracles

The intervention engine's correctness rests on one idea: every behaviour it
implements is re-implemented independently with raw
`torch.nn.Module.register_forward_hook` / `register_forward_pre_hook`, and the
two are compared. The hand-rolled version is the **oracle**; the engine has to
match it.

The oracles touch no intervention backbone at all — they install PyTorch hooks
and do the arithmetic inline. That is what makes them survive a backbone change:
when the pyvene backbone was replaced by nnsight (#380), these tests re-ran
unchanged and were the acceptance gate for the swap. They are still the gate for
any future change to `neural/activations/`.

Shared helpers live in [`tests/neural/activations/hook_oracle.py`](../tests/neural/activations/hook_oracle.py).

## Coverage map

| Behaviour | Oracle file | Tests |
|---|---|---|
| Collect: ordering, featurized + per-head collection | `tests/neural/activations/test_collect_hook_oracle.py` | 9 |
| Interchange, across model families | `tests/neural/activations/test_interchange_hook_oracle.py` | 18 |
| Interchange, exhaustive (spans, heads, featurizers, groups) | `tests/neural/activations/test_interchange_mode.py` | 42 |
| Feature-space replace/steer + error preservation | `tests/neural/activations/test_feature_space_hook_oracle.py` | 9 |
| Interpolation (arbitrary `fn(f_base, f_src)`) | `tests/neural/activations/test_interpolation_hook_oracle.py` | 12 |
| Differentiable binary mask (DBM) | `tests/neural/activations/test_mask_hook_oracle.py` | 15 |
| Seeded noise corruption | `tests/neural/activations/test_noise_hook_oracle.py` | 12 |
| Cross-model patching | `tests/neural/activations/test_cross_model_hook_oracle.py` | 6 |
| Path patching, two-pass | `tests/methods/path_patching/test_two_pass_hook_oracle.py` | 6 |
| Path patching, per-head value/query receivers + GQA | `tests/methods/path_patching/test_head_receivers_hook_oracle.py` | 5 |
| Gradients through an intervention (forward **and** grad) | `tests/neural/test_gradients.py` | 4 |
| Component resolution, all types x 3 families | `tests/neural/test_components.py` | 71 |
| **Everything above, on the real model** | `tests/neural/activations/test_chat_coherent_hook_oracle.py` | 29 |

Counts are collected tests, so a suite parametrized over the three model
fixtures counts three times.

The CPU tiers run on tiny-random stubs (Llama, GPT-2, and a synthetic
decoupled-`head_dim` GQA model). The last row re-runs the whole matrix on
`Qwen/Qwen3-4B-Instruct-2507` — real grouped-query attention (32 heads / 8 KV
heads) with a **decoupled `head_dim`** (128, not 2560/32 = 80) and the real
chat-template tokenization path. It is `@pytest.mark.golden`, the GPU tier.

## Why the model families are what they are

Each fixture exists to make a specific failure observable:

- **Llama (tiny-random)** — rotary positions, separate `q_proj`/`k_proj`/`v_proj`.
- **GPT-2 (tiny-random)** — *learned absolute* positions, so a wrong `position_ids`
  under left padding changes the activations. A RoPE model cannot fail that test:
  a uniform left-pad shift cancels. Also fused QKV (`c_attn`), which exercises the
  channel-slice path.
- **GQA + decoupled `head_dim` (synthetic)** — `head_dim=8` while
  `hidden // n_head == 4`, and `n_kv < n_q`. Per-head reads must follow
  `config.head_dim` and address k/v in KV space. This configuration was
  *unsupported* before the nnsight migration (#386).
- **Qwen3-4B-Instruct** — the same properties at real scale, on a real tokenizer.

## What an oracle looks like

The pattern is always: capture with a hook, do the arithmetic by hand, compare
against the engine's public entry point.

```python
# ground truth: capture the source activation, patch it into the base by hand
src = capture_residual(pipeline, LAYER, source_inputs)[:, [POS], :]
expected = next_token_logits(pipeline, base_inputs, LAYER, [POS], src)

# the engine, through its public API
result = run_interchange_interventions(pipeline, dataset, target, output_scores=True)

torch.testing.assert_close(result["scores"][0][0], expected, atol=1e-4, rtol=1e-3)
```

Assertions are against **public entry points** (`run_interchange_interventions`,
`collect_features`, `run_steering_interventions`, …), never against engine
internals. That is deliberate: it is what let the tests outlive the backbone they
were written under.

## Rules for adding one

1. **No backbone in the oracle.** If the ground truth imports anything from
   `neural/activations/`, it is not an oracle — it is the engine checked against
   itself.
2. **Assert on a public entry point.** Internals are free to change.
3. **Make the test non-vacuous.** Check that the intervention actually moved
   something before checking that it moved to the right place. Several bugs here
   have been caught by a "does this differ from no-intervention at all?" line —
   and one test was silently vacuous for a while because it patched at position 0,
   whose BOS activation is identical for every prompt.
4. **Pick the fixture that can fail.** Position/padding contracts need GPT-2;
   per-head width contracts need the GQA fixture.

## Related

- [`docs/TESTS.md`](TESTS.md) — the test tiers (`unit`, `property`,
  `numerical_unit`, `smoke`, `golden`) and what each may assert.
- [`causalab/neural/README.md`](../causalab/neural/README.md) — the engine these
  oracles check, and its ordering contract.
