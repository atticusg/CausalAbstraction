"""Reach the two things nnsight's `.source` could not before in Qwen3.6-35B-A3B:
the DeltaNet recurrent state as it builds up, and the individual expert outputs.

Needs the patched ~/wd/nnsight (0.8 branch). The model is 72 GB in bf16, so it is
split between the GPU and CPU memory (device_map="auto" with a GPU cap so the
offloaded experts have room to page in); every forward streams them through, so
each section takes a minute or two.

NDIF's worked example, adopted near-verbatim as causalab's N0 verification run
(notes/causalab-nnsight-engine-plan.md §6-N0, §7.3). One deviation: the GPU cap
this docstring already describes is made explicit below (`max_memory`) — on a
single 80 GB H100 an uncapped device_map="auto" leaves no headroom for
activations and expert paging.

Deviation 2 (measured, 2026-08-28, both on the tiny fixture and on the real
35B here): on transformers 5.16.1 the delta-rule kernels are themselves behind
a dispatcher — `torch_chunk_gated_delta_rule_0.source` shows only
`implementation_0` etc., not the kernel body — so every kernel drill peels one
more level (`.source.implementation_0.source`) than NDIF's original, which ran
against a transformers where the kernel was called directly.
"""

import sys

import torch
import nnsight
from nnsight.modeling.transformers import TransformersModel

model = TransformersModel(
    "Qwen/Qwen3.6-35B-A3B", task="text-generation", dtype=torch.bfloat16,
    device_map="auto", max_memory={0: "60GiB", "cpu": "400GiB"},
)
# Reading only: with grad on, autograd would keep every paged-in expert weight
# alive for the backward graph and the offloaded layers would pile up on the GPU.
torch.set_grad_enabled(False)
prompt = "The quick brown fox jumps over the lazy dog. " * 20   # ~200 tokens = 4 chunks of 64


# ---------------------------------------------------------------------------
# 1. The DeltaNet recurrent state, per chunk
# ---------------------------------------------------------------------------
# In prefill, transformers runs the chunked delta rule: the state S is advanced
# once per 64-token chunk by `last_recurrent_state = (...)`, a product, never a
# call's return value. `.source` names that assignment `last_recurrent_state_1`
# (`_0` is the zero init before the loop), and it fires once per chunk, so
# `tracer.iter` picks the chunk. How many chunks there are is read from the loop
# itself: its `range(0, total_sequence_length // chunk_size)` is the op `range_1`
# (`range_0` is the loop inside a chunk), and it fires just before the loop does.
# Layer 0 is a linear-attention layer.
linear_attn = model.model.layers[0].linear_attn
# print(linear_attn.source)      # every call and assignment in forward, labelled
kernel_op = next(n for n in linear_attn.source.names if "chunk_gated_delta_rule" in n)

with model.trace(prompt) as tracer:
    # drill into the kernel function (through the 5.16.1 dispatcher — deviation 2)
    kernel = getattr(linear_attn.source, kernel_op).source.implementation_0.source
    n_chunks = nnsight.save(len(kernel.range_1.output))      # the loop's own range(...)
    states = nnsight.save([])                                # save the container, append raw values
    for _ in tracer.iter[:n_chunks]:                         # one fire per chunk
        states.append(kernel.last_recurrent_state_1.output)
    final = nnsight.save(getattr(linear_attn.source, kernel_op).output[1])  # what the cache stores
    clean = nnsight.save(linear_attn.output)
print("state after each chunk:", len(states), "x", tuple(states[0].shape), "(heads, k_dim, v_dim)")
print("last one is the cached final state:", torch.equal(states[-1], final))

# The same location is writable: zero the state after chunk 0 and the tokens of
# later chunks change, chunk 0's don't.
with model.trace(prompt) as tracer:
    kernel = getattr(linear_attn.source, kernel_op).source.implementation_0.source
    for _ in tracer.iter[0]:
        kernel.last_recurrent_state_1.output[:] = 0
    patched = nnsight.save(linear_attn.output)
changed = (patched != clean).any(-1)[0]
print("tokens changed by zeroing S after chunk 0:", changed[:64].sum().item(), "of 64 |",
      changed[64:].sum().item(), "of", changed[64:].numel())


# ---------------------------------------------------------------------------
# 2. Expert outputs
# ---------------------------------------------------------------------------
# transformers wraps Qwen3_5MoeExperts.forward in a dispatcher that picks an
# implementation at run time (grouped_mm by default; the eager loop the class
# defines never runs). `experts.source` shows that dispatch, and
# `experts_forward_1` is the call that actually ran -- drill into it.
experts = model.model.layers[0].mlp.experts
top_k = model.config.num_experts_per_tok

# (a) default grouped_mm: all (token, slot) pairs go through one fused matmul,
#     so the per-expert outputs come back as one (seq * top_k, d) tensor.
with model.trace(prompt):
    impl = experts.source.experts_forward_1.source            # requests go in the order the ops run
    sorted_rows = nnsight.save(impl.torch_sort_0.output)      # (expert id per row, row -> token*top_k+slot)
    by_expert = nnsight.save(impl.proj_out_3.output)          # down-projection, rows sorted by expert (_2 is the untaken else branch)
    by_slot = nnsight.save(impl.weighted_out_1.output)        # x router weight, back in token order
expert_ids, perm = sorted_rows
expert_out = by_slot.view(-1, top_k, by_slot.shape[-1])       # the diagram's expert_out (seq, 8, d)
first = expert_ids.unique()[0]                                # lowest expert id that got a token
rows = expert_ids == first
grouped_first = by_expert[rows][(perm[rows] // top_k).argsort()]  # that expert's outputs, in token order
print("grouped_mm: expert_out", tuple(expert_out.shape), f"| expert {first.item()} unweighted rows:", tuple(grouped_first.shape))

# (b) eager: one Python loop iteration per expert that got any token, so the
#     loop's assignments fire once per expert and `tracer.iter` picks the expert.
#     The count comes from the loop's own `expert_hit = (...).nonzero()`.
model.set_experts_implementation("eager")
with model.trace(prompt) as tracer:
    loop = experts.source.experts_forward_1.source
    n_hit = len(loop.nonzero_0.output)
    per_expert = nnsight.save([])
    for _ in tracer.iter[:n_hit]:
        top_k_pos, token_idx = loop.torch_where_0.output      # which tokens this expert serves
        per_expert.append((token_idx,
                           loop.current_hidden_states_0.output,    # silu(gate) * up
                           loop.current_hidden_states_1.output))   # down_proj output
token_idx, hidden, out = per_expert[0]                        # iteration 0 is the lowest hit expert id
print(f"eager: {len(per_expert)} experts hit; expert {first.item()} served {token_idx.numel()} tokens,",
      "hidden", tuple(hidden.shape), "out", tuple(out.shape))
diff = (out[token_idx.argsort()].float() - grouped_first.float()).abs().max() / grouped_first.float().abs().max()
print(f"eager vs grouped_mm for that expert: max relative difference {diff.item():.2e} (bf16 kernels)")
model.set_experts_implementation("grouped_mm")


# ---------------------------------------------------------------------------
# 3. The same state during generate()
# ---------------------------------------------------------------------------
# Under generate, step 0 (prefill) runs the chunked kernel (one fire of its loop
# state per chunk) and every decode step runs the *recurrent* kernel once (1 fire).
# Fires are counted per location, so `tracer.iter[k]` binds the FIRST read in its
# body to the k-th fire of that read's location, and later reads follow the model
# in order. Two consequences:
#   - the chunk states are just fires 0..n-1 of the chunk kernel's state, all in
#     step 0 -- no nesting needed;
#   - a decode-only op has no fire in step 0, so its fire k is step k+1. Read a
#     per-step module location first (`linear_attn.input`) to make `step` mean
#     step, then the kernel reads line up on their own.
# (Nesting works too: `for step in tracer.iter[:N]:` with `for c in tracer.iter[:n]:`
# inside the step-0 branch -- the inner loop restores the outer pin on exit.)
rec_op = next(n for n in linear_attn.source.names if "recurrent_gated_delta_rule" in n)
with model.generate(prompt, max_new_tokens=3, do_sample=False) as tracer:
    kernel = getattr(linear_attn.source, kernel_op).source.implementation_0.source
    chunk_states, decode_states, handed_in = nnsight.save([]), nnsight.save([]), nnsight.save([])
    for _ in tracer.iter[:len(kernel.range_1.output)]:        # prefill: one fire per chunk
        chunk_states.append(kernel.last_recurrent_state_1.output)
    for step in tracer.iter[1:3]:                             # decode steps 1, 2
        _ = linear_attn.input                                 # anchor this body to step `step`
        rec = getattr(linear_attn.source, rec_op)
        handed_in.append(rec.inputs[1]["initial_state"].clone())  # the cache's buffer: clone, it is updated in place
        decode_states.append(rec.source.implementation_0.source.last_recurrent_state_2.output)  # after this token's update
    generated = nnsight.save(tracer.result)
print("generate:", repr(model.tokenizer.decode(generated[0, -3:])), "|", len(chunk_states), "chunk states +", len(decode_states), "decode states")
chain = [chunk_states[-1], *decode_states[:-1]]               # what each decode step should start from
print("cache continuity (state handed to step k == state after step k-1):",
      all(torch.allclose(a, b) for a, b in zip(handed_in, chain)))


# ---------------------------------------------------------------------------
# 4. Attention scores and pattern need attn_implementation="eager"
# ---------------------------------------------------------------------------
# The full-attention layers (3, 7, 11, ...) call whatever attention_interface the
# config selects. Under the default "sdpa" that is a fused kernel: the scores and
# the softmax'd pattern are never materialised in Python, so there is nothing to
# hook and the interface returns attn_weights=None. Under "eager" the interface is
# eager_attention_forward, whose assignments name them: attn_weights_0 (q k^T *
# scale), attn_weights_1 (+ mask), attn_weights_2 (softmax -- the diagram's
# `pattern`), attn_output_0 (the weighted values, the diagram's `z`).
self_attn = model.model.layers[3].self_attn
with model.trace(prompt):
    ops = nnsight.save(self_attn.source.attention_interface_1.source.names)
    weights = nnsight.save(self_attn.source.attention_interface_1.output[1])
print(f"{model.config._attn_implementation}: softmax ops {[n for n in ops if 'softmax' in n]}, attn_weights returned: {weights}")

model.set_attn_implementation("eager")
with model.trace(prompt):
    attn = self_attn.source.attention_interface_1.source
    scores = nnsight.save(attn.attn_weights_0.output)      # (batch, heads, seqQ, seqK) before the mask
    pattern = nnsight.save(attn.attn_weights_2.output)     # after softmax
print(f"eager: scores {tuple(scores.shape)}, pattern {tuple(pattern.shape)}, rows sum to 1:",
      torch.allclose(pattern.float().sum(-1), torch.ones(pattern.shape[:-1], device=pattern.device), atol=1e-2))
