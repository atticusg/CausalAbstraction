# NNsight 0.7 — Overview

> **Status:** reference — compiled from official sources (see §12), not code-verified against a local install.
> **Scope:** what nnsight is, its execution model, the full intervention API, local vs. remote (NDIF) execution,
> version history, footguns, and how it compares to forward hooks / TransformerLens / pyvene.
> **Version targeted:** nnsight **0.7.x** (current line; 0.6.x at 0.6.3). Notes flag version-sensitive items.
> **Relevance here:** causalab pins `nnsight>=0.5.9` (alongside `pyvene`); this doc is the shared reference for
> the nnsight half of the neural interface. See §11 for the causalab-specific notes.

---

## 0. At a glance

| | |
|---|---|
| **What it is** | A Python library for reading and intervening on the internals of PyTorch / HuggingFace models. Wraps a model so that arbitrary intermediate activations can be captured, edited, or differentiated through, with familiar PyTorch syntax. |
| **Core idea** | **Deferred execution.** A `with model.trace(...)` block does not run line-by-line; the body is captured and executed *interleaved with the model's forward pass*, in a worker thread, via auto-managed hooks. The model actually runs when the `with` block exits. |
| **Headline feature** | The **same code** runs locally on a small model or **remotely on a 70B–400B+ model hosted by NDIF** — toggle `remote=True`. Remote loads the model as meta tensors (zero local GPU). |
| **PyPI** | `nnsight` — `pip install nnsight`. Current **0.7.0**; `requires-python >=3.10`. |
| **Paper** | *"NNsight and NDIF: Democratizing Access to Foundation Model Internals"* — arXiv:2407.14561 (ICLR 2025). |
| **Maintainer** | NDIF (National Deep Inference Fabric) / David Bau lab, Northeastern University. |
| **License** | MIT. |
| **Position in the stack** | The **fine-grained primitive layer**. Higher layers build on it: `nnterp` (cross-architecture standardization + built-in methods), `nnsightful`/`workbench` (no-code viz), and **causalab** (causal-abstraction orchestration on nnsight + pyvene). |

---

## 1. The deferred-execution model

The central abstraction is a context manager that **defers and interleaves** execution:

```python
from nnsight import LanguageModel

model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)

with model.trace("The Eiffel Tower is in the city of"):
    # Intervene: zero the first block's residual output (in execution order!)
    model.transformer.h[0].output[0][:] = 0
    # Read: capture a later layer's hidden state, and the model output
    hidden = model.transformer.h[-1].output[0].save()
    out    = model.output.save()

print(model.tokenizer.decode(out.logits.argmax(dim=-1)[0]))
```

What is actually happening:

- The trace body is **captured and run in a worker thread, interleaved with the forward pass.** When the forward reaches a module you referenced, the worker is unblocked, your code for that hook point runs, and control returns to the forward.
- **From 0.5 onward you manipulate the real activation values, not opaque symbolic proxies.** Inside a trace, `module.output` is the actual tensor — `.shape`, slicing, arithmetic, and `print()` all work, and ordinary Python `if`/`for` work too. (Pre-0.5 used a symbolic proxy graph + a DSL; that is gone — see §6.)
- **Intermediate values are freed on `with`-exit unless saved.** Anything you want to read afterward must be `.save()`-ed (or `nnsight.save(...)`-ed). Accessing an unsaved value after the block raises.

### Module addressing

Module paths mirror the underlying PyTorch module tree — use the model's own names. Discover them with `print(model)`.

- GPT-2: `model.transformer.h[i]`, `.attn`, `.mlp`, `.ln_1`; final norm `model.transformer.ln_f`; unembed `model.lm_head`.
- Llama-style: `model.model.layers[i]`, `.self_attn`, `.mlp`; final norm `model.model.norm`; unembed `model.lm_head`.

> Transformer blocks usually return a **tuple**, so the residual stream is `...h[i].output[0]` / `...layers[i].output[0]`, not `.output`. (For a uniform `layers_output[i]` accessor across architectures, see `nnterp` — §11.)

---

## 2. Core API surface

### 2.1 Model wrappers

| Wrapper | For |
|---|---|
| `LanguageModel(repo_or_module, **hf_kwargs)` | HuggingFace causal LMs. Kwargs forward to HF loading: `device_map="auto"`, `torch_dtype=...`, `attn_implementation=...`, `dispatch=True` (load weights eagerly instead of lazily). |
| `NNsight(any_nn_module)` | Wrap an arbitrary `torch.nn.Module` — nnsight is **not** transformer-only. |
| `VisionLanguageModel(...)` | Multimodal (0.6+). |
| `DiffusionModel(...)` | Diffusion models (0.6+); pair with the `[diffusers]` extra. |
| `VLLM(..., mode="sync"\|"async")` | vLLM-backed serving with tracing (0.6+); pair with the `[vllm]` extra. |

### 2.2 The contexts

| Context | Purpose |
|---|---|
| `model.trace(input)` | One forward pass with interventions. |
| `model.scan(input)` | Shape/validation pass using PyTorch `FakeTensorMode` — **no real compute, no memory**. Still a tracing context, so `.save()` applies. Use it to resolve shapes before a real run. (Replaces the old `trace(..., scan=True, validate=True)` kwargs.) |
| `model.generate(input, max_new_tokens=N)` | Autoregressive decoding with interventions across generated tokens. |
| `model.session()` | Bundle multiple traces into one unit; traces run sequentially and can reference each other's values **without** `.save()`. The mechanism for multi-pass experiments and for collapsing N remote round-trips into one request. |
| `model.edit()` | Define **persistent** interventions applied on every future forward (§3.5). |

### 2.3 Reading and saving values

```python
import nnsight
with model.trace(prompt):
    h = model.model.layers[6].output[0]
    saved = nnsight.save(h)        # preferred form
    also  = h.save()               # back-compat form (see note)
```

- **`nnsight.save(obj)` is the preferred API.** `obj.save()` is a backwards-compatible shim (implemented by injecting a `.save()` method onto CPython's base `object` type; toggle with `CONFIG.APP.PYMOUNT`). It can be **silently shadowed** by objects that already define `.save()` (some torch classes), so prefer `nnsight.save(...)` in library code.
- `.input` / `.inputs` / `.output` access a module's input args / output at its hook point. Assigning to them is an intervention (§3).

### 2.4 Batching and multiple inputs — `invoke`

```python
with model.trace() as tracer:
    with tracer.invoke("The Eiffel Tower is in"):
        emb  = model.transformer.wte.output
        out1 = model.lm_head.output.save()
    with tracer.invoke("_ _ _ _ _ _"):
        model.transformer.wte.output = emb     # value from invoke #1
        out2 = model.lm_head.output.save()
```

- Each `tracer.invoke(prompt)` is a separate input, run in its own worker thread, in definition order.
- A **prompt-less** `tracer.invoke()` operates on the entire concatenated batch of all prior invokes.
- **Cross-invoke value passing requires a barrier.** Because invokes run concurrently up to a sync point, *setting* a value in one invoke from a value produced in another needs `tracer.barrier(n)` (see §3.3). Reading independent values does not.

### 2.5 Generation iteration

```python
with model.generate("Hello", max_new_tokens=5) as tracer:
    toks = nnsight.list().save()
    for _ in tracer.iter[:]:                       # cursor over generation steps
        toks.append(model.lm_head.output[0][-1].argmax(dim=-1))
```

- `tracer.iter[...]`, `tracer.all()`, `tracer.next()` move the cursor across decode steps.
- **Footgun:** an unbounded `tracer.iter[:]` blocks all code *after* the loop (it waits for a step that never comes). Bound it (`iter[:N]`) or put trailing work in a separate empty `tracer.invoke()`.

### 2.6 Other handles

- `tracer.result` — the final forward output of the traced call (cleaner than `model.output` / `model.generator.output`).
- `tracer.cache(modules=..., include_inputs=True)` — auto-collect activations for many modules; read after the trace as `cache[...]`. (Closest analog to TransformerLens `run_with_cache` / `ActivationCache`.)
- `tracer.stop()` — halt the forward after the layers you need (early exit; saves compute).
- `module.skip(value)` — substitute a module's output and **skip its compute**.
- `module.source` — reach operations **inside** a module's forward (e.g. the attention-probability matrix the block never returns) by AST-rewriting that forward; `module.source.<op>.output`. Inspect with `module.<...>.print_source()`.

---

## 3. Capabilities, with code

All snippets assume a loaded `model`. Replace module paths per architecture (§1).

### 3.1 Extract / save activations

```python
with model.trace(prompt):
    per_layer = [model.model.layers[i].output[0].save() for i in range(model.config.num_hidden_layers)]
    attn_out  = model.model.layers[10].self_attn.output[0].save()
    mlp_out   = model.model.layers[10].mlp.output.save()
```

### 3.2 Ablation

`[:] =` mutates the tensor in place; bare `=` replaces it.

```python
with model.trace(prompt):
    model.model.layers[10].output[0][:] = 0                 # zero a whole layer's residual
    model.model.layers[10].self_attn.output[0][:, -1, :] = 0  # last-token attention output only
```

Mean / noise ablation: capture, transform, write back.

```python
with model.trace(prompt):
    h = model.model.layers[-1].mlp.output.clone()
    model.model.layers[-1].mlp.output = h + 0.01 * torch.randn_like(h)
```

### 3.3 Activation patching / interchange (cross-prompt) — the causal-tracing primitive

Two invokes plus a barrier: capture from a clean run, paste into a corrupted run, measure the recovery.

```python
LAYER = 8
with model.trace() as tracer:
    barrier = tracer.barrier(2)
    with tracer.invoke(clean_prompt):
        clean_hs = model.model.layers[LAYER].output[0][:, -1, :]
        barrier()                                   # signal: clean_hs ready
    with tracer.invoke(corrupt_prompt):
        barrier()                                   # wait for clean_hs
        model.model.layers[LAYER].output[0][:, -1, :] = clean_hs
        patched = model.lm_head.output[:, -1, :].save()
    with tracer.invoke(corrupt_prompt):
        baseline = model.lm_head.output[:, -1, :].save()   # no patch
```

Sweep `LAYER` (and token position, or head) to build a causal map. This is the canonical denoising / IOI / causal-mediation recipe.

### 3.4 Steering (add a direction to the residual stream)

```python
with model.trace() as tracer:
    with tracer.invoke(prompt):
        base = model.lm_head.output[:, -1, :].save()
    with tracer.invoke(prompt):
        model.model.layers[LAYER].output[0][:, -1, :] += coef * direction
        steered = model.lm_head.output[:, -1, :].save()
```

No barrier needed here — the invokes write disjoint variables. `direction` is typically a contrastive mean-difference (ActAdd) or a probe direction.

### 3.5 Gradients / attribution

`tensor.backward()` is itself a tracing context. Access `.output` first, then `.grad` **inside** `backward()`, in reverse layer order. Gradients are also settable (edit/ablate grads).

```python
with model.trace(prompt):
    h = model.model.layers[-1].output[0]
    h.requires_grad_(True)
    loss = model.lm_head.output.sum()
    with loss.backward():
        g = h.grad.save()
        h.grad[:] = 0           # intervene on the gradient itself
```

Underpins gradient-based saliency and **attribution patching** (a linear approximation of activation patching: `corrupt_grad × (clean_act − corrupt_act)`, summed). Cheaper than patching (one backward vs. a forward per site) but approximate — validate by correlating against ground-truth patching on a subset.

### 3.6 Editing module outputs / persistent edits

```python
with model.edit() as edited:                 # non-destructive: returns an edited copy
    model.model.layers[0].output[0][:] = 0
# `edited` applies the edit on every future trace; `model` is untouched.

with model.edit(inplace=True):               # apply to `model` itself
    ...
model.clear_edits()                          # remove persistent edits
```

### 3.7 Logit lens

Call the final norm + unembed on an intermediate hidden state (ad-hoc module calls bypass normal hooks):

```python
with model.trace(prompt):
    h = model.model.layers[L].output[0]
    logits = model.lm_head(model.model.norm(h)).save()
    top    = logits.argmax(dim=-1).save()
```

### 3.8 Attention patterns / per-head

Attention probabilities are usually not returned by the block; reach them via `.source`, or enable eager attention. Per-head granularity is an in-trace reshape on the `(batch, heads, seq, seq)` / `(batch, heads, seq, d_head)` tensors.

### 3.9 Auxiliary modules (e.g. an SAE)

Wire an external module in and trace through it as a first-class submodule — capture a layer's activation, run it through the SAE, read/edit feature activations, and write the reconstruction back.

---

## 4. Local vs. remote (NDIF) execution

One API, two backends; the only code change is `remote=True`.

### 4.1 Local

```python
model = LanguageModel("openai-community/gpt2", device_map="auto", dispatch=True)
```

- Real weights load locally; runs on whatever device(s) you have (GPT-2-scale runs on CPU).
- Light install; the heavy transitive dep is just `torch>=2.4`. No NDIF account needed.

### 4.2 Remote (NDIF)

```python
from nnsight import CONFIG
CONFIG.set_default_api_key("<ndif-key>")    # from login.ndif.us; also set HF_TOKEN env var

model = LanguageModel("meta-llama/Llama-3.1-8B")   # model.device == "meta" — skeleton only
with model.trace("The Eiffel Tower is in the city of", remote=True):
    tok = model.lm_head.output[0][-1].argmax(dim=-1).save()
print(model.tokenizer.decode(tok))
```

- Model loads as **meta tensors** (zero local GPU). Your intervention code is serialized, executed on NDIF's hosted GPUs, and only the `.save()`-ed values are downloaded back.
- Hosts models up to **400B+** params; availability is tiered. Check with `nnsight.ndif_status()` / `nnsight.is_model_running("meta-llama/Llama-3.1-8B")` and nnsight.net/status. Request lifecycle: RECEIVED → QUEUED → DISPATCHED → RUNNING → COMPLETED.
- **Remote gotchas:**
  - `.save()` is the *transmission* mechanism. Appending to a list created **outside** the trace silently fails — create the list inside the trace and save it.
  - Move tensors to CPU before saving (`.detach().cpu().save()`) to minimize download.
  - Gradients are off by default remotely — set `requires_grad = True`.
  - A module whitelist applies; ship local helper code with `ndif.register(...)` (0.6+ serializes functions by source).
  - Use `model.session(remote=True)` to bundle a multi-pass experiment into a single queued request.
  - Remote historically wants Python 3.12.* and a recent nnsight.

> **Data-egress note:** `remote=True` sends your intervention graph and inputs to NDIF's shared infrastructure, which only hosts specific public models. Keep private models and data on local GPUs; treat NDIF as a public-model escape hatch, not a default.

---

## 5. Performance

- Tracing overhead is **negligible on real models**: ~0.3 ms fixed trace setup, ~0.03–0.2 ms per `.input`/`.output` access. On tiny micro-benchmark MLPs the ratio looks large (2–5×) but shrinks to ~1.5× by hidden-dim 2048 and is noise for billion-param GPU models in generation loops.
- **0.7 lazy one-shot hooks:** modules you never reference pay zero per-forward cost; self-removing hooks add +10–50% trace speedups on real models over 0.6.

---

## 6. Version history & API stability

nnsight is pre-1.0 — APIs still move. The relevant arc:

| Version | What changed |
|---|---|
| **0.4 → 0.5** *(the big break)* | Moved to a **thread-based interleaving** architecture: trace bodies are now ordinary Python over real tensors. The entire old intervention DSL was **removed** — `nnsight.cond`, `nnsight.list/dict/bool`, `session.iter`, `nnsight.local`, the `@nnsight.trace` decorator, and `nnsight.apply(fn, ...)` are gone (just write Python and call functions directly). `nnsight.save(x)` became preferred over `x.save()`. `model.scan(...)` replaced the `scan=`/`validate=` kwargs. `tracer.barrier(n)` introduced for cross-invoke value passing. |
| **0.6** *(Feb 2026)* | Remote custom-code execution (`ndif.register()`, serialize functions by source); 2.4–3.9× tracing speedups; `VisionLanguageModel` + `DiffusionModel`; full vLLM integration; cleaner tracebacks; first-class **AI-coding-agent support** (ships `CLAUDE.md` + the `ndif-team/skills` pack). **Removed the 0.4-compat layer.** `model.iter` / `model.all()` / `model.next()` now warn — use the `tracer.` equivalents. |
| **0.7** | Lazy one-shot hooks (zero overhead for untouched modules); `eproperty` promoted to a first-class **public extension API** for adding your own hookable values to a model. |

**Practical pin guidance:** code written to the **0.5+ idioms** (`nnsight.save`, `tracer.barrier`, `model.scan`, `tracer.iter`) runs on 0.6 and 0.7 unchanged. Avoid anything from the removed pre-0.5 DSL. causalab's `nnsight>=0.5.9` floor is on the right side of the break.

---

## 7. Footgun cheat-sheet

1. **Forgot `.save()`** → the value is freed on `with`-exit; reading it afterward errors. Save everything you need later.
2. **Wrong tuple index** → `block.output` is a tuple; the residual is `block.output[0]`.
3. **Out-of-order access** → within one invoke you must touch modules in **forward-pass order**, or the worker deadlocks (`OutOfOrderError`). Read "out of order" in a separate invoke.
4. **Cross-invoke set without a barrier** → setting a value in invoke B from a value made in invoke A needs `tracer.barrier(n)`; otherwise a `NameError`/race.
5. **Unbounded `tracer.iter[:]`** → blocks everything after the loop. Bound it or move trailing work to an empty invoke.
6. **`obj.save()` shadowed** → prefer `nnsight.save(obj)` in library code.
7. **Using a value's Python result mid-trace before it exists** → in 0.5+ real values are available at the hook point, but only once the forward has reached it; respect execution order (see #3).
8. **Remote list-append silently dropped** → create the list inside the trace and save it (§4.2).

---

## 8. Comparison to alternatives

| Approach | When you'd reach for it instead |
|---|---|
| **Raw PyTorch forward hooks** | A single trivial capture on a model you already loaded, with zero deps. Hooks must be registered/torn down manually, only fire at module boundaries (can't reach intermediate ops), and get unwieldy for cross-prompt patching, multi-step-generation interventions, or gradient interventions. nnsight auto-manages (lazy, self-removing) hooks, reaches intermediate ops via `.source`, makes no permanent edits, and scales to remote. |
| **HF `output_hidden_states=True`** | You only need residual-stream hidden states at layer boundaries and nothing else. It's read-only and exposes only what the model author opted into. nnsight gives uniform read **and write** access to any submodule and intermediate op. |
| **TransformerLens** | Curated transformer-circuit work on its supported architectures, with canonical hook names + `ActivationCache`. But it **reimplements** models into `HookedTransformer` (possible numerical drift, limited architecture set, local-only). nnsight wraps the **original HF model** in place (no drift, any architecture), adds write/edit/grad/skip/generation interventions, and offers the local↔remote path. `tracer.cache(...)` is its `run_with_cache` analog. |
| **pyvene** *(used by causalab)* | Declarative, config-driven interventions and a typed intervention abstraction (good for systematic interchange-intervention / DAS sweeps). nnsight is lower-level and more imperative — better for ad-hoc reads, custom multi-site interventions, gradients, and intermediate-op access. causalab uses **both** (pyvene 0.1.8 + nnsight ≥0.5.9); they are complementary, not competing. |

---

## 9. Install & dependencies

```bash
pip install nnsight            # core
pip install "nnsight[vllm]"    # + vLLM serving
pip install "nnsight[diffusers]"   # + diffusion models
pip install "nnsight[all]"
```

- `requires-python >=3.10` (NDIF remote historically wants 3.12.*).
- Core deps: `torch>=2.4.0`, `transformers`, `accelerate`, `pydantic>=2.9.0`, plus `astor`, `cloudpickle`, `httpx`, `python-socketio[client]`, `toml`, `ipython`, `rich`, `zstandard`.
- No GPU required to *install*; you only need hardware to run whatever model you load.

---

## 10. The ecosystem above nnsight (orientation)

- **`nnterp`** (ndif-team, MIT) — a cross-architecture standardization layer + built-in interpretability methods on top of nnsight. The most directly useful neighbor for causalab; **see §11 for full detail.**
- **`ndif-team/skills`** — the nnsight team's agent-skills pack (Claude Code + Codex): `nnsight-basics`, `logit-lens`, `activation-patching`, `attribution-patching`, `causal-tracing`, `model-steering`. Good prior-art reference, but currently **unlicensed** (all-rights-reserved) — reference it, don't copy its prose.
- **`nnsightful` → `workbench`** — higher-level method library + no-code React viz UI (education/exploration).

---

## 11. nnterp — standardized cross-architecture interface

`nnterp` is a thin layer **on top of** nnsight (by Clément Dumas / "Butanium", now under the ndif-team org) that solves nnsight's one real friction for cross-family work: module paths differ per architecture (`gpt2: transformer.h[i]` vs. `llama: model.layers[i]`). nnterp **renames every architecture to one LLaMA-like scheme** and ships built-in methods + per-model validation — **without reimplementing the model** (it keeps the original HF weights/impl, unlike TransformerLens, so there is no numerical drift).

| | |
|---|---|
| **PyPI / version** | `nnterp` — current **1.3.0** (first 1.0.0 Oct 2025; ~7 releases by Feb 2026). |
| **License** | MIT. |
| **Requires** | `requires-python >=3.10`, **`nnsight>=0.6`** (open upper bound → admits 0.7), `transformers` (see compat note below). No `torch` pin (transitive). |
| **Paper** | *"nnterp: A Standardized Interface for Mechanistic Interpretability of Transformers"* — arXiv:2511.14465. |
| **Docs** | ndif-team.github.io/nnterp. nnsight 0.6 ships an official "NNterp Integration" — `StandardizedTransformer` is a first-class nnsight citizen (remote NDIF works through it). |
| **Lock-in** | Low — it **does not hide nnsight**. You still `model.trace(...)` with nnsight semantics and can reach unrenamed modules / `.grad` directly. |

### 11.1 Loading & canonical naming

```python
from nnterp import StandardizedTransformer
model = StandardizedTransformer("gpt2")                      # or "meta-llama/Llama-3.1-8B", "Qwen/Qwen3-8B", …
print(model.num_layers, model.hidden_size, model.num_heads, model.vocab_size)
```

Every architecture is mapped to the same module tree:

```
StandardizedTransformer
├── embed_tokens
├── layers
│   ├── self_attn
│   └── mlp
├── ln_final
└── lm_head
```

### 11.2 Uniform accessors (the headline value)

The same code reads internals on GPT-2 and Llama alike:

```python
with model.trace("The Eiffel Tower is in the city of"):
    layer_5  = model.layers_output[5]
    attn_3   = model.attentions_output[3]
    mlp_3    = model.mlps_output[3]
    logits   = model.logits.save()
```

Per layer index `i`: `layers[i]` / `layers_input[i]` / `layers_output[i]`; `attentions[i]` / `attentions_input[i]` / `attentions_output[i]`; `mlps[i]` / `mlps_input[i]` / `mlps_output[i]`; plus `embed_tokens` / `token_embeddings`, `ln_final`, `lm_head`, `logits`.

**Intervene by assignment**, exactly like raw nnsight (so activation patching, ablation, and grafting all carry over — just with stable names):

```python
with model.trace("Hello world"):
    model.layers_output[10] = model.layers_output[10] + layer_3_output   # graft / patch
```

### 11.3 Attention probabilities & per-head

Opt in at load (forces eager attention); per-head is inherent in the `(batch, heads, seq, seq)` tensor:

```python
model = StandardizedTransformer("gpt2", enable_attention_probs=True)
with model.trace("..."):
    probs = model.attention_probabilities[5].save()      # (batch, heads, seq, seq)
    probs[:, :, :, 0] = 0                                 # ablate attention to the first token
    probs /= probs.sum(dim=-1, keepdim=True)              # renormalize
# model.attention_probabilities.print_source()           # debug which module it hooks
```

Attention-prob access is provided "for models that support it" — it is **not** universal.

### 11.4 Built-in methods

```python
from nnterp.interventions import logit_lens, patchscope_lens, TargetPrompt, repeat_prompt

probs = logit_lens(model, ["The capital of France is"])                  # (batch, layers, vocab)

target = TargetPrompt("city: Paris\nfood: croissant\n?", -1)             # or repeat_prompt(words=[...])
probs  = patchscope_lens(model, source_prompts=[...], target_patch_prompts=target, layers=[3,5,7])

with model.trace(["The weather today is", "I feel very"]):
    model.steer(layers=[1, 3], steering_vector=v, factor=0.5)            # layer_output += factor * v
    model.steer(layers=1, steering_vector=v, batch_index=1, token_positions=[0, 1])
```

Also: `model.project_on_vocab(hidden)` (logit-lens primitive), `model.skip_layer(i)` / `model.skip_layers(a, b, skip_with=saved_act)` (layer skipping / activation grafting), and `Prompt.from_strings(...)` + `run_prompts(...)` for target-token tracking.

**Not shipped as named helpers:** dedicated ablation, gradient/attribution, or activation-patching functions. Ablation = assign zeros; patching = assign a saved activation; gradients = raw nnsight `.grad` within the trace. So the curated surface is `logit_lens` / `patchscope_lens` / `steer` / `project_on_vocab` / `skip_layers`; everything else is hand-rolled nnsight **on top of the standardized accessors**.

### 11.5 Cross-architecture coverage & validation

- **Coverage** (paper): write once, run across **50+ model variants / 16 architecture families** — named families include GPT-2, GPT-J, Llama (all), Mistral, Mixtral (MoE), **Gemma 2 & 3**, **Qwen 2 & 3**, Phi-3, OPT, "and more." (Coverage is by *family*, not an exhaustive checkpoint list — e.g. Gemma **1** is not named.) Adding an architecture is a `rename_config`, **not** a reimplementation.
- **Validation** (the genuinely distinctive feature, and a direct fit for causalab's verification culture): on load, `StandardizedTransformer` auto-runs tests that check (1) module renaming conforms to the scheme, (2) layer outputs are `(batch, seq, hidden)`, and (3) — when enabled — attention probs are `(batch, heads, seq, seq)`, rows normalize to 1, and edits actually affect output. Run explicitly against your own checkpoints:

```bash
python -m nnterp run_tests --model-names "gpt2" "meta-llama/Llama-3.1-8B"
python -m nnterp run_tests --class-names "LlamaForCausalLM" "GPT2LMHeadModel"
```

`allow_dispatch=False` skips the tests that need a forward dispatch. The project itself flags two limits: it cannot guarantee attention probs are unmodified by later transforms, and renaming "edge cases might behave differently" — so validate a non-standard/custom checkpoint before trusting interventions.

### 11.6 Constructor

```python
StandardizedTransformer(
    model, trust_remote_code=False, check_renaming=True, remote=False,
    allow_dispatch=True, enable_attention_probs=False,
    check_attn_probs_with_trace=True, allow_multimodal=False,
    rename_config=None, **kwargs)
```

### 11.7 Caveats / maturity

- **Young.** 1.x line (1.0.0 Oct 2025 → 1.3.0 Feb 2026). Actively maintained, ndif-team-backed, MIT, published paper — **early-stable for research use**, not a battle-hardened dependency.
- **`transformers` is the real compatibility risk.** PyPI metadata leaves `transformers` unpinned, but the load-time validation is version-sensitive and a community note reports it is effectively tested against **transformers ~4.53.x**. Pin `transformers` to a tested version and verify before bumping — this is the pin most likely to break load.
- **Decoupled cadence.** nnterp ships independently of NDIF ("you always run the version you have locally"), so it does **not** force an older nnsight; the floor is 0.6.
- **No grad/attribution helpers** — drop to raw nnsight for those (it doesn't hide nnsight, so this costs nothing architecturally).

### 11.8 When to reach for nnterp (vs. raw nnsight)

- **Use nnterp** when intervention code must run across model families (Llama/Qwen/Gemma/GPT-2/Mistral/Phi), when you want `logit_lens`/`patchscope`/`steer` off the shelf, or when its per-model validation reinforces a correctness gate.
- **Stay on raw nnsight** for gradient/attribution work, exotic/custom checkpoints nnterp can't standardize, or attention-prob access on unsupported architectures.
- **Default pattern:** nnterp for the common path, raw nnsight as the documented escape hatch — they share the same trace semantics.

---

## 12. Relevance to causalab

- causalab depends on **`nnsight>=0.5.9`** and **`pyvene`** (git `main`); see the root `pyproject.toml`. The 0.5+ idioms in this doc are the ones to use; avoid removed pre-0.5 DSL.
- nnsight is the imperative, fine-grained primitive for reads and custom interventions; **pyvene** (0.1.8) is the declarative intervention layer the path-patching method is built on (see `docs/PATH_PATCHING.md`). They coexist by design.
- For systematic interchange interventions across model families, consider whether `nnterp`'s standardized accessors + validation tests (§11) would reduce per-architecture branching in `causalab/neural/`. It sits cleanly on the `nnsight>=0.6` line — but note causalab's current floor is `nnsight>=0.5.9`, so adopting nnterp would bump the nnsight floor to 0.6 and add a `transformers`-version constraint (§11.7) to verify against pyvene 0.1.8's own `transformers` needs.
- If you ever need a model too large for your local GPUs, `remote=True` is the escape hatch — but heed the data-egress note in §4.2 for private models.

---

## 13. Sources

- **nnsight** repo: `github.com/ndif-team/nnsight` — `README.md`, `CLAUDE.md` (agent task taxonomy + gotcha cheat-sheet), `NNsight.md` (architecture deep-dive), `docs/patterns/` cookbook.
- nnsight docs: nnsight.net/documentation, /tutorials, /features; "Introducing NNsight 0.6" blog (2026-02-26); "NNsight 0.5 Changes" walkthrough.
- PyPI metadata for `nnsight` 0.7.0; GitHub release notes. Paper: arXiv:2407.14561 (ICLR 2025).
- **nnterp** (§11): `github.com/ndif-team/nnterp` (`README.md`, `pyproject.toml`), docs ndif-team.github.io/nnterp (basic-usage, interventions, model-validation, api), PyPI `nnterp` 1.3.0, nnsight.net "NNterp Integration" blog (2026-02-26), paper arXiv:2511.14465.

> Implementation specifics (e.g. the `PYMOUNT` `.save()` shim, exact dependency pins, lazy-hook internals) are
> drawn from the architecture deep-dive and package metadata and may evolve across patch releases — verify
> against the version actually installed before relying on an edge-case detail.
