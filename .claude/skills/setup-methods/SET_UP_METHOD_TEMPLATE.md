---
name: <method_name_in_snake_case>
---

# Method spec: `<method_name>`

5-section spec consumed by `/setup-methods`. Lands at `${SESSION_DIR}/code/methods/<name>/set_up_method.md` after the skill writes it. The skill scaffolds code from this spec; downstream `/setup-analyses` reads it to know what symbol to import.

---

## §1. Identity

**Purpose** (one paragraph): what does this method compute, and what is it for? Frame it in terms of an interpretability primitive — a featurizer, a distance, a scorer, a training loop. Avoid research-question phrasing (that belongs in the consuming analysis, not here).

> Example: *Computes the per-head logit difference between a base prompt and a counterfactual prompt at a specified layer. Used by an analysis to attribute behavior to individual attention heads.*

---

## §2. Surface

The single callable that consumers import. State its signature with full type annotations, including input shapes/dtypes and output structure.

```python
def <method_name>(
    activations: torch.Tensor,        # shape (B, T, D), float32 or float16
    target_token_ids: torch.Tensor,   # shape (B,), int64
    *,
    layer: int,
    head: int,
) -> dict[str, torch.Tensor]:
    """Returns {'logit_diff': shape (B,) float32}."""
```

If a class is more natural than a function, declare the class signature with the public methods that consumers will call.

**Rule:** every keyword argument that is a *hyperparameter* (subspace dim, learning rate, regularization coefficient, batch size, …) must have **no default value**. Defaults live in Hydra (`ARCHITECTURE.md` §3 invariant 5).

---

## §3. Dependencies

List every symbol imported from `causalab/`:

| Symbol | Source | Why |
|---|---|---|
| `LMPipeline` | `causalab.neural.pipeline` | wraps the model + tokenizer |
| `compute_reference_distributions` | `causalab.methods.metric` | reused for class centroids |
| `save_tensor_results` | `causalab.io.artifacts` | (only if the analysis needs to be reminded) |

Plus third-party imports (torch, numpy, …).

**Forbidden:** anything under `causalab/runner/` or `causalab/analyses/` (invariants 1, 2). The skill will refuse to scaffold if §3 lists a forbidden symbol.

---

## §4. Hyperparameters

Every knob that changes the method's behavior. None of these gets a default in the function signature — they are passed explicitly by the caller.

| Name | Type | Range / values | Description |
|---|---|---|---|
| `layer` | `int` | `0..n_layers-1` | which transformer layer to read activations from |
| `head` | `int` | `0..n_heads-1` | which attention head |
| `temperature` | `float` | `> 0` | softmax temperature for the reference distribution |

Defaults for these go in the consuming analysis's `analysis.yaml`, not here.

---

## §5. Side effects

Must be `None`. The method returns an in-memory results dict; the consumer (an analysis) decides where to persist them.

If the method is doing something stateful (training a model, optimizing in place), document the in-memory side effect (e.g. "mutates the input `featurizer` by overwriting its rotation matrix"). Disk side effects are not allowed at this layer (invariant 4).

---

## Notes (optional)

Free-form section for things the spec author wants the implementation agent to know — references to similar shipped methods to model from, known performance pitfalls, expected runtime per call, etc.
