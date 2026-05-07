# Interactive Elicitation: `set_up_method.md`

This file is read by `/setup-methods` Step 1 when no spec paths are provided (single-spec interactive elicitation fallback). Walk the user through the 5 sections of `SET_UP_METHOD_TEMPLATE.md` one at a time, getting approval after each section.

Tell the user up front:

> "I'll guide you through creating a method spec. We'll fill out 5 sections (identity, surface, dependencies, hyperparameters, side effects). I'll save the draft to `${SESSION_DIR}/code/methods/<name>/set_up_method.md` as we go and ask for approval after each section."

While iterating, write the partial draft to `${SESSION_DIR}/code/methods/<name>/set_up_method.md` so the user can see the cumulative result. Promote nothing to `${SESSION_DIR}/code/methods/<name>/<name>.py` until Step 3 of the main workflow.

---

## Section 1 — Identity

**Ask:** "What's a snake_case name for this method?"

Examples: `logit_diff`, `belief_norm_loss`, `attention_head_score`, `geodesic_distance_estimator`.

Validate: must be `[a-z][a-z0-9_]*`, must not collide with `causalab/methods/<name>/` or `causalab/methods/<name>.py`. If it collides, ask again.

**Ask:** "In one paragraph, what does this method *compute*, and what is it for?" Frame as a primitive (featurizer / distance / scorer / training loop), not a research question.

Examples:
- *"Computes the per-head logit difference between a base prompt and a counterfactual prompt at a specified layer. Used to attribute behavior to individual attention heads."*
- *"Fits a TPS spline to a set of class centroids in the activation manifold. Returns the trained spline parameters; downstream callers use it for projection and reconstruction."*

---

## Section 2 — Surface

**Ask:** "What's the callable's signature? Function or class?"

Provide a worked example so the user can adapt it:

```python
def <name>(
    activations: torch.Tensor,        # shape (B, T, D)
    target_token_ids: torch.Tensor,   # shape (B,), int64
    *,
    layer: int,
    head: int,
) -> dict[str, torch.Tensor]:
    """Returns {'logit_diff': shape (B,) float32}."""
```

Make sure every input has a documented shape (or a clear "scalar"/"sequence of strings") and every output is named. The signature lines up with what `/setup-analyses` will see when it lists §3 dependencies.

**Reminder to the user:** hyperparameters (anything that changes behavior) must NOT have default values in the signature — they belong in §4 with no defaults, so the consuming Hydra config can resolve them.

---

## Section 3 — Dependencies

**Ask:** "Which `causalab/` symbols does this method import?"

Walk through plausible candidates by area:

- **`causalab.neural`** — `LMPipeline`, `AtomicModelUnit`, `ResidualStream`, intervention modes (collect/interchange/replace/interpolate), `IntervenableModel`.
- **`causalab.methods`** — distances, metric functions, existing featurizers, training loops, scoring primitives.
- **`causalab.io`** — only `causalab.io.artifacts` save/load is used by methods *and only* for tensor primitives that don't smell like analysis-layer responsibilities (rare; usually leave I/O to the analysis).
- **`causalab.causal`** — `CausalModel`, `Mechanism`, `Trace`.
- **`causalab.tasks`** — `load_task`, task module accessors.

**Forbidden** (skill will refuse): anything under `causalab.runner.*` or `causalab.analyses.*`.

If the user names a symbol that doesn't exist, search `causalab/` for the closest match and offer it as a suggestion. If the closest match is in `analyses/` or `runner/`, this is a layering smell — ask whether the responsibility actually belongs in a method, or whether the method should consume an in-memory tensor that the analysis prepared.

---

## Section 4 — Hyperparameters

**Ask:** "List every knob that changes the method's behavior — type, range, and what it controls. None of these get a default in the signature; defaults live in the consuming Hydra config."

Provide a worked example table format (see SET_UP_METHOD_TEMPLATE.md §4).

Common categories:
- Subspace dimensions (`k_features`).
- Training-loop knobs (`learning_rate`, `n_epochs`, `batch_size`).
- Regularization coefficients (`reg_coef`, `smoothness`).
- Component selectors (`layer`, `head`, `token_position_name`).
- Numerical knobs (`temperature`, `eps`, `tolerance`).

For each, ask the user to confirm the type and range. Record proposed defaults *for the analysis's `analysis.yaml`* in a "Notes" line — the method itself takes no defaults.

---

## Section 5 — Side Effects

**Ask:** "Does this method only return an in-memory dict, or does it have any side effects?"

Default expectation: **no side effects**. If the user describes:

- Saving a tensor / writing JSON / writing a checkpoint → push back; that belongs in the analysis layer (invariant 4). Offer to refactor: the method returns the tensor; the analysis saves it.
- Mutating an input object (e.g. overwriting a featurizer's rotation) → record the in-memory mutation in §5 explicitly so consumers know.
- Running a forward pass through a `pipeline` → not a side effect; that's normal.

If the user insists on disk I/O at the method layer, file an issue via `/document-issues` describing the layering exception so it surfaces at promotion time.

---

## Save the spec

After all 5 sections are approved:

1. Combine into a single markdown file matching `SET_UP_METHOD_TEMPLATE.md`. Frontmatter `name: <name>`.
2. Write to `${SESSION_DIR}/code/methods/<name>/set_up_method.md` (creating the directory if needed).
3. Tell the user: **"Spec saved at `${SESSION_DIR}/code/methods/<name>/set_up_method.md`. Proceeding to scaffold."**
4. Continue with Step 2 (approval checkpoint) of the main workflow.
