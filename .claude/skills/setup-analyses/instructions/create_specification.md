# Interactive Elicitation: `set_up_analysis.md`

This file is read by `/setup-analyses` Step 1 when no spec paths are provided (single-spec interactive elicitation fallback). Walk the user through the 5 sections of `SET_UP_ANALYSIS_TEMPLATE.md` one at a time, getting approval after each section.

Tell the user up front:

> "I'll guide you through creating an analysis spec. We'll fill out 5 sections (identity, position in DAG, methods used, config schema, outputs). I'll save the draft to `${SESSION_DIR}/code/analyses/<name>/set_up_analysis.md` as we go and ask for approval after each section."

While iterating, write the partial draft to `${SESSION_DIR}/code/analyses/<name>/set_up_analysis.md` so the user can see the cumulative result.

---

## Section 1 — Identity

**Ask:** "What's a snake_case name for this analysis?"

Validate: must be `[a-z][a-z0-9_]*`, must not collide with `causalab/analyses/<name>/` or `causalab/configs/analysis/<name>.yaml`. If it collides, ask again.

**Ask:** "What research question does this analysis answer? Phrase it as a single sentence — it'll be italicized in the README."

Examples:
- *"Which attention heads carry the variable's representation?"*
- *"Does the manifold preserve causal distances under linear interventions?"*
- *"How does the embedding trajectory deviate from a geodesic when we ablate a head?"*

**Ask:** "In one paragraph, what does the analysis do mechanically? What does it read, what does it produce, and where does it sit in the pipeline?"

If the user struggles to compress this, offer the worked example from `SET_UP_ANALYSIS_TEMPLATE.md` §1 and ask them to adapt it.

---

## Section 2 — Position in the DAG

**Ask Part 1:** "Which upstream artifacts does this analysis read?"

Walk through the shipped DAG (see `ANALYSIS_GUIDE.md`):

- Reads from `baseline` → typically `per_class_output_dists.safetensors`.
- Reads from `locate` → typically the `best_cell` from `locate/<method>/<variable>/results.json` (often via auto-discovery, value `null`).
- Reads from `subspace` → rotation tensors, training features.
- Reads from `activation_manifold` / `output_manifold` → `manifold_spline/ckpt_final.pt`.

Record each one with: artifact name, producing analysis, full path under `${experiment_root}/...`, and whether to use auto-discovery (`null` in the eventual `analysis.yaml`) or a hardcoded subdir.

If the analysis depends on no upstream artifacts (rare — only the case for `attention_pattern`-style standalone analyses), say so explicitly.

**Ask Part 2:** "Which downstream analyses, if any, will read this analysis's outputs?"

- `null` if terminal (most session-local analyses are terminal — they're the leaves of the experiment).
- A list of analysis names if the outputs feed something else (e.g. a session-local `head_pruning` that reads a session-local `head_logit_diff`).

---

## Section 3 — Methods used

**Ask:** "Which method symbols (callables) does the analysis invoke? List each one with its source."

Walk the user through the available method libraries:

- **`causalab.methods.metric`** — `compute_base_accuracy`, `compute_reference_distributions`, loss helpers.
- **`causalab.methods.distances`** — Hellinger / Wasserstein / Fisher-Rao / conformal-geodesic distances.
- **`causalab.methods.interchange`** — interchange-mode primitives.
- **`causalab.methods.steer`** — steering + path-collection primitives.
- **`causalab.methods.scores`** — coherence, isometry, distance-from-manifold scoring.
- **`causalab.methods.{pca, sae, umap, standardize, noise}`** — featurizer subclasses.
- **`causalab.methods.{flow, spline}`** — manifold builders.
- **`causalab.methods.pullback`** — geodesic / LBFGS optimizers.
- **`methods.<name>` (session-local)** — anything scaffolded by `/setup-methods` in this session.

For each session-local method listed:
1. Confirm the file `${SESSION_DIR}/code/methods/<name>/<name>.py` exists.
2. If it doesn't, advise the user to run `/setup-methods` for `<name>` *before* finishing this spec.

---

## Section 4 — Config schema

**Ask:** "Which knobs need defaults in the analysis's Hydra config?"

Show the user `causalab/configs/analysis/baseline.yaml` and `causalab/configs/analysis/locate.yaml` as references. Walk through these auto-emitted fields (the user does not need to specify them):

- `_name_: <name>`
- `_subdir: ...` — propose a pattern based on the methods/parameters above.
- `_output_dir: ${experiment_root}/<name>/${._subdir}`

For each user-specified knob, ask for: name, type, proposed default, one-line description.

**Reminder to the user:**
- Do NOT include `n_train`, `n_test`, `enumerate_all`, `balanced` — those live in `cfg.task.*` (invariant 12).
- Do NOT include `seed` — lives at root (`cfg.seed`).
- Hydra interpolations are allowed (e.g. `batch_size: ${batch_size}` to inherit from root).

If the analysis emits matplotlib figures, also include:

```yaml
visualization:
  figure_format: pdf
```

---

## Section 5 — Outputs

**Ask:** "Which files does the analysis write?"

For each, capture:
- File name (relative to `_output_dir`).
- Format (JSON schema, tensor shape + dtype, image format).
- What it shows / what to look for (drives the README "Interpretation" subsection).
- Downstream consumer (an analysis name, or "human reference").

Always include `metadata.json` (every analysis writes one via `save_experiment_metadata`).

If the analysis emits a heatmap PDF, capture the row/column semantics so the README can document them.

---

## Save the spec

After all 5 sections are approved:

1. Combine into a single markdown file matching `SET_UP_ANALYSIS_TEMPLATE.md`. Frontmatter `name: <name>`.
2. Write to `${SESSION_DIR}/code/analyses/<name>/set_up_analysis.md` (creating the directory if needed).
3. Tell the user: **"Spec saved at `${SESSION_DIR}/code/analyses/<name>/set_up_analysis.md`. Proceeding to scaffold."**
4. Continue with Step 2 (approval checkpoint) of the main workflow.
