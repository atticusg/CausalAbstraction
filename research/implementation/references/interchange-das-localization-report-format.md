# Interchange & DAS Localization Report Format

A fixed template for reporting neural localization results, full-vector interchange
(activation patching) and DAS (distributed alignment search), on a causal model whose
hypotheses were certified upstream (see the hypothesis report format at
`../../answer-research-question/hypothesis-generation/hypothesis-report-format.md`). Derived from the iterated name-sorting
localization reports.

This format applies to **any** interchange/DAS localization report, **regardless of
whether causalab ran the interventions**, causalab is one way to produce these results,
not a requirement of the format.

Treat the **positive-control discipline, the variable-by-variable macrostructure, the
per-variable contents, and the plot-type rules as requirements**. Treat anything marked
*bespoke* or *conditional* as a judgment call.

---

## 1. Tone (fixed)

- Report localization as **causal evidence with controls**, not as a label on a cell. A
  high score is a hypothesis until the positive control and the alternatives are in the
  picture.
- Lead each variable's section with the **verdict for that variable**, then support it.
- **Nulls are first-class when method-validated:** "this variable is not a
  transplantable / low-rank variable here" is a real finding, but only if a known-present
  variable localizes with the same pipeline (see §2).
- State what the methods can and cannot rule out. Interchange + DAS can rule out a
  **transplantable / low-rank-linear** representation; they do **not** exclude a
  distributed or non-linear computation.
- No overclaiming. A localized cell is "where the variable is read/transplantable", not
  "where the model computes it", unless an intervention shows the downstream behavior
  changes.

---

## 2. The central principle: a localization is only interpretable with a working positive control

Every localization claim, **especially every null**, must be read against a **positive
control**: a variable known to be present (typically the output / identity content)
localized with the **same pipeline on the same data**.

- If the positive control localizes (e.g. identity transplants at ~0.97–1.0) and the
  target sits at the control floor, the target null is an **interpretable finding**, not
  an instrument failure.
- If the positive control **fails** (e.g. DAS cannot recover even identity on held-out),
  the method is **not validated here**: report the target result as **inconclusive for
  that method** and fall back to the method that did validate. (In the name-sorting
  example where DAS failed its identity control, the comparison verdict rested on
  interchange, which validated.)
- Always report the **control floor** (random-subspace for DAS, the null/baseline for
  interchange) next to the target score, so "above chance" is **visible, not asserted**.

---

## 3. Macrostructure: variable by variable (fixed)

Organize the report **one variable (or variable group) at a time**, in this order:

1. **Input variables** (e.g. the raw inputs),
2. **Output variable** (the thing the model emits; usually the working positive control),
3. **Intermediate variables** (the hypothesized computation: selection indices, keys,
   comparisons).

Each variable gets its own section that centers on localizing that variable and covers
**both interchange and DAS** for it. A single localization report (or experiment) should
be built around a specific variable or set of variables; do **not** interleave variables
within one figure except in the multi-hypothesis line graph (§5).

Keep the **instrument-validation result (the positive control) up front**, before the
intermediate-variable sections, so the reader knows whether the verdicts that follow are
supported by the checks.

---

## 4. Per-variable section contents (fixed)

For each variable's section, present all three:

1. **Counterfactual dataset explanation.** What the dataset varies vs holds fixed to
   isolate this variable, a concrete **ORIGINAL** input and its **COUNTERFACTUAL** input
   shown verbatim, and the **EXPECTED** output after an interchange intervention on the
   variable (what the patched output should become if the variable is localized at that
   cell). For a comparison/loser variable, state the expected effect at the **specific
   output position** where it would surface.
2. **Alternative-hypothesis distinguishability.** The variable's distinguishability
   against its alternatives, carried from the causal certification, at minimum vs the
   **null** and vs the **full-mediator (`all`)** hypothesis, plus the **nearby**
   hypotheses that could be confused with it. This tells the reader which alternative a
   positive neural score would be confounded with.
3. **Plots** (see §5).

---

## 5. Plot-type rules (fixed)

- **Heatmap**, for a **single hypothesis/variable** across tokens and layers. Axes:
  layer (y) × token position (x). One heatmap per variable per method. Use the sequential
  **cream→ember** colorscale for scores in [0,1]; show **all tick labels** on both axes
  (force `tickmode='array'`, `automargin`, angled x-labels, Plotly thins ticks by
  default).
- **Line graph**, to focus on **specific token position(s)** with **multiple hypotheses
  overlaid** (the target variable vs its alternatives vs the control vs null), typically
  across layers at the key position. This is where the variable-vs-alternative comparison
  and the control floor are read at a glance.
- **Qualitative exemplars** (encouraged): for a localized cell, show a few concrete
  patched examples (the actual patched prediction, **including cases matching neither base
  nor counterfactual**), not just the aggregate score.

Every figure carries the **caption convention**: a **Description** (what it tests / how to
read the axes) and a **Findings** line (what it shows in this run). Follow the worker
`figures` / `publish-results` contracts for palette and numerical conventions.

---

## 6. Method-specific requirements

### Interchange (full-vector patching)

- Report the **pairwise score** (rate the patched output matches the
  counterfactual-expected label) per (variable, layer, position), on **held-out** data.
- Identify the **best cell(s)**; distinguish a genuine variable-specific signal from **residual
  content being moved** (e.g. an elevated early cell on an input span is often the input's
  own embedding content, not a computed intermediate, compare to that position's
  embedding baseline).

### DAS

- Report **held-out** interchange accuracy, **never train** accuracy (DAS overfits; a
  train-only number is not a localization). Report the train number too only to show
  whether it could even fit.
- Include a **matched-k random-subspace control** at every cell, and a **k sweep** (small
  k first, since most task variables are low-dimensional).
- **Instrument validation is mandatory** (§2): the identity/positive-control DAS must beat
  its random control on held-out before any target null is trusted.
- **Dissociate known confounds in the data.** If two variables are bijective in the
  dataset (e.g. one-name-per-letter makes rank ≡ identity), DAS cannot separate them;
  expand the data so they dissociate and use a **held-out generalization test** (e.g.
  same-rank, unseen identity) as the decisive number.

---

## 7. Reading the numbers

- Distinguishability and localization scores are **interpretive baselines, not pass/fail
  gates**. Carry the causal baseline into the neural reading: if a dataset only separates
  the target from a nearby alternative at rate *r*, a neural score near *r* is confounded
  with that alternative.
- Distinguish a **method-validated null** (positive control works, target at floor ⇒ real
  refutation of a transplantable/low-rank representation) from an **inconclusive** one
  (positive control failed ⇒ method can't see anything here).
- Report **under-powered hints** as
  follow-up targets, not as findings.

---

## 8. Quick checklist

- [ ] A positive control (known-present variable) is localized with the same pipeline, and
      its result is shown **before** the target verdicts.
- [ ] Control floors shown next to every target score (random-subspace for DAS,
      null/baseline for interchange).
- [ ] Sections organized variable by variable: inputs → output → intermediates.
- [ ] Each variable's section: counterfactual explanation + original/CF example + expected
      post-interchange output; distinguishability vs null + full-mediator + nearby; plots.
- [ ] Heatmaps = single variable across layers × tokens; line graphs = specific
      position(s) with multiple hypotheses overlaid.
- [ ] All tick labels visible on heatmap axes.
- [ ] DAS reports held-out (not train) accuracy, with matched-k random control and k
      sweep; instrument validated via the positive control.
- [ ] Known data confounds dissociated (expanded data + held-out generalization test)
      where a variable is bijective with another.
- [ ] Nulls labeled method-validated vs inconclusive; under-powered hints flagged as
      follow-ups, not findings.
- [ ] State the scope in plain terms: the result can rule out a transplantable or
      low-rank linear representation, but it does not rule out a distributed or
      nonlinear one.
- [ ] Every figure has a Description + Findings caption.
