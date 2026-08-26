# Hypothesis report format

A fixed template for the report produced by hypothesis generation
(causal-model-level certification: competing causal models + counterfactual
datasets + CPU distinguishability). Derived from the iterated name-sorting
reports.

Treat the **macrostructure, the per-dataset distinguishability contract, and the
figure-caption convention as requirements**. Treat anything marked *bespoke* or
*conditional* as a judgment call.

---

## 1. Tone (fixed)

- Write as a **causal-model certification**, not a neural claim. Every result is
  about whether hypotheses are *deconfoundable in principle*, not about the
  network.
- Lead every section with the point, then support it.
- State scope limitations prominently: always-confounded verdicts are
  empirical at the random-N used (e.g. 100k), the input space may be small, the
  output regime is restricted, and the neural localization is a separate downstream
  experiment.
- Read distinguishability numbers as **interpretive baselines, not pass/fail**.
- No overclaiming: "separable" / "confounded everywhere" are statements about the
  datasets and the causal models, not about Llama.

---

## 2. Macrostructure (fixed)

The report runs top to bottom in exactly this order:

1. **Title**, one findings-forward sentence naming the main result, e.g.
   *"Under first-token output, explicit pairwise name comparison is not a testable
   hypothesis: the loser-order variable is confounded with the null on all 100,000
   pairs."* Not "Report" or a bare task name.

2. **The task: inputs and outputs**, open by presenting the task concretely
   *before any causal machinery*: a real example input as the model sees it, and the
   output. If the report is scoped to one output regime (e.g. first-token), show
   only that regime's output and say so here. If multiple output regimes are in
   scope, show each, and explain up front **why there is more than one** (what the
   regime contrast is meant to expose) before any matrices appear.

3. **The causal model(s)**, name and explain the model(s) in full.
   - When there is **one** model: one block, every variable named (inputs,
     intermediates, output) with what each represents and how it is computed.
   - When there are **multiple** models: present each **separately and stacked
     vertically** (never side by side, side-by-side renders badly and cramps the
     variable lists). Each model gets its own clearly-labeled block, its own
     full-width computation-graph figure, and its own complete variable list +
     mechanisms. (In the two-model name-sorting example: M-select first, then
     M-pairwise below it.)

4. **Always-confounded groups**, the hypotheses that no pair in the large random
   run deconfounds (identical output vectors). For each cluster, say which
   hypotheses collapse together and **why** (e.g. "keys determine the slot, so
   `{argmin_slot, keys}` are one hypothesis"; "winner identity = output in a
   selection task, so `{winner_name, all}`"). Include the supporting figure.

5. **Counterfactual datasets, with distinguishability inline**, the core of the
   report. Go through **each** dataset in its own sub-block (see §3). Distinguishability
   visualizations live **here, per dataset**, not collected into an aggregate
   section at the end.

6. **Observations and limitations**, cross-dataset observations that tie the datasets
   together, then the scope limitations. **No distinguishability graphs here**
   (they belong inline in §5, section 4 above). This is the last section.

---

## 3. Counterfactual-datasets section (per-dataset block)

For **each** counterfactual dataset, in its own sub-block, in this order:

1. **Role**, one line: what this counterfactual dataset is for (e.g. "the
   positional counterfactual dataset the result depends on", "the broad baseline", "the
   selection control").
2. **A concrete example pair**, the base prompt → the counterfactual prompt,
   verbatim, with the relevant variables that change/are-held-fixed called out.
3. **The distinguishability heatmap** for this dataset (the contract in §4).
4. **A short reading**, which hypotheses this dataset deconfounds, which it leaves
   confounded, and why.

---

## 4. Distinguishability heatmap contract (fixed)

One heatmap **per counterfactual dataset**, rendered inline in that dataset's
block.

- **Axes:** hypotheses × hypotheses, the **same full hypothesis set on both the
  x-axis and the y-axis**. Include **every** hypothesis, each as its own
  row/column, none collapsed:
  - the input variables individually (e.g. `name_0`, `name_1`, `name_2`),
  - the key/intermediate variables (e.g. `keys` or `key_0..n`),
  - **each comparison variable individually** when the model has them
    (`cmp_01`, `cmp_02`, `cmp_12`, …) plus any role-based discriminator
    (`cmp_losers`), do not fold the comparison variables into a single entry,
  - the selection/output variables (`argmin_slot`, `winner_name`),
  - the two reference hypotheses `null` and `all`.
- **Cell value:** the `can_distinguish_with_dataset` score for that dataset between
  the row hypothesis and the column hypothesis. Show the numeric value in each cell.
- **Colorscale:** sequential cream→ember (`#F6F5F0 → #C4650D`), domain 0..1
  (scores are unsigned rates). Never rainbow/viridis.
- **All ticks must be visible.** Plotly auto-thins tick labels and will drop ~half
  of them; force every tick on **both** axes: `tickmode='array'` with `tickvals` =
  every category index and `ticktext` = every hypothesis name, `automargin=True`,
  and angle the x-tick labels (e.g. `tickangle=-45`) so they fit without overlap or
  dropout.
- **One matrix per dataset when scoped to a single output regime.** If the report
  covers multiple output regimes, show one matrix per regime per dataset,
  and the §2.2 up-front explanation of why the regimes differ must precede them.

---

## 5. Figure-caption convention (fixed)

Every figure (computation graphs, always-confounded figure, each distinguishability
heatmap) carries:

- An `<h3 class="figure-title">` titled **"Figure N: <title>"**.
- A **Description** (plain prose): what the figure tests + how to read the axes +
  any method detail.
- A **Findings** line (prefixed "Findings."): what this figure actually shows in
  this run. Terse and concrete.

Keep palette, styling, and numerical conventions consistent across the report's
figures. Cards are for scalar metrics only; figures and prose sit flat.

---

## 6. Quick checklist

- [ ] Title is one findings-forward sentence.
- [ ] Section order: task I/O → causal model(s) → always-confounded groups →
      counterfactual datasets (inline distinguishability) → observations & limitations.
- [ ] Task I/O shown concretely before any causal machinery; output regime(s) named
      (and, if >1, why) up front.
- [ ] Causal models presented in full; multiple models stacked vertically, each
      with its own visual + complete variable list + mechanisms (never side by side).
- [ ] Always-confounded clusters explained with the reason each collapses.
- [ ] Each counterfactual dataset: role + concrete example pair + inline heatmap +
      reading.
- [ ] Distinguishability heatmaps are per-dataset and inline (not an aggregate
      section at the end).
- [ ] Heatmap axes include EVERY hypothesis individually (inputs, keys, each
      comparison variable, argmin_slot/winner_name, null, all) on both axes.
- [ ] Cell = can_distinguish_with_dataset score, values shown, sequential cream→ember 0..1.
- [ ] ALL tick labels visible on both axes (tickmode array + automargin + angled x).
- [ ] One matrix per dataset for a single-regime report; per-regime matrices only
      if multiple regimes are in scope, with the up-front rationale.
- [ ] Every figure has Figure N title + Description + Findings.
- [ ] Observations & limitations last; no distinguishability graphs in that section.
- [ ] State that the verdict depends on the sample size used here, the output
      regime is restricted, and neural localization is a separate downstream
      experiment.
