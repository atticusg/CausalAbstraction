# Hypothesis report format

Use this template for the HTML report produced by one intermediate-variable
experiment during hypothesis generation. The report describes the target
variable, its counterfactual datasets, and the input, output, and intermediate
variables that a CPU computation can distinguish from it.

Follow the required section order, the rules for reporting distinguishability for
each dataset, and the convention for figure captions. Use your judgment for
anything marked *bespoke* or *conditional*.

---

## 1. Tone (fixed)

- Describe what the causal model and datasets can test. Do not make a claim about
  what the network represents. Every result in this report concerns whether the
  datasets can distinguish the hypotheses in principle.
- Lead every section with the point, then support it.
- State the limits prominently. A conclusion that no pair distinguishes two
  hypotheses is based only on the random sample used, such as 100,000 pairs. The
  input space may be small, the output may be restricted to one form, and a later
  experiment must still test where the variables are represented in the network.
- Use distinguishability numbers as baselines for interpretation, not as pass or
  fail thresholds.
- Do not overstate the result. "Separable" and "confounded everywhere" describe
  the datasets and causal models, not Llama.

---

## 2. Required section order

The report runs top to bottom in exactly this order:

1. **Title.** Write one sentence that states the main result, for example:
   *"Under first-token output, explicit pairwise name comparison is not a testable
   hypothesis: the loser-order variable is confounded with the null on all 100,000
   pairs."* Not "Report" or a bare task name.

2. **The task: inputs and outputs.** Present the task concretely before explaining
   the causal model. Show a real example input as the model sees it, and the
   output. If the report covers one form of output, such as only the first token,
   show only that form and state the restriction here. If the report covers
   several forms of output, show each one and explain what comparing them is meant
   to reveal before showing any matrices.

3. **The causal models.** Name and explain every model in full.
   - When there is **one** model, use one block. Name every input, intermediate
     variable, and output, and explain what each represents and how it is computed.
   - When there are **multiple** models: present each **separately and stacked
     vertically**. Do not place them side by side because that leaves too little
     room for the variable lists. Give each model a clearly labeled block, a
     computation graph that uses the full width, and a complete list of variables
     and mechanisms. In the name sorting example with two models, show M-select
     first and M-pairwise below it.

4. **Target and alternatives.** Define the one intermediate variable targeted by
   this experiment. List every input, output, and intermediate variable that
   could plausibly be confused with it. Then list the variables considered but
   excluded and explain briefly why each one is not a plausible alternative.

5. **Groups that the sampled pairs never distinguish.** List the hypotheses that
   produce identical output vectors for every pair in the large random sample. For
   each group, state which hypotheses cannot be distinguished and **why**. For
   example, "keys determine the slot, so
   `{argmin_slot, keys}` are one hypothesis"; "winner identity = output in a
   selection task, so `{winner_name, all}`"). Include the supporting figure.

6. **Counterfactual datasets and distinguishability.** Give **each** dataset its own
   subsection as described in section 3. Put the visualization for a dataset in
   that dataset's subsection. Do not collect all visualizations in a separate
   section at the end.

7. **Valid hypothesis tests.** State exactly which target-versus-alternative
   comparisons the datasets support in hypothesis testing and which remain
   confounded.

8. **Observations and limitations.** Explain patterns across the datasets, then
   state the limits of the result. Do not put distinguishability graphs in this
   section. This is the last section.

---

## 3. Section for each counterfactual dataset

Give **each** counterfactual dataset its own subsection with the following parts in
this order:

1. **Purpose.** State in one line what this counterfactual dataset is for (e.g. "the
   positional counterfactual dataset the result depends on", "the broad baseline", "the
   selection control").
2. **A concrete example pair**, the base prompt → the counterfactual prompt,
   verbatim. State which relevant variables change and which remain fixed.
3. **The distinguishability heatmap.** Follow the requirements in section 4.
4. **A short interpretation.** State which hypotheses this dataset distinguishes,
   which it cannot distinguish, and why.

---

## 4. Required format for distinguishability heatmaps

Show one heatmap for every counterfactual dataset in the subsection for that
dataset.

- **Axes:** Use the same focused set on both axes: the target intermediate
  variable, every plausibly confusable input, output, and intermediate variable,
  plus the `null` and `all` references. Give every included variable its own row
  and column. Do not combine distinct input tokens, spans, or intermediate
  variables into one label. Do not add unrelated variables merely to make the
  matrix exhaustive.
- **Cell value:** the `can_distinguish_with_dataset` score for that dataset between
  the row hypothesis and the column hypothesis. Show the numeric value in each cell.
- **Colors:** Use a sequential scale from cream to ember (`#F6F5F0 → #C4650D`)
  over the range 0 to 1 because the scores are unsigned rates. Do not use a rainbow
  or Viridis scale.
- **Show every tick label.** Plotly automatically removes some tick labels. Force
  every tick to appear on **both** axes: use `tickmode='array'` with `tickvals` =
  every category index and `ticktext` = every hypothesis name, `automargin=True`,
  and angle the x-tick labels (e.g. `tickangle=-45`) so they fit without overlap or
  dropout.
- **Show one matrix per dataset when the report covers one form of output.** If the
  report covers several forms of output, show one matrix for every form and dataset.
  Explain why the report compares these forms in the second part of section 2
  before showing the matrices.

---

## 5. Required figure captions

Every figure (computation graphs, always-confounded figure, each distinguishability
heatmap) carries:

- An `<h3 class="figure-title">` titled **"Figure N: <title>"**.
- A **Description** in plain prose that explains what the figure tests, how to read
  its axes, and any necessary details about the method.
- A **Findings** line that begins with "Findings." and states what this figure
  shows in this run. Keep it brief and concrete.

Use the same colors, styling, and numerical conventions for every figure in the
report. Use cards only for individual numerical metrics. Present figures and prose
without cards.

---

## 6. Final check

Confirm that the report follows the section order above and that:

- it shows the task before the causal models and states which outputs it covers;
- it defines one target intermediate variable, its plausible alternatives, and
  why other variables were excluded;
- it explains every model, included variable, and group that the sample cannot
  distinguish;
- each dataset has an example, heatmap, and interpretation in one subsection;
- every heatmap follows the axis, label, value, and color rules in section 4;
- every figure has a numbered title, Description, and Findings line; and
- it states which later hypothesis tests are valid; and
- the conclusion states its limits and makes no claim about where variables are
  represented in the network.
