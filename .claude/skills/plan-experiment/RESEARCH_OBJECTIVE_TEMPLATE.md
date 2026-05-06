# Research Objective Template

Fill this in at the start of `/plan-experiment`. The completed file lands at `${SESSION_DIR}/plan/RESEARCH_OBJECTIVE.md` and is read by `/run-experiment` and `/interpret-experiment`.

This template covers **§A — research framing only**. The full task spec, analysis DAG, sweep strategy, and expected artifacts live in the sibling `plan/PLAN.md` (see `PLAN_TEMPLATE.md`).

---

## Session

**Session:** `agent_logs/{YYYY-MM-DD}--{topic}--{adj-noun}/`

---

## Objective

One sentence. State what you want to learn about the model — not what you will *do*, but what you want to *know* afterwards.

> Example: *Determine whether the day-of-week variable is encoded in a low-dimensional, geometrically-isometric subspace of Llama-3.1-8B's residual stream around layer 14.*

---

## Motivation

Two to four sentences on why this matters now. What earlier result, paper, or hunch is driving this? What changes downstream if the answer is yes vs no?

---

## Scope boundaries

Explicit list of what is **not** being investigated in this session. Boundaries protect the plan from scope creep when results turn out interesting.

- Out-of-scope variable: `…`
- Out-of-scope model: `…`
- Out-of-scope analysis: `…`

---

## Success criteria *(recommended)*

What evidence would make you say "yes, the objective is met"? Be concrete enough that you could write the conclusion paragraph today, leaving only the numbers to fill in.

> Example:
> - `locate` finds at least one (layer, position) cell with KL drop ≥ 0.5 from baseline.
> - `subspace` (PCA, k=8) reconstructs ≥ 80% of the locate cell's intervention effect.
> - `path_steering` reports an isometry score ≥ 0.7 on geodesic paths.

If left blank, mark `(not specified — `/plan-experiment` recommended adding these)`. Do not invent.

---

## Hypotheses *(recommended)*

Concrete, falsifiable predictions you'll be testing. State the prediction, then how you'd know it was wrong.

> Example:
> - **H1.** `day` is encoded around layers 12–16. *Falsified if* `locate` finds the strongest cell outside that window.
> - **H2.** The encoding is approximately linear in PCA-8. *Falsified if* PCA-8 reconstruction drops more than 30% vs the raw cell.

If left blank, mark `(not specified)`. Do not invent.
