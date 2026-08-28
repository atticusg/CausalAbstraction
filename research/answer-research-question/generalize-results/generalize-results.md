# Step 5 — Generalize results

A tested hypothesis initially applies only to the model, task, prompt, data split,
and seeds used in step 4. Test each boundary before making a broader claim.

## Expand one axis at a time

Each is a separate experiment that can fail independently. Roughly increasing cost:

- **Inputs within the task** — edge values, longer operands, distribution tails,
  cases where the answer collides with an operand. Step 1's error catalogue tells
  you which regions are most likely to break the story.
- **Prompt format** — different template, phrasing, separator, zero- vs few-shot.
  Format sensitivity is common and is not a minor caveat: a mechanism that exists
  only under one template is a mechanism for that template.
- **Position** — if the mechanism was localized at a fixed token index, does it
  move when the content moves? Interchange at a fixed index cannot distinguish a
  mechanism pinned to a position from one pinned to a semantic role.
- **Datasets** — ideally one you did not construct. The strongest test here, since
  your own dataset shares its assumptions with your causal model.
- **Related tasks** should have the same abstract structure but present it in a
  different form. To test whether conditions share a mechanism, train the method
  that locates the mechanism on one condition and evaluate it on a condition held
  out from training. The distinguishability matrix in step 3 cannot test whether a
  mechanism is shared across conditions.
- **Models** — another checkpoint, size, or family. Most informative about whether
  you found something about this network or about how transformers do this task.
  Layer indices, head counts, and tokenization all change, so redefine "the same
  location" before comparing.

Report the **fraction of behavior** explained. A result of 60% supports a different
claim from 100%.

Repeat step 4's controls in every new setting. If full mediation fails, the test is
broken; the mechanism has not necessarily failed to generalize.

## Reading it

- **A failed generalization is a result.** The boundary is often more informative
  than the interior: holding at two digits and breaking at three says something
  specific about what the model built.
- **Say where you stopped and why.** "Not tested on other model families" is an
  honest, useful sentence; silence reads as generality.
- **Don't widen the claim to match the strongest result.** If it held on four of
  six formats, the claim is about those four and the two failures go in the writeup.
- **Keep the limits of the claim visible in the prose.** A statement can become an
  overclaim when it is repeated without the conditions that made it true.

## Typical units of work

- One per axis attempted.
- One per new dataset or model brought in, including establishing that the setting
  is comparable at all.
- One for the final statement of scope: the widest claim the evidence supports,
  with its boundaries.

## Handoff

You leave with a claim and an explicit boundary around it. Update `ROADMAP.md`,
then go to [`../save-results/save-results.md`](../save-results/save-results.md).

If an expansion turns up a mechanism you did not expect, that is a new question and
gets its own roadmap rather than being appended to this one.
