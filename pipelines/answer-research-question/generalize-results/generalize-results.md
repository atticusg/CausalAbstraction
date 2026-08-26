# Step 5 — Generalize results

A tested hypothesis is a claim about one model, one task, one prompt format, one
split, and one set of seeds. This step asks how far past that the claim actually
reaches — and the answer is found by testing, not by asserting.

The reason this is its own step rather than a paragraph in the writeup is that the
scope of a claim is the part most often overstated, and it is cheap to overstate
because nothing in the previous step pushes back. Expanding the scope
deliberately, and reporting where the expansion stopped, is what makes the result
usable by someone else.

## Expand along one axis at a time

Each axis is a separate experiment, and each can fail independently. Run them in
roughly increasing order of cost.

**Inputs within the task.** The easiest expansion and the one most likely to
succeed. Does the mechanism hold on inputs that the original design never sampled —
edge values, longer operands, the tails of the distribution, cases where the answer
collides with an operand? Note that this is where behavioral analysis pays off: the
error structure you catalogued in Step 1 tells you which input regions are most
likely to break the story.

**Prompt format.** Different template, different phrasing, different separator,
zero-shot versus few-shot. Format sensitivity is extremely common and is not a
minor caveat — a mechanism that only exists under one template is a mechanism for
that template. If you used few-shot in Step 1, this is where you find out what that
cost you.

**Position.** If the mechanism was localized at a fixed token index, does it move
with the content when the content moves? A mechanism pinned to an absolute position
and a mechanism pinned to a semantic role are different findings, and interchange
at a fixed index cannot tell them apart.

**Datasets.** A different dataset for the same behavior, ideally one you did not
construct. This is the strongest test in this step, because your own dataset shares
its assumptions with your causal model.

**Related tasks.** Does the mechanism appear in tasks that share the abstract
structure but not the surface? If your hypothesis is about a carry bit, does it
hold for a different arithmetic framing? A hypothesis about a *shared* mechanism
across conditions is settled here and nowhere else: train the localizer on one
condition, evaluate on a held-out one. Step 3d's distinguishability matrix is
structurally blind to this, so it has to be a generalization experiment.

**Models.** Another checkpoint, another size, another family. The most expensive
and the most informative about whether you have found something about *this network*
or something about how transformers do this task. Be careful: layer indices, head
counts, and tokenization all change, so "the same location" needs redefining before
the comparison means anything.

**Scale of the claim.** A mechanism that accounts for 60% of the behavior and a
mechanism that accounts for all of it are different claims. Report the fraction.

## Run the same controls

Every expansion is a new experiment and needs the controls from Step 4 — a positive
control on the new setting especially. A mechanism that "fails to generalize" to a
setting where full mediation also fails to hit ceiling has not failed to
generalize; the pipeline is broken there.

## Reading it

- **A failed generalization is a result, not a setback.** The boundary is often
  more informative than the interior: knowing the mechanism holds for two-digit and
  breaks at three-digit tells you something specific about what the model built.
- **Say where you stopped and why.** "Not tested on other model families" is an
  honest, useful sentence. Silence reads as generality.
- **Do not quietly widen the claim to match the strongest result.** If it held on
  four of six formats, the claim is about those four, and the two failures go in
  the writeup.
- **Watch for the scope creeping in the prose.** The most common form of
  overclaiming is not a false sentence but a true one stated without its
  qualifiers, three paragraphs after the qualifiers were last mentioned.

## Typical units of work

- One unit per axis of expansion attempted.
- One unit per new dataset or model brought in, including the work of establishing
  that the setting is comparable at all.
- One unit for the final statement of scope: the widest claim the evidence
  supports, with its boundaries.

## Handoff

You leave this step with a claim and an explicit boundary around it. Update
`ROADMAP.md`, then go to [`../save-results/save-results.md`](../save-results/save-results.md).

If a generalization attempt turns up a mechanism you did not expect — which
happens, particularly when moving to a new input regime — that is a new question,
and it gets its own roadmap rather than being appended to this one.
