# Steering

Add a direction to an activation and see whether the behavior moves the way you
predicted. Where ablation asks whether the model *needs* something, steering asks
whether pushing on something *produces* an effect — the sufficiency side of the
same coin.

Steering appears in this protocol in two distinct roles, and conflating them is a
common source of overclaiming:

- **As a probe** (this document) — a cheap directional test used to generate a
  hypothesis. That is what belongs in exploratory experimentation.
- **As an intervention inside a causal design** — a write in an interchange
  experiment, scored against alternatives. That belongs in hypothesis testing.

## What it tells you

Whether a candidate direction has a causal effect on the behavior at all, and
which way. A direction that steers is worth building a hypothesis around; one that
doesn't, under a range of magnitudes, probably isn't.

## Where the direction comes from

- **A difference of means** between two conditions at a site. The cheapest option
  and often the strongest.
- **A probe weight vector** — see [`probing.md`](probing.md). This is the best
  reason to have run a probe.
- **A PCA component** at a site where PCA showed structure.
- **A decoder column from an SAE or other feature dictionary** — see
  [`feature-labels.md`](feature-labels.md).

## The design choices that matter

**Sweep the magnitude.** A single coefficient tells you nothing. Sweep it, and
look at the whole curve. What you want to see is a graded, monotone effect over
some range. What you often see instead is nothing until a threshold and then
incoherent output, which is the signature of having broken the model rather than
having moved a variable.

**Check that the model is still coherent.** A steering result where the output is
degenerate is not a steering result. Track output quality alongside the target
effect, and report the magnitude at which coherence fails.

**Run a control direction.** A random direction of matched norm, or the same
direction at a site where it shouldn't matter. Steering with a large enough
coefficient changes behavior in *some* way essentially always; the control is what
separates your effect from that.

**Pick the site and position deliberately.** Steering at every position is a
different experiment from steering at one, and usually a less interpretable one.

## Reading it

- **A successful steer is not a localization.** It shows the direction has an
  effect at that site; it does not show the model normally uses that direction
  there, and it does not show the direction is *the* representation of the
  variable. Interchange with a proper alternative set is what settles that.
- **A failed steer is weak evidence.** The direction may be wrong, the site may be
  wrong, the magnitude range may be wrong, or the variable may be encoded
  nonlinearly. Say which of these you ruled out.
- **Watch for the effect being downstream of what you intended.** A direction that
  changes the answer may be changing the model's read of the *question* rather than
  the value you were targeting. A control input where the target variable is
  irrelevant will usually catch this.

> **Execution: stub.** The mechanism exists on the document layer: a `params` entry
> holding the direction, a write with `do: {"add_scaled": {"op": <param>, "alpha":
> a}}` at the target site and position, an intervened model listing it, and a
> metric over the `lm_head` read — with the magnitude sweep expressed as a sweep on
> `alpha`. What is missing is the harvest-and-difference step that produces the
> direction file and the coherence-tracking metric. Fill in once causalab
> stabilizes.
