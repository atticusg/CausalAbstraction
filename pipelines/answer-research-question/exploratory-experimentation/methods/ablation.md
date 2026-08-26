# Ablation

Remove a component's contribution and measure how much the behavior degrades. The
most direct cheap test of **necessity**: does the model need this to do the task?

## What it tells you

Which components the behavior depends on. Unlike a probe, this is already causal —
which is why a knockout sweep is often the highest-value thing in this step,
despite being the most expensive.

## The design choices that matter

**What you ablate to.** Zeroing a component is not neutral: zero is off the data
manifold, so a large drop may reflect the model being handed an impossible
activation rather than the component mattering. Run at least two references and
compare:

- **Zero** — drop the contribution entirely. Simple, and the standard baseline.
- **Mean** — replace with the corpus-mean activation at that site. On-manifold and
  usually the more trustworthy of the two.
- **Resample** — replace with the activation from a different input. Closest to an
  interchange intervention and the best-behaved, at the cost of needing a
  distribution to resample from.

When zero and mean disagree, believe mean and say that they disagreed.

**What you score.** With no task label, grade against the model's own unablated
output, using two metrics with different sensitivities:

- **Behavioral drop** — the fraction of inputs whose greedy output changes.
  Interpretable and coarse; an undisturbed output scores zero.
- **Logit drop** — the fall in the base model's predicted token's logit, averaged
  over inputs. Graded, so a component that suppresses the prediction without
  flipping the argmax still registers. It routinely reveals sensitivity that the
  behavioral metric reports as all-or-nothing.

Report both.

**What you ablate.** Two families, each swept two ways:

- **Attention heads** — the full (layer × head) grid, one drop per head, plus
  whole-sublayer bands: every head in a contiguous layer band ablated jointly, at
  widths 1, 3, 5, 10. Width 1 knocks out an entire attention sublayer.
- **MLP layer bands** — contiguous bands at widths 1, 3, 5, 10. Width 1 is the
  per-layer scan; wider bands ablate every layer in the window jointly, showing how
  much the behavior leans on progressively wider bands.

**Necessity versus sufficiency.** The band sweeps above measure necessity — does
the behavior need this band? Invert them to measure sufficiency: keep each band and
ablate every *other* layer. A low drop under the inverted sweep means that band
alone largely carries the behavior. Both readings together are much more
informative than either.

**Where.** Scope the ablation to the essential tokens by default; an
ablate-everywhere sweep answers a different and usually less interesting question.

## Reading it

- **A near-zero drop means the component is not necessary under this ablation** —
  not that it does nothing. Redundancy is common, and two components that each
  compensate for the other's removal will both read as unnecessary. If you suspect
  this, ablate them jointly.
- **Wide bands overstate.** A width-10 MLP band with a large drop tells you very
  little; almost any ten contiguous layers matter. The informative quantity is how
  the drop *grows* with width, and where it grows fastest.
- **Ablation localizes to components, not to variables.** "Head 14.3 is necessary"
  is not "head 14.3 carries the carry bit." Getting from one to the other is what
  hypothesis testing does.

## Cost

The head scan is the expensive part — one ablation per (layer, head). If it is too
slow as a single job, split the attention and MLP families rather than sharding
per layer: the MLP bands each need their full contiguous window, so a per-layer
split breaks them.

> **Execution: stub.** Ablation *is* expressible on the document layer — a write
> with `do: {"swap": <zero or mean param>}` at the component's site, an
> intervened model listing it, and a `match` or `token_logit` metric over the
> resulting `lm_head` read, swept over the component grid. What does not exist is
> the swept knockout document itself, the corpus-mean harvest that feeds the mean
> reference, or the band-ablation expansion (a band is many simultaneous writes,
> which the sweep vocabulary does not generate on its own). Fill in once causalab
> stabilizes.
