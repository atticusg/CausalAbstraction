# Attribution

Estimate each component's contribution to a behavior from gradients, rather than
by running one intervention per component. A way to get an approximate
localization over a large grid at roughly the cost of a couple of backward passes.

## What it tells you

An approximate ranking of components by how much the metric would move if they
changed. Its purpose in this step is triage: it turns "sweep two thousand
components" into "sweep the fifty that attribution flagged."

## The forms

- **Attribution patching** — a first-order approximation of the effect of patching
  a component from a counterfactual, computed as the gradient of the metric with
  respect to the activation, dotted with the difference between the counterfactual
  and base activations. Approximates an interchange sweep over the whole grid
  from two forward passes and one backward pass.
- **Integrated gradients** — accumulate gradients along a path from a baseline to
  the actual activation. More faithful where the first-order term is a bad
  approximation, at the cost of the path samples.
- **Direct effect decomposition** — attribute the logit difference to the
  individual components writing into the residual stream, exploiting the fact that
  the residual stream is a sum. Exact for the direct path, and it says nothing
  about paths that route through later nonlinearities.

## Reading it

- **This is an approximation, and it is known to fail in specific places.**
  Attribution patching is a linearization, so it is least reliable exactly where
  the effect is largest and most nonlinear — which is to say, at the components you
  most care about. It is a triage tool, not a result.
- **Always verify the top hits with a real intervention.** The workflow is:
  attribute over the whole grid, take the top candidates, then run the actual
  ablation or interchange on those. Reporting attribution scores as if they were
  intervention results is the single most common misuse.
- **Saturation makes attribution read zero.** If the metric is at a ceiling, the
  gradient vanishes and every component looks irrelevant. Check that your metric
  has headroom — a logit difference usually does, a probability often doesn't.
- **Sign and magnitude are on different footings.** The sign of an attribution is
  usually trustworthy; the magnitude much less so.

> **Execution: stub.** There is no attribution support on the document layer. The
> `train` section is what currently triggers gradient computation, and the backend
> exposes a `grad` capability, but nothing surfaces per-activation gradients as a
> readable value, and the closed metric vocabulary has no attribution kind. This
> needs new protocol vocabulary, not just a document. Fill in once causalab
> stabilizes.
