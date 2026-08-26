# Probing

Train a small classifier or regressor to predict a variable from the activation at
a site, and read the accuracy as evidence that the variable is linearly available
there.

## What it tells you

Whether a variable is *decodable* from a site, and from which layer onward. That
is a statement about the representation, not about the computation. It is useful
here precisely because it is cheap and can be swept over many sites at once.

## How to run it

- **Pick the variable and the site grid.** One probe per (layer, position) is
  normal; the grid is what makes this informative.
- **Hold out properly.** The split has to hold out whatever the probe could
  otherwise memorize — templates, entities, or whole input families depending on
  the task. A probe evaluated on inputs that share a template with its training set
  will report a number that means nothing.
- **Keep the probe small and linear** unless you have a reason not to. A
  sufficiently expressive probe will find anything, which is why the result stops
  being informative.
- **Establish a baseline.** Probe accuracy is uninterpretable without knowing what
  a probe on a control site, or on shuffled labels, achieves.

## Reading it

The standard warning applies and is not a formality: **high probe accuracy does
not mean the model uses the information.** A probe finds information that is
present and linearly available. The residual stream carries a great deal of
information that nothing downstream reads. This is exactly the correlational
result that hypothesis testing exists to convert into a causal one.

What a probe *is* good for at this step:

- **Negative results are informative.** If a variable is not linearly decodable
  anywhere, hypotheses that require it to be explicitly represented get weaker.
- **Depth profiles are informative.** The layer at which accuracy rises is a lead
  about where the variable is computed.
- **Contrasts are informative.** Two variables with different depth profiles at
  the same position are probably not the same thing, which prunes the hypothesis
  space.

A probe direction is also a candidate steering direction and a candidate subspace
to test causally — which is often the best use of it.

> **Execution: stub.** Probing has no first-class expression on the document layer.
> Two partial routes exist: harvest activations at a site with the `harvest`
> preset and fit a probe outside causalab, or express the probe as a trained
> featurizer via the `train` section — but the metric vocabulary is closed over
> `lm_head` reads, so a probe's own loss is not currently expressible as a metric.
> Fill in once causalab stabilizes.
