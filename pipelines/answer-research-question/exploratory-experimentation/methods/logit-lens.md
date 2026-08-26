# Logit lens

Decode intermediate activations through the model's own output head to see what
token the residual stream at a given (layer, position) would predict if the
network stopped there. The cheapest way to find the depth at which the answer
becomes available, and often the first thing worth running.

## What it tells you

- **Where the answer emerges.** The layer at which the correct answer first
  appears in the top-k at the final position. This bounds where the computation
  that produces it can be.
- **What the model passes through on the way.** Intermediate predictions at the
  final position are frequently interpretable — an intermediate value, a
  superficially related token, the answer to a simpler version of the task.
- **What lives at non-final positions.** The lens at an operand token often
  decodes to something about that operand. This is where the lens earns its keep:
  it is a per-position picture, not just a depth curve.

## How to run it

Compute the lens over **all tokens and all layers** for a handful of inputs —
around a dozen — rather than a single input at the final position. The per-position
structure is most of the value, and the cost is one forward pass per input.

For each input, save the top-k tokens per (layer, position) cell, not just the
top-1. The runner-up tokens are frequently where the signal is.

## Reading it

Report which tokens have the highest logits at each position and depth, the layer
at which the answer first appears, and what is being predicted at the essential
token positions before that point.

**Cautions that matter more than the method's reputation suggests:**

- The lens applies the final layer norm and unembedding to an activation that was
  never meant to be read there. Early layers frequently decode to noise or to
  high-frequency tokens, and that is an artifact of the projection, not a finding.
- A token appearing in the top-k does not mean anything downstream reads it.
- The lens is basis-dependent in a way that varies by model family. Structure that
  appears sharply in one model and not another may be about the models' output
  heads rather than their algorithms.

Treat "the answer is decodable from layer L onward" as a bound on where to look,
which is exactly what this step needs, and not as a claim that layer L computes it.

## Variants worth knowing

- **Tuned lens** — fit an affine map per layer to correct for the basis mismatch
  above. More faithful, and no longer free: it needs a training set.
- **Lens on a specific component's output** rather than the residual stream —
  what a given attention head or MLP writes, decoded. Useful once ablation has
  named a component.

> **Execution: stub.** There is no logit-lens primitive on the intervention-protocol
> document layer. The natural expression is a write that swaps the residual stream
> at layer L into the input of the final block (or of `ln_final`), leaving the
> output head to decode it, with a `top_k` metric over the resulting `lm_head`
> read — one point per (layer, position) via a sweep. That has not been written or
> validated against the reference backend, and the tuned-lens variant would need a
> trained affine featurizer per layer. Fill in once causalab stabilizes.
