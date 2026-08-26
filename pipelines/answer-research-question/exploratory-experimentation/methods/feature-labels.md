# Feature labels — SAE, BSF, and neurons

Read out which human-legible features are active on the inputs you care about,
using a feature dictionary trained on the model. The purpose here is hypothesis
generation: a labeled feature that fires exactly on the inputs where the behavior
appears is a strong lead about what the model is representing.

## What it tells you

A vocabulary. The other probes in this step tell you *where* something is; this one
proposes *what* it is. That makes it unusually good at the specific job of turning
an observation into a nameable hypothesis.

## How to use it here

- **Contrast, don't enumerate.** The useful query is not "what features are active"
  — hundreds are — but "which features differ between inputs where the behavior
  appears and inputs where it doesn't." Build the contrast set first.
- **Check the label against the actual activating examples.** Autointerp labels are
  generated, and they are wrong often enough that a label alone should never enter
  a hypothesis. Pull the top activating examples for any feature you plan to build
  on and read them.
- **Prefer features that fire at the position you care about.** A feature active
  somewhere in the prompt is much weaker evidence than one active at the essential
  token.
- **Treat the dictionary's reconstruction error as part of the result.** A feature
  decomposition that reconstructs the activation poorly at your site is describing
  something other than what the model has there.

## Reading it

- **Feature activity is correlational**, exactly like probing. A feature that fires
  on the right inputs may be read by nothing. It is a candidate variable for a
  causal model, not a finding.
- **A feature is a direction, so it is directly testable.** This is the payoff: the
  decoder column for a labeled feature is a steering direction and a one-dimensional
  subspace you can run an intervention against. A feature that survives that test is
  a much stronger object than one that merely fires.
- **Dictionary coverage is uneven.** The absence of a feature for a concept you
  expect is weak evidence about the model and strong evidence about the dictionary.

## Relation to the rest of the protocol

A labeled feature is a natural bridge into
[`../../hypothesis-generation/hypothesis-generation.md`](../../hypothesis-generation/hypothesis-generation.md):
it proposes a variable, the variable goes into a causal model, and the feature's
decoder direction becomes the subspace that hypothesis testing adjudicates.

> **Execution: stub.** causalab has an `sae` featurizer kind that loads a fitted
> dictionary from a file and exposes encode/decode with the error term handled
> correctly, so an SAE *feature* can be read and written inside a protocol
> document. What does not exist in causalab is any of the labeling side — no
> dictionary training, no autointerp, no feature index, no BSF or neuron-label
> source. Those come from tooling outside this repo, and the handoff format
> (what file, what identity check, how a feature id maps to a dimension) is not
> specified. Fill in once causalab stabilizes.
