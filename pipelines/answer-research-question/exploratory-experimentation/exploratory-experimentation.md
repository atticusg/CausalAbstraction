# Step 2 — Exploratory experimentation

Cheap experiments whose purpose is to produce signal a hypothesis can be built
from. Not to establish anything. The output of this step is a set of observations
and a much smaller space of plausible mechanisms; the output is *not* a claim.

The discipline that makes this step work is keeping it cheap and keeping it
honest. Cheap, because you are going to run several of these and most of them will
show nothing. Honest, because every method here produces a picture that is easy to
over-read — a logit-lens heatmap always has structure in it, and a probe can
always be trained to some accuracy above chance. Treat every result here as
*correlational and suggestive*, and write it down that way.

## Before you start

Two things from the roadmap:

- **What would each probe show if your current guess is right?** Write the
  prediction down before running. A probe with no prediction attached will confirm
  whatever you already believed.
- **Which sites and positions are even plausible?** With causal attention,
  information flows left to right, so a fact introduced at token 12 cannot be read
  at token 5. That one constraint usually removes most of the grid before you run
  anything.

## The probes

Each has its own document. They are independent — run the ones that fit the
question, in whatever order, and skip the rest.

| Probe | What it gives you | Document |
|---|---|---|
| **Logit lens** | Where in depth the answer becomes decodable, and what the model is "thinking" at intermediate layers | [`methods/logit-lens.md`](methods/logit-lens.md) |
| **Probing** | Whether a variable is linearly decodable from a site, and from which layer | [`methods/probing.md`](methods/probing.md) |
| **Ablation** | Which components the behavior *needs* | [`methods/ablation.md`](methods/ablation.md) |
| **Steering** | Whether pushing a direction changes behavior the way you predict | [`methods/steering.md`](methods/steering.md) |
| **Attribution** | A cheap approximation of which components matter, without a sweep per component | [`methods/attribution.md`](methods/attribution.md) |
| **Feature labels** | What human-legible features are active — SAE features, BSF, neuron labels | [`methods/feature-labels.md`](methods/feature-labels.md) |

Two more that belong here even though they are not in the list above, because they
are the cheapest useful things you can run:

- **Single-pair interchange.** Take one base input and one counterfactual input
  that differ in the variable you care about, and swap the activation at each
  (layer, position) cell in turn, recording where the output flips. It is a full
  localization sweep at a sample size of one, which is exactly the right price for
  this step. Cells where the output flips are where that information is being
  read. See [`../hypothesis-testing/hypothesis-testing.md`](../hypothesis-testing/hypothesis-testing.md)
  for the powered version.
- **PCA of the residual stream.** Collect the residual stream at a token of
  interest across a few thousand inputs, fit centered PCA per layer, and color the
  projection by every label the task supports — the token's own value, the correct
  answer, the model's answer, correct-vs-incorrect. What the top components
  organize by, and the depth at which that organization appears, is often the
  single most informative thing in this step.

## Reading results at this altitude

The failure mode of this step is promoting a suggestive picture to a finding.
Three guards:

- **A structure you can see is not a structure the model uses.** PCA separating by
  a label means the information is present and linearly available, not that
  anything downstream reads it. Only an intervention settles that.
- **A negative is weaker than it looks.** A probe that fails may mean the
  information is absent, or nonlinearly encoded, or present at a site you didn't
  check. Say which you can rule out.
- **Sample sizes here are small on purpose.** A single-pair interchange sweep on
  three pairs is a lead, not a rate. Do not report it with a number that implies
  otherwise.

## Typical units of work

- One unit per (probe × site family). "Logit lens over all layers and positions"
  is one unit; "ablate every attention head" is another.
- One unit per essential token, when a probe is run per token.
- One unit for assembling the observations into the input for hypothesis
  generation — this is real work, not bookkeeping.

## Essential tokens

Several probes are scoped to the tokens that matter, so identify them once here
and reuse the list.

An essential token is one whose replacement changes the output. Aim to be
**exhaustive** rather than obvious — anything you can substitute to flip the answer
counts, not just the operands. Confirm each one empirically: edit that token
alone, leave everything else fixed, and check that the output actually moves.
Intuition is wrong about this often enough to be worth the compute.

Record each token's surface text and its position in the prompt. When the position
varies across inputs (variable-width operands, for instance), record a per-input
position rather than a fixed index.

## Handoff

You leave this step with a set of observations, each tagged with how much weight
it can bear. Update `ROADMAP.md`, then go to
[`../hypothesis-generation/hypothesis-generation.md`](../hypothesis-generation/hypothesis-generation.md).

If you arrived here *from* hypothesis testing, you came because the existing
evidence didn't support another hypothesis worth testing. Do not re-run the probes
you already ran. Pick sites, positions, or input slices you have not looked at,
and let the failed test tell you where to look — a hypothesis that was refuted at
one location has usually mislocated the mechanism rather than misdescribed it.
