# Causal Analysis of Internal Representations — Handbook

This handbook explains causal analysis of neural networks: how to plan the work,
design causal models and counterfactual datasets, choose interventions, and state
the resulting conclusions. It is method-agnostic. The documents under
[`answer-research-question/`](answer-research-question/answer-research-question.md)
explain how to execute this work with CausaLab.

## Summary

Causal analysis starts with a claim about an internal mechanism in a neural network, gathers evidence on whether the hypothesized mechanism causally mediates the network's outputs, and writes a detailed quantitative report.

**Correlation vs. causation.** Probing, PCA, SVD, clustering, representation similarity, saliency, and SAE feature visualizations are useful for discovering candidate variables or mediators, but they are correlational unless verified with interventions. Correlation can tell you where to look; it does not entail causation. Causal interventions tell you whether a representation actually matters for a model behavior.

The end-to-end roadmap starts by finding a task the model can solve above random
chance and analyzing its errors. Exploratory experiments then produce competing
guesses about intermediate variables and where those variables may be carried.
Hypothesis generation turns those guesses into explicit high-level causal models.
Hypothesis testing builds counterfactual datasets for individual intermediate
variables and tests whether the corresponding neural representations mediate the
model's output. The ultimate goal is strong evidence for meaningful intermediate
causal variables, together with clear limits on that evidence.

## What causal abstraction is

Causal abstraction is the framework behind interchange-based analysis. A **high-level causal model** is a process-level hypothesis about what a network computes on a task: a directed graph of input, intermediate, and output variables, each computed from its parents by a deterministic mechanism. Formally, it consists of *variables* (input, intermediate, and output), each taking on a range of possible values, and each a function of the variables that appear as parents in the graph.

A **concrete, testable hypothesis** is narrower — three parts:

1. a causal model (its variables and mechanisms),
2. a subset of that model's variables, and
3. a claimed neural location (e.g. the residual stream at layer 12, above the last token).

The claim is that the chosen subset of causal variables is a *faithful abstraction* of that location in the network: fixing the network's representation there to the value it would take under a counterfactual input moves the output the same way that fixing those high-level variables moves the causal model's output. Interchange interventions (below) are how the claim is tested.

Building the causal model and the counterfactual datasets that distinguish
competing hypotheses is the
[`hypothesis-generation`](answer-research-question/hypothesis-generation/hypothesis-generation.md)
step. That step is CPU-only and never touches the target network; ablation and
steering need nothing of the sort. The design craft it draws on is laid out below,
under "Designing causal-model hypotheses and counterfactual datasets".

## The causal-analysis pipeline

**Input.** The user starts with a claim about the behavior of a target network — for example a natural-language description of a behavior, how manipulating activations in a specific subspace changes behavior, or a pointer to a subspace with no intuition for what information it mediates.

**Pipeline.**

1. Write the causal claim. State it as something like "Intervening on representation site S, or component A within site S, should change behavior C in direction D, while leaving off-target behavior E mostly unchanged." Omit E when off-target behavior is not relevant; omit A when intervening on an entire representation rather than a subcomponent. Adapt the template to your setting and intervention method. If the provided claim lacks the detail to formulate a concrete causal claim, do an initial exploration of the target behavior first.
2. Build the datasets. For steering or ablation, datasets of single examples suffice. For interchange interventions, use datasets of paired (base, counterfactual) examples.
3. List the sites to test. Identify the representations to intervene on — particular layers, tokens, attention heads, MLPs. If you are not intervening on the full representation, also specify which part — a direction defined by an SAE feature, a subspace constructed through PCA.
4. Define the behavioral metric before looking at the intervention result — accuracy, logit difference, target-class probability, an LM-as-a-judge evaluation, etc.
5. Run the interventions and any controls on the neural model.
6. Write a quantitative HTML report.

**Output.** The final report presents:

- inputs and datasets,
- the intervention method,
- intervention effects on model behavior,
- no-op controls where no behavioral change is expected,
- embedded figures.

The CausaLab research documents define the required organization, tone, and
figure conventions for each report. Use
`answer-research-question/hypothesis-generation/hypothesis-report-format.md`
for a report that compares causal models and counterfactual datasets on a CPU. Use
`implementation/references/interchange-das-localization-report-format.md`
for a report that locates variables in the network with interchange, activation
patching, or DAS.

**Running the whole pipeline.**
`answer-research-question/answer-research-question.md` defines the
autonomous, end-to-end pipeline. It starts with a task or behavior and runs through
behavioral analysis, exploratory experimentation, hypothesis generation,
hypothesis testing, generalizing the result, and saving it. The pipeline maintains
its roadmap and routes a negative test back to exploration or hypothesis generation
when needed.

The older entry path at
`subspace-causal-analysis-pipeline/subspace-causal-analysis-pipeline.md`
starts with a supplied subspace bundle. The bundle may be a rotation in a
`.safetensors` file, a manifest, or a `{manifold_bundle, community_id}` pointer.
This older pipeline uses interchange IIA or mediation on the fixed rotation to
decide whether the subspace is causal. It predates the causalab protocol refactor
and is marked as stale.

### Workflows that run this pipeline

The steps above apply regardless of the specific method. Every path below is
relative to this `research/` directory:

- **Establishing the behavior** → `answer-research-question/behavioral-analysis/behavioral-analysis.md`.
- **Initial exploration** → `answer-research-question/exploratory-experimentation/exploratory-experimentation.md` (from a behavior) · `explore-subspace/explore-subspace.md` (from a given subspace).
- **Causal model + counterfactual design** (CPU-only, never touches the network) → `answer-research-question/hypothesis-generation/hypothesis-generation.md`.
- **Steering execution** → use the steering system provided by the host research
  environment; CausaLab does not currently ship this end-to-end workflow.
- **Ablation / interchange / DAS / DBM on the target network** → `answer-research-question/hypothesis-testing/hypothesis-testing.md`, with the codebase how-to in `implementation/implementation.md`.
- **Scoping and recording the result** → `answer-research-question/generalize-results/generalize-results.md` · `answer-research-question/save-results/save-results.md`.
- **The whole pipeline, end-to-end** → `answer-research-question/answer-research-question.md` (the flow chart, the roadmap, and the routing) · `subspace-causal-analysis-pipeline/subspace-causal-analysis-pipeline.md` (the older path for adjudicating a given subspace).

## From an initial claim to a hypothesis

The researcher starts causal analysis with a claim about an internal mechanism driving a behavior. Example claims:

- A semantic description of a behavior ("The model represents the user's emotions in a coherent representation.")
- An internal subspace or site ("This subspace contains a cluster of activations on webtext inputs; let's find out which concept it represents.")
- A semantic description of a subspace ("This subspace causally mediates days of the week.")
- A high-level causal model of how the network solves a task internally ("Run causal analysis on the pre-implemented entity-binding task.")

A good claim for initiating a causal analysis describes a concrete mechanism and provides input-output pairs of the target model that clearly demonstrate the target behavior — for example the "Intervening on representation site S, ..." template above, alongside example input-output pairs for behavior C.

If the claim might lack specificity, conduct an initial exploration. Look for early signs of life of the target behavior. The scale is intentionally small: fail fast and iterate quickly to identify input prompts that induce the behavior and model sites that mediate it. Choose the exploration approach by the type and specificity of the claim and the evidence already available:

- If the claim broadly names a model behavior, complete behavioral analysis and
  then run the required logit lens, PCA, and full-vector patching experiments.
  Follow patching signal with residual stream DAS, attention head DBM, and MLP
  neuron DBM as specified by the exploratory-experimentation step.
- If the claim points to a specific model-internal subspace, characterize that subspace directly (the explore-subspace workflow).
- If the claim already asserts a behavioral mechanism or a semantic role for a subspace, verify it independently and quickly (<5 min) with one of the above before committing to a full analysis.

After gathering initial evidence, prepare for causal interventions by collecting a comprehensive dataset.

## Designing causal-model hypotheses and counterfactual datasets

The work *before* the neural experiment is building the causal model and the candidate variable subsets, and designing and certifying the counterfactual datasets that tell competing subsets apart — all CPU-only, never touching the target network. (The grounding — what a causal model is, what a concrete, testable hypothesis is, and what a faithful abstraction is — is in "What causal abstraction is" above.) This is the design craft that interchange / causal-abstraction analysis rests on; the hypothesis-generation step runs it on causalab.

### One value per variable

This is the rule that most shapes how useful your model is.

**One variable per conceptual unit, each holding a single value — not a list-, dict-, or tuple-valued variable bundling several.** (The value may be discrete, like a weekday or a boolean, or numeric; what matters is that it is one thing, not a collection.) A hypothesis can only target a variable as a whole, so a variable whose value is "the list of matching positions" can only ever express *"the network has a representation of that entire list"* — never *"the network represents position 0 here and position 1 there."* Splitting the bundle into separate variables is what lets you localize each piece independently.

The consequence is that a good causal model looks like an over-engineered version of a one-line program. Entity binding is the exemplar: instead of one `binding_map` dict, it spells the retrieval algorithm out as separate variables —

- one variable per entity slot (e.g. `entity_g0_e0`, `entity_g0_e1`, …);
- the queried entity *per position*, so "is the person represented here" and "is the food represented here" are separate hypotheses;
- the search result per position;
- the single resolved group index.

You would never write retrieval this way in normal code; the verbosity is the point, because every named variable is a candidate hypothesis. (Entity binding is *almost* fully decomposed — a per-position "matching groups" variable still returns a tuple; a finer decomposition would be a boolean `match` per group×position. Treat full decomposition as the ideal and tuple-bundling as a compromise to justify, not a default.)

**Indefinite arity via a factory.** To support an arbitrary number of entities, positions, or slots, do not reach for a list variable — generate a *family of separate variables in a loop* over config dimensions, inside a factory function. A factory that loops over `max_groups × max_entities_per_group` and emits one variable per slot gives you arbitrarily many, each still a single variable you can localize. That is how you get unbounded structure without bundling.

**Variables can still be coupled — then mechanisms must be total.** Separate variables are not always independent. In a unified-arithmetic model, `entity` and `domain` are distinct inputs, but a token like `"Monday"` is only valid when `domain = weekdays`. An interchange that patches `entity` alone into a base whose domain is `integer` produces an off-distribution `(entity, domain)` pair, and a mechanism that does `decode[domain][entity]` will fail. Either make the mechanism **total** (define it for every combination — e.g. a global-fallback encode) or accept that a counterfactual dataset over one of the coupled variables is ill-posed and drop it. Interchange interventions routinely build input combinations the sampler never produces, so any mechanism on a coupled input has to survive them.

Hold the model against the standard task-quality objectives (granularity, grading totality, input determinism, single-token decoding, value coverage) before going further.

### Curate the hypothesis space, don't enumerate

The space encompasses every variable subset of every causal model that solves the task — far too large to iterate over. The scientific work is **curation**: pick the few compelling, competing hypotheses worth distinguishing, organised around the variables you actually care about. Test one intermediate variable in each experiment. Compare it only with input, output, and intermediate variables that could plausibly be confused with it. Variables may still co-occur at one neural location, but preserve them as separate targets so the evidence for one does not silently count as evidence for the others.

Prune with architecture reasoning: with causal attention, information flows left-to-right, so, e.g., a subset is only realizable at a location at or after the tokens that carry its information.

### Distinguishing hypotheses, and reading the numbers as baselines

An interventional dataset is a list of **(input, counterfactual input)** pairs. To relate two hypotheses A = (model_A, targets_A) and B = (model_B, targets_B) on a pair: run the input through each model, but at the target variables patch in the values those variables take under the *counterfactual* input; compare the two outputs. A **distinguishing measure** is the fraction of pairs on which the two outputs differ. On a single pair the two targets are **confounded** when their interventions give the same output and **deconfounded** (distinguished) when the outputs differ; the reported rate is just the fraction of pairs that deconfound them, and a dataset confounds two hypotheses to the extent its pairs fail to. (One counterfactual per example; cross-model comparisons require the two models to share input-variable names.)

Read these fractions as **interpretive baselines, not pass/fail gates**, and read them **target-centrically** — fix a target hypothesis, and for each alternative ask "how often does this dataset move the alternative's output away from the target's?":

- **A wide dataset distinguishes only imperfectly, and that is fine.** A rate of 0.50 is not a failure — it can be valuable information.
- **A dataset need not separate every pair.** It is fine for a counterfactual dataset to confound two *alternatives with each other*, as long as both are separated from your target.
- **Carry the baseline into interpretation.** If the neural interchange later scores ~0.70 for your target on this dataset, and the dataset only distinguishes the target from a nearby alternative 0.70 of the time, that neural result is confounded with the alternative. The causal-level number is what tells you so.

Two reference alternatives are always in play: the **null** (`targets = []`, output unchanged) and the **all / full-mediation** slice (transplant the whole output). Per target, *vs-null* says whether the counterfactual dataset moves the output at all (≈0 ⇒ the dataset has no power for this target), and *vs-all* says whether the target differs from transplanting the entire output.

**The null↔all relationship is set by the counterfactual, and that is a tool.** A counterfactual that swaps the queried groups but keeps the final output identical between base and counterfactual makes the whole-output transplant ("all") inert — *all collapses onto null on that dataset*. By making the whole-output transplant change nothing, the counterfactual guarantees that any output movement comes from an *internal* variable, which is why it probes the internal pathway. A variant that makes base and counterfactual answers differ separates all from null again.

### Fixable confounding vs. confounding no pair can remove

Two hypotheses producing the same output is **confounding**, and it comes in two grades:

- **Dataset confounding (fixable).** *This* dataset's pairs happen to confound them — the counterfactual dataset does not exercise their difference — but a sharper counterfactual can deconfound them.
- **Confounding no pair can remove.** They give the same output on *every* input pair, so no counterfactual ever deconfounds them. They are the same hypothesis; pick one representative.

How to estimate which you are facing: run a **dedicated large random dataset** — much larger than the design counterfactual datasets, **≥10,000 pairs, 100,000 preferred** — and group the hypotheses that *no pair in it deconfounds* (identical output vectors). Hierarchical equality is the worked example: on its single wide counterfactual dataset a result-equality hypothesis is confounded with its operands (rate 0.00), but the 100k random run finds no pair that deconfounds a whole group of equality hypotheses — they are confounded everywhere — while the two operands are deconfounded on ~0.50 of pairs. The first 0.00 was a fixable design gap; the always-confounded group is a fact about the hypotheses.

A hypothesis that **no pair deconfounds from the null** is inert under the task's sampler — intervening on it never moves the output, so it cannot be tested no matter the counterfactual dataset. Drop such targets, or change the sampler/config so they get exercised. (Entity binding shows this when a query position is never sampled: the variables for that position are structurally live in the DAG but never exercised, so they stay mutually confounded with the null.)

**Whether a pair *can* deconfound two hypotheses depends on the kind of pair.** In a shared-calculator model across arithmetic domains, the pre-reduction sum and the reduced result are confounded by every *within-domain* pair (base and counterfactual share a modulus) but **deconfounded by cross-domain pairs** (e.g., different moduli). A within-domain-only random run never deconfounds them and would wrongly report them always-confounded; a run that samples base and counterfactual domains independently makes most pairs cross-domain and the two come apart. The large random run must include the *kinds of pairs capable of deconfounding* the hypotheses you care about — narrow the pairing and you silently leave distinct hypotheses confounded.

**Caveat — this is empirical, not a proof.** With N pairs the run can miss a pair that would deconfound two hypotheses if such pairs are rare (a 1-in-10,000,000 combination almost never appears at N = 10,000), so they can look always-confounded while really being deconfoundable. "No pair deconfounds them across N random pairs" is a claim whose confidence grows with N and shrinks with the size of the input space — strong evidence, not a theorem; prefer larger N when the input space is large.

### Wide, narrow, and single-token counterfactuals

Design at these altitudes and keep them distinct:

- **Wide counterfactual datasets** — random resampling under task-appropriate balancing, or systematic manipulations (swap order, shuffle, hold the template and resample infills). Broadly interesting and robust to the causal model being wrong. **Your articulated causal model is usually at least a little wrong**; the process is iterative, and wide datasets keep you from gerrymandering yourself into only confirming a model you will later discard.
- **Narrow counterfactual datasets** — sharply targeted pairs that hold one variable fixed and flip another to separate two specific hypotheses.
- **Single-token (minimal-edit) counterfactual datasets** — base and counterfactual differ by *exactly one token*, and that token realizes a single input variable (swap one name, one digit, one operator). They have little sharp distinguishing power — moving only one thing leaves most hypotheses confounded with each other, so expect low deconfounding rates across the matrix — but distinguishing is not their job. **Their job is to track one piece of information.** With everything else held fixed, the single changed variable is the only thing that can move the output, so you can localize where that variable is represented and trace its path layer-by-layer and position-by-position through the network (logit lens, single-pair interchange / path patching). They are the cheapest, most legible way to make contact with the actual flow — what carries this variable, and where — and the natural bridge to the downstream neural localization experiments. Build one for any variable you want to follow through the model; reach for narrow datasets instead when the goal is to adjudicate between competing hypotheses.

Expect overlap: a narrow dataset is often a slice of a wide one, and a single-token dataset is the narrowest narrow — its breadth and its purpose just pull in different directions (minimal edit, but aimed at flow-tracing rather than deconfounding). Most tasks tend to *under-provide* both narrow and single-token datasets, which is what a critique pass surfaces.

### Train/eval splits and generalization

These datasets feed downstream **supervised localizers** — methods like a learned rotation onto a subspace, or learned masks over components — that train parameters on counterfactuals. So:

- **Never train and evaluate on the same pairs.**
- **Hold out structure, not just instances** — eval splits can hold out entities-within-templates, and optionally whole templates, so a high eval score reflects a *generalizing* localization, not memorized infills.
- **Train wide, evaluate everywhere** — train on wide swaths; evaluate on both wide and narrow classes, reading per-hypothesis signal strength on the narrow ones.
- **Watch overfitting** — a narrow-train / narrow-eval win is suggestive but must be compared against other settings before it counts; supervised localizers can overfit a narrow distribution.

Record whether each dataset is wide, narrow, or single-token and its split (train/eval, what it holds out).

**A "shared mechanism" hypothesis is a generalization question, not a distinguishing question.** Whether *one* module computes a variable across several conditions, versus a separate module per condition, is behaviourally invisible — both produce the right output, so every pair confounds them and any distinguishing measure returns 0. The claim lives entirely in whether a localizer **trained on one condition generalizes to another**. The shared-calculator model is the sharp case: "a single calculator computes the sum for all arithmetic domains" cannot be read off any distinguishability matrix; you settle it by training the localizer on one domain's sum and evaluating on another — which is exactly why those per-domain datasets are split as train and held-out eval. When the hypothesis is about *sharing*, the cross-condition split *is* the experiment.

## Dataset collection

Causal analysis needs a comprehensive dataset of clear demonstrations of the model behavior, eliciting all of its relevant facets. As a rule of thumb, a sufficient dataset contains at least 100 samples, and — most importantly — multiple demonstrations of each possible concept value. It optionally carries a metric of conceptual distance between samples.

For example, when analyzing the concept of emotion across the valence-arousal scale, the dataset should contain multiple examples spanning the full range of emotions, annotated with points in valence-arousal space and a distance metric within that space.

Choose the dataset type by the claim:

- For a free-form test of whether a representation affects a target behavior, generate synthetic input texts that elicit all possible characteristics of the output behavior, or gather them from real webtext. The dataset consists of single samples, optionally equipped with a metric quantifying conceptual order or semantic distance.
- For testing whether the model implements a pre-defined causal mechanism, develop a causal model and a counterfactual dataset in the hypothesis-generation step. The result consists of counterfactual (base, counterfactual) pairs.

Formulating high-level causal models and designing the counterfactual datasets that distinguish them is the prerequisite specific to interchange/causal-abstraction analysis — ablation and steering require nothing of the sort. **All interventions on the high-level causal model are CPU-only; make sure no GPU is used.**

## Choosing an intervention method

There are three kinds of intervention. Choose by the claim:

- To test whether a representation is **necessary**, or whether an input effect flows *through* a representation, use **ablation**. It is a good first-pass smoke test of whether the representation affects the target behavior at all.
- To test whether a representation can **control** a behavior, use **steering**.
- To test whether a representation **realizes a high-level causal variable** — whether a high-level causal model abstracts the neural computation, or whether an algorithm is implemented by the network — use **interchange interventions**, comparing neural counterfactual behavior to high-level counterfactual behavior.
- If you are unsure how to proceed, default to ablation or steering.

Ablation and steering are self-contained in this handbook. Interchange interventions additionally require a certified causal model and counterfactual dataset from the hypothesis-generation step.

## Ablation

Intervene on a representation to destroy or remove information, then measure whether the target behavior degrades in the predicted way. Use ablation to test whether a site, direction, subspace, feature, channel, head, or token is necessary for a behavior. Common ablations include setting activations to zero, replacing them with an empirical mean, adding calibrated noise, nulling a direction or subspace, masking channels/features, or replacing a component with a distribution-matched baseline.

Ablation evidence supports claims about necessity or mediation, not semantic identity by itself. That is, it cannot tell you what the representation actually represents (if anything). A behavior drop after ablation shows that the ablated information matters for the metric; it does not prove that the ablated component "represents" the hypothesized variable unless paired with stronger controls or interchange-style tests.

Default to running both mean and zero ablations, and sweep across many representations.

### Choosing the ablation baseline (mean vs. zero vs. noise)

These are general guidelines, not hard rules — but they apply equally to direct ablation and to the **corrupt** step of causal tracing, since corruption is itself an ablation.

- **Mean ablation is a good default**, and is the right baseline for the **residual stream**. The residual stream carries *everything*, so zeroing it out is an unnatural state for the model to be in; replacing a site with its empirical mean removes the example-specific information while keeping the activation on-distribution.
- **Zero ablation is the better choice for attention heads and MLPs.** These components *write* their output into the residual stream, so zeroing them just deletes their contribution — a clean "what if this component had written nothing" counterfactual.
- **Noise ablation is rarely the right reach for general necessity testing.** Mean and zero baselines cover almost every case. Its main use is faithfully replicating a method that prescribes it — e.g. ROME-style causal tracing, which corrupts with calibrated (≈3σ) noise.

### Causal tracing (corrupt then restore)

Causal tracing inverts the ablation question. Ablation removes a site and asks whether the behavior breaks — a test of **necessity**. Causal tracing instead degrades the information everywhere it enters, establishing a broken-behavior floor, then **restores one site at a time** and asks whether the behavior recovers — a test of **sufficiency**, identifying which sites carry that information forward (where it is **mediated**). It is the same intervene-and-measure logic as ablation, run against a corrupted backdrop rather than a clean one.

The pattern, abstractly:

1. **Corrupt.** Ablate the activations where the information enters the computation — a zero, mean, or noise ablation — and record the degraded behavior as the floor.
2. **Restore and sweep.** For each candidate site, run the model with everything still corrupted *except* that one site, whose clean value is restored from a parallel clean run — making the restore step an interchange. Measure how much behavior returns.
3. **Localize.** Across the swept sites, those whose restoration recovers behavior are the mediators; laid out over a layer × token grid, this is the familiar restoration heatmap.

## Steering

Intervene on a representation by adding, subtracting, scaling, or otherwise manipulating a direction, feature, subspace, or neuron in order to predictably change model behavior. Use steering to test whether (part of) a representation provides causal control over a behavior. Steering can take activations off the input-induced activation manifold, so successful steering is evidence of causal control — not, by itself, evidence that the model naturally uses the representation that way.

Default to a "steering vector" approach, where a vector is added into a representation and scaled by a scalar alpha. Steering vectors typically come from (1) unsupervised decomposition methods like sparse autoencoders, (2) correlational analysis like probing or PCA, or (3) difference-in-means vectors.

When the host research environment provides an operational steering pipeline, use
it for SAE-feature extraction, difference-of-means vectors, and geometry-aware
follow-up when a concept is not well described by one linear vector.

A difference-in-means vector is constructed as follows. First, build two datasets contrasting along a particular quality — e.g. images of rabbits vs. images of dogs. Then harvest the activations produced for each, compute the average activation for each dataset, and use the difference of those two means as the steering vector.

A more involved approach fits a manifold to representation space and interpolates along it as the steering intervention. Varying alpha for a linear steering vector interpolates along a straight line; steering along a manifold is the analogue for a non-Euclidean geometry (see "Manifold steering" below).

A steering vector that changes behavior is not automatically an explanation of how the model normally performs the behavior. It may reveal a control knob rather than a naturally used causal variable.

### Manifold steering

Manifold steering manipulates a low-dimensional curved structure in activation space. It generalizes the difference-in-means steering vector to multidimensional structures with non-linear trajectories. We move from linear directions to bounded manifolds, so ideal manifold steering does not go off-distribution for activations occurring on natural-language webtext — the model's original training distribution. Manifold fitting and steering takes multiple steps:

1. Obtain a dataset that exhibits all facets of the target behavior.
2. Identify the subspace of activation space at the desired model site that contains the geometry, either correlationally (PCA) or causally (DAS):
   1. For PCA, cache the point cloud of activations at the site and take the subspace that captures 95% of the variance.
   2. For DAS, train as described under "Interchange interventions."
   3. The subspace dimensionality is usually smaller than 20.
3. Based on the geometric analysis, choose a method to fit a continuous manifold (splines and thin-plate splines are common choices).
4. Intervene along the continuous manifold (steer).
5. Quantitatively analyze the manifold fit — fit the manifold in output space as well, and compute isometry metrics.

## Interchange interventions

This is the core method for determining whether a neural network implements an algorithm or causal process — known as causal abstraction analysis. Such an analysis formulates hypotheses as high-level causal models and follows a hypothesis-testing framework over interchange interventions.

The core idea: a high-level causal variable in a causal model that performs some task is a faithful abstraction of an internal representation of a network performing the same task. We evaluate such claims by performing interchange interventions on both the network and the causal model, and measuring the agreement between the two intervened outputs.

An interchange intervention is a special intervention: run a network on an original input while fixing some internal representation to the value it takes under a counterfactual input. Run the matching interchange on the high-level causal model — same original and counterfactual input, targeting the high-level variable(s). Then measure the agreement between the network's intervened output and the causal model's intervened output (e.g. exact match or logit-based metrics), aggregated across a full dataset of (original, counterfactual) pairs.

Interchange/causal-abstraction analysis requires competing high-level causal models and counterfactual datasets that distinguish them — the prerequisite specific to this method. Build and certify the causal models and counterfactual datasets in the hypothesis-generation step; that step is CPU-only and never touches the target network. Its handoff format is documented in `answer-research-question/hypothesis-generation/hypothesis-report-format.md`. After receiving the handoff, run the interventions on the target network — which usually requires GPU resources.

Unlike steering and ablation, interchange uses activation values that actually occur during normal model execution, making it better suited for testing whether the model controls itself through the hypothesized variable.

### Localizing variables: DAS and DBM

Distributed Alignment Search (DAS) and Desiderata-Based Masking (DBM) are supervised methods for localizing abstract causal variables to (parts of) neural representations. Both use interchange interventions on a high-level causal model as a source of supervision to learn a site in the network on which to perform interchange interventions.

DAS targets a fixed hidden vector and learns a low-dimensional subspace parameterized by a low-rank matrix with orthogonal columns. At first the subspace is random and there is no correspondence between the intervened causal model and the intervened network; as learning proceeds, DAS finds a subspace that increases their alignment. For an original activation vector `h_o`, a counterfactual activation vector `h_c`, and a learnable matrix with orthogonal columns `R`, the intervention is:

```
h = h_o + R(R^T h_c − R^T h_o)
```

DBM targets a set of potential representations (or a set of potential parts of a single representation) and learns to select a subset of sites via masks. For each potential (part of a) representation, the masked intervention is:

```
h_intervened = (1 - m) * h_base + m * h_source
```

Throughout training, the masks are clamped to the range [0, 1].

For presenting interchange and DAS localization results — the positive-control discipline, the variable-by-variable structure (inputs → output → intermediates), and the plot-type rules — follow the interchange/DAS localization report format at `implementation/references/interchange-das-localization-report-format.md`.

## Limits of interpretation

State exactly what the experiment supports and what it does not. Use conservative language.

Allowed:

- "This intervention caused a change in the metric under this dataset and intervention."
- "This site is a candidate mediator for this behavior."
- "This direction provides causal control over the behavior."
- "This representation passed an interchange-intervention test for this high-level variable under this alignment."

Avoid:

- "This neuron represents concept X" from ablation alone.
- "This feature plays this role in the mechanism" from steering alone.
- "The model implements algorithm A" without an explicit high-level causal model and sufficient interchange tests.
- "The circuit is complete" without completeness checks and off-target controls.
- "No effect means no role" without considering redundancy, compensation, insufficient intervention strength, or metric insensitivity.
