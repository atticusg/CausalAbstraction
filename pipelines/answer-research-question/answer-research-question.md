# Answer a research question

A protocol for taking a question about what a language model is doing internally
and turning it into a claim you can defend. It runs in six steps. Each step has
its own document in a subdirectory beside this one; this file owns the shape of
the whole thing — the order, the roadmap that plans it, and the routing that
decides where you go after a test comes back.

Read this file before entering any step.

## The flow

```
                    ┌────────────────────────────┐
                    │    behavioral analysis     │
                    │  can the model do it, and  │
                    │  how does it fail?         │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
        ┌──────────▶│ exploratory experimentation│
        │           │  cheap probes for signal   │
        │           └─────────────┬──────────────┘
        │                         │
        │                         ▼
        │           ┌────────────────────────────┐
        │  ┌───────▶│   hypothesis generation    │
        │  │        │  a causal model + the      │
        │  │        │  counterfactuals to test it│
        │  │        └─────────────┬──────────────┘
        │  │                      │
        │  │                      ▼
        │  │        ┌────────────────────────────┐
        │  └────────┤    hypothesis testing      │
        │           │  the hypothesis against    │
        └───────────┤  its alternatives          │
                    └─────────────┬──────────────┘
                                  │  strong positive result
                                  ▼
                    ┌────────────────────────────┐
                    │    generalize results      │
                    │  how far does the claim    │
                    │  actually reach?           │
                    └─────────────┬──────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────┐
                    │       save results         │
                    └────────────────────────────┘
```

| Step | Document | The question it answers |
|---|---|---|
| 1 | [`behavioral-analysis/`](behavioral-analysis/behavioral-analysis.md) | Can the model do this at all, and what does it get wrong? |
| 2 | [`exploratory-experimentation/`](exploratory-experimentation/exploratory-experimentation.md) | What cheap internal evidence is there, before committing to a hypothesis? |
| 3 | [`hypothesis-generation/`](hypothesis-generation/hypothesis-generation.md) | What algorithm might the model be running, and what would distinguish it from the alternatives? |
| 4 | [`hypothesis-testing/`](hypothesis-testing/hypothesis-testing.md) | Does the evidence favor this hypothesis over its alternatives? |
| 5 | [`generalize-results/`](generalize-results/generalize-results.md) | How far does the claim reach beyond the exact setting it was tested in? |
| 6 | [`save-results/`](save-results/save-results.md) | What gets written down, and where? |

## Start with a roadmap

**The first thing to do is not an experiment.** Before running anything, write a
roadmap: a sketch of how you expect this whole investigation to go, step by step
through the six above. Copy [`ROADMAP_TEMPLATE.md`](ROADMAP_TEMPLATE.md) into your
working directory as `ROADMAP.md` and fill it in.

The roadmap is not a schedule and not a contract. It is a statement of what you
currently expect, written down so that the ways it turns out to be wrong are
visible instead of absorbed. Its value is mostly in the parts that later change.

A roadmap that is worth writing says, for each step:

- **What you expect to find.** Not "run a logit lens" — "I expect the answer to
  become decodable somewhere in the second half of the network, and if it doesn't
  the task is probably not solved the way I think it is."
- **What would make you skip or shorten this step.** Some questions arrive with
  behavioral analysis already done. Say so, rather than re-deriving it.
- **What would make you stop entirely.** The most useful line in a roadmap is
  often the one that names the result that would kill the project.

It also names, up front:

- **The research question**, in a form that has an answer. "How does the model do
  X" is a topic, not a question. "Is the answer to X computed at the final token,
  or carried from the operand tokens?" is a question.
- **What changes once you know.** If nothing does, that is worth discovering
  before the compute is spent, not after.

### Revising the roadmap

**Update `ROADMAP.md` at the end of every step, before starting the next one.**
This is the mechanism that makes the loop in the flow chart honest — without it,
re-entering exploratory experimentation for the third time looks identical to
entering it for the first.

Each revision appends an entry to the roadmap's revision log saying: what the step
actually produced, how that differs from what the roadmap expected, and what in
the plan below changed as a result. Do not rewrite history — the earlier
expectation stays on the page. A roadmap whose revision log is empty after four
steps is either a very lucky investigation or one that is not being honest with
itself.

Revise the plan, not just the log. If exploratory experimentation showed the
behavior lives entirely in the first six layers, the hypothesis-testing plan that
assumed a late-layer mechanism is now wrong and should say so.

## A step is a phase, not an experiment

Each box in the flow chart is a **phase of work**, not a single run. A phase may
be one experiment; more often it is several, and occasionally it is a dozen.
Exploratory experimentation in particular is a family of independent probes that
happen to share a purpose.

So: decompose every step into units of work in the roadmap before starting it, and
track them there. Each step document names the units that are natural for that
step under a **Typical units of work** heading — use those as the starting
decomposition and adapt.

Two rules keep this from degenerating:

- **A unit is something that can come back with a result.** "Understand the
  attention pattern" is not a unit. "Ablate each attention head at the final token
  and record which ones move the output" is.
- **A unit that has run and produced nothing still counts.** Record it. Half of
  what a roadmap is for is preventing the same dead probe from being run twice by
  someone who forgot it was dead.

## Routing after hypothesis testing

Hypothesis testing is the only step with a branch. The evidence either supports
the hypothesis over its alternatives or it doesn't, and what you do next depends
on which.

**Strong positive result** — the hypothesis beat its alternatives on a test that
had the power to separate them. Go to **generalize results**.

**No strong positive result** — the test came back null, ambiguous, or against the
hypothesis. You are going back into the loop, and the choice is between two doors:

- **Back to hypothesis generation** when the evidence you already have suggests
  other hypotheses worth proposing. This is the common case after a clean
  refutation: the test worked, it told you the answer is not this, and the same
  observations that motivated the dead hypothesis usually motivate a couple of
  live ones. It is also where you go when the test could not separate two
  candidates — the fix there is a sharper counterfactual, which is a
  hypothesis-generation problem, not a testing problem.
- **Back to exploratory experimentation** when it doesn't. If you have run out of
  hypotheses the existing evidence can support, generating another one now means
  guessing. Go get more signal first — a different probe, a different site, a
  different slice of inputs — and let the next hypothesis come from something you
  observed rather than something you hoped.

The distinction is worth being deliberate about, because the cheap move is always
to generate another hypothesis, and a sequence of hypotheses generated from no new
evidence is just enumeration. If you find yourself proposing a third hypothesis
without having looked at anything new since the first, that is the signal to take
the other door.

Either way, **record the negative result and the routing decision in
`ROADMAP.md`** before re-entering. The refuted hypothesis is a finding; it belongs
in the final writeup whatever else happens.

## Working directory

Everything an investigation produces lands under one working directory. The
documents refer to it as `$WORKDIR` and assume an absolute path. A layout that
works:

```
$WORKDIR/
├── ROADMAP.md          the plan and its revision log
├── docs/               protocol and workflow documents you author
├── data/               serialized dataset tables
├── runs/               run trees produced by execution
└── result/             the writeup and the figures it references
```

Nothing is written into the causalab checkout itself; causalab is read-only source
that runs against `$WORKDIR`.

## What is not in the six steps

Three reference trees sit beside this one and are entered from within the steps
rather than being steps themselves:

- [`../implementation/`](../implementation/) — hands-on work against the causalab
  codebase: building a task package, the analysis catalog, running things, common
  failures. Hypothesis generation and hypothesis testing both route into it.
  **Substantially stale** — see the banner on that document.
- [`../explore-subspace/`](../explore-subspace/) — the entry path for when you are
  handed a subspace and need its description verified before designing causal
  experiments against it. **Stale.**
- [`../subspace-causal-analysis-pipeline/`](../subspace-causal-analysis-pipeline/)
  — the older end-to-end pipeline for adjudicating whether a *given* subspace is
  causal. **Stale**; its sequencing role is what this document now does.

## Execution stubs

The protocol refactor retired the Hydra runner, the `analyses/` tree, `methods/`
as Python, and SLURM dispatch. Documents in this tree keep the science and mark
every place where the concrete invocation used to be:

> **Execution: stub.** What this needs, and why it isn't written yet.

Those are deliberate and greppable. `../README.md` indexes them.
