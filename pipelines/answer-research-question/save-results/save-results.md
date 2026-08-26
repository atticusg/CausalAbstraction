# Step 6 — Save results

Record the claim, its boundary, and the evidence behind it, so that someone else —
including you in three months — can find out what was done and check it.

> **Execution: stub.** How results are persisted is not settled. The pieces that
> exist today are the run tree written by the workflow runner (per-step output
> directories, the saved parquet metric tables and safetensors bundles, and the
> `workflow.json` manifest), plus the campaign and per-point digests that identify
> a document. What is not settled is what sits above that: where a writeup lives,
> how it references a run tree durably, whether validated causal models and task
> packages get promoted back into causalab, and what the identity of a whole
> investigation is. Fill in once causalab stabilizes.

What follows is what should be recorded, independent of the mechanism.

## The claim

Stated once, precisely, with its scope attached. The scope is not a caveat section
at the end — it belongs in the sentence. "In Llama-3.1-8B, on the weekdays task
under the zero-shot template, the day-offset variable is mediated by a
2-dimensional subspace of the residual stream at layer 18, final position" is a
claim. "The model represents day offsets" is not.

Include what was ruled out. A claim that survived three alternatives is a stronger
object than the same claim with no rivals named, and the reader cannot tell them
apart unless you say.

## The evidence

For each result in the writeup, a pointer to the artifact that backs it — the run
tree, the table, the figure's underlying data. A number in prose with no path
behind it cannot be checked, and results that cannot be checked stop being useful
about as fast as they stop being remembered.

Record enough to re-run: the model key and revision, the dataset refs and their
digests, and the documents themselves. Protocol documents are the good case here,
since a document plus its digest *is* the specification of the experiment.

## The negative results

Every hypothesis that was tested and refuted, every probe that came back empty,
every generalization axis that failed. These are the most commonly discarded and
the most commonly re-derived part of a project. The roadmap's revision log already
has them; the job here is to not drop them on the way to the writeup.

## The roadmap

`ROADMAP.md` and its revision log go with the result. It is the record of how the
investigation actually went, as opposed to the tidied-up version the writeup
presents, and it is what makes the difference between a result someone can build on
and a result someone has to redo.

## What is worth promoting

Some of what an investigation produces is reusable and should not stay in a working
directory:

- **A validated causal model and its counterfactual generators** — a task package
  others can test against.
- **Protocol and workflow documents** that turned out to be generally useful,
  rather than specific to this question.
- **Fitted artifacts** — rotations, dictionaries — with the identity stamps that
  let a later run verify it is using the right one.

Whether and how these get contributed back into causalab is part of what the stub
above is waiting on.

## Typical units of work

- One unit for the writeup.
- One unit for collecting and checking the evidence pointers.
- One unit per artifact promoted out of the working directory.
