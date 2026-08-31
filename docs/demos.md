# Writing a demo

A demo is one markdown file that answers one research question with documents
you can run. It is not a tour of the API and it is not a notebook: the
protocol layer made execution a `causalab run` away, so what a reader needs is
the *experiment* — the question, the document that encodes it, the design
choices behind it, and the numbers, in that order.

This page is the format. It exists so that four demos written by four people
read like one document, and so that a demo that has gone stale says so instead
of quietly lying.

Shipped demos: [`demos/`](../demos/).

## 1. What a demo is

**One demo, one question.** If it needs two models or two tasks, it is two
demos. If it needs four questions that feed each other, it is a *workflow
demo* (§3) and each question is a step.

**The document is the thesis.** A demo's centre is the JSON: reading it should
be enough to know what ran. Prose exists to say why the document says what it
says, and what the numbers mean. Prose that restates a JSON field is deleted.

**Every claim carries its number, and every number carries its floor.** "IIA is
high at the answer slot" is not a result. "IIA is 0.92 at the answer slot,
against a floor of 0.22 set by the pairs whose two answers agree" is.

**A demo is reproducible or it is marked.** The header's `Reproduced` field
(§2) is not decoration — it is the difference between a record and an
advertisement.

## 2. Section skeleton

Seven sections, in this order. A section is never silently dropped; an empty
one says why it is empty.

```markdown
# <Title — the question, not the method>

<the header table>

## TL;DR
## The protocol
## Run it
## Experimental design
## Results
## Limits
## Next
```

### The header table

Immediately under the title, before any prose. Every row is mandatory.

```markdown
| | |
|---|---|
| **Question** | Where does the model carry the answer symbol? |
| **Method** | interchange intervention, scored by IIA |
| **Model** | `meta-llama/Llama-3.2-1B-Instruct` @ `main`, bf16 |
| **Data** | `mcqa/pairs_n64_s0` — 64 pairs, `different_symbol` design |
| **Documents** | [`protocols/mcqa_locate_scan.json`](protocols/mcqa_locate_scan.json) |
| **Cost** | 128 points × 2 forwards; one 16-layer model, minutes on one GPU |
| **Reproduced** | ⚠ figures carried from the pre-refactor reference run |
```

`Reproduced` takes one of exactly two forms:

| form | means |
|---|---|
| `✓ <date>, <engine>, digest <first 8 hex>` | the numbers below came out of the documents above, at that digest |
| `⚠ <what is stale, in one clause>` | they did not — say which numbers are borrowed and from what |

Anything else is not a value of this field. A demo whose figures came from a
retired pipeline is useful; a demo that hides it is not.

### TL;DR

At most five sentences: the question, the method in one clause, the answer, and
the one number that carries it. Written last, read first. No forward references
("as we will see below") — a reader who stops here has the finding.

### The protocol

The **complete, unelided** document, once, in a fenced `json` block, and
immediately beside it a **link to the file it copies**. Not a fragment, not a
diff against another demo, and not a reflow: the block is the file's bytes, so
that pasting it back over the file is a no-op. If it is too long to read, the
demo is too big.

Both halves are load-bearing, and neither is redundant. The copy is what a
reader on GitHub sees without a second click — a demo whose thesis lives in
another file is a demo nobody reads. The file is what `causalab run` reads, so
it is the *copy* that can be wrong, and a wrong copy is worse than no copy: the
prose reads as authoritative while the run uses the other bytes. Two places for
one fact is precisely the arrangement that needs a check rather than a habit,
which is why `tests/demos/test_demos.py` fails the moment a block stops matching
the file it names.

Then two things, in this order:

1. **A flow chart** of the model graph — which forward reads what, which write
   consumes it, what the metric reduces. Mermaid, because GitHub renders it and
   a reader can edit it:

   ```mermaid
   flowchart LR
     CF["original<br/>on counterfactual"] -->|v_cf| P["patched<br/>on base"]
     P -->|logits| M["match vs cf_answer"]
   ```

2. **A reading, section by section**, only where the document is not
   self-evident. A three-column table (`section` · `says` · `why this and not
   that`) beats three paragraphs. The third column is the one that earns its
   place: the reader can see *what* the field says.

A demo that touches no network — a task-design demo — has no protocol
document. It puts the **dataset build** in this section instead and says so in
its first line, because the artifact that fully determines the experiment is
what this section is for.

### Run it

Three commands, in this order, with their real output pasted in:

```bash
uv run causalab validate <doc> --data-root <root> --data
uv run causalab explain  <doc> --data-root <root>
uv run causalab run      <doc> --data-root <root> --out runs/<name> \
    --device cuda --dtype bf16
```

`validate` and `explain` are pure — no weights, no network, no accelerator — so
a reader can run the first two on a laptop before deciding to spend a GPU. That
is the point of pasting them: `explain`'s `points` and `forwards` **are** the
cost estimate, and the demo must not restate them in prose.

Then one short **hardware** paragraph: how many GPUs, how much memory, roughly
how long. Say what it was measured on, or say it is an estimate.

### Experimental design

The numbered questions, `Q1 … Qn`, each with:

- **what it asks**, in one sentence;
- **what would answer it** — the number and where it comes from;
- **what a null looks like** — the value the number takes if the answer is no.

This is where predictions belong. A prediction stated here and contradicted in
Results is the most valuable thing a demo can contain; a prediction quietly
adjusted afterwards is the least.

Design choices that constrain the questions live here too — the counterfactual
design, why this position and not that one, why this metric. One bold question
per choice, then the mechanism:

> **Why fixed indices and not named variables?** …

### Results

Exactly one `###` subsection per question, in the same order, with the same
number and the same heading text. A question with no result gets its
subsection anyway, saying what is missing and what would produce it.

Each subsection is: the figure or the number, then one paragraph reading it,
then a one-line **verdict** that answers the question as asked.

Separate the guaranteed from the found. An observation that follows from the
setup is a sanity check and is marked ✓; only what could have come out
otherwise is a finding.

> ✓ The embedding row flips at the symbol token — that is where the two
> prompts differ, so anything else would be a bug.
>
> **Finding.** The flip leaves that column at L12 and appears at the answer
> slot from L13 on.

**Figure captions** carry three things and nothing else: what is plotted, which
run produced it, what to look at.

### Limits

At most five bullets. What the demo does not show, what would falsify it, which
confound survives. A demo with no limits section has not been read carefully
enough to have one.

### Next

The demos that follow, one line each, naming what they take from this one.

## 3. Workflow demos

A demo whose question decomposes into questions that feed each other is a
**workflow demo**. It keeps the same seven sections, with three differences:

- **The protocol** section leads with the *workflow* document, unfolded: the
  workflow is the thesis, the steps are its sentences. The step documents are
  inlined too — §2's rule holds for every document a demo runs — but after
  the step table and each inside a `<details>` block, so the chain stays the
  thing the section reads as. The `<summary>` names the step and its path;
  the markdown link in the step table is the cross-reference.
- The flow chart draws the **derived schedule**, not the model graph. `explain`
  prints the levels; the chart is that, drawn.
- **Experimental design** numbers its questions `RQ1 … RQn` and each maps to
  named steps. `Results` keeps one subsection per RQ, as always.

The handoffs between steps are the interesting part, so say them: which value
the `select` step emits, which document's `set` consumes it, what happens if
the earlier step chooses differently.

## 4. Layout

```
demos/
├── README.md                  # the index: one line per demo
└── <demo>/
    ├── <demo>.md              # the demo, or 01_x.md, 02_y.md for a series
    ├── protocols/*.json       # every document the demo runs
    ├── workflows/*.json       # a workflow demo's chain
    ├── data/<ref>/*.json      # the serialized tables, with their manifests
    └── figures/*.png
```

Documents live **beside** the demo, not in `causalab/configs/`. A demo's
document is free to be pinned, small, and pedagogically shaped; the shipped
presets are none of those things, and a demo that edits one to make a point
breaks the preset.

**Tables are committed** with their `<ref>.manifest.json` sidecar. The table is
a build product, the manifest is the recipe, and a reader who wants to know
where 64 rows came from reads the sidecar rather than guessing.

**`.png` is the figure format.** `.pdf` only when a vector figure is genuinely
needed, `.html` only for a figure that must be interactive
(`causalab.io.plots.figure_format`). A figure carries no record, so a demo
whose figure matters declares the numbers beside it — the shipped
`workflow_figures` script writes a `plotted` table for exactly this.

## 5. Voice

Distilled from the notebook demos this format replaces; the point is that the
tone was already right.

**Person and tense.** First person plural, present tense — "we patch every
cell", not "the cell is patched" and not "you should patch". Second person
appears in exactly one place: the invitation to change something and re-run.

**Emphasis carries meaning, not volume.** **Bold** marks a term at its first
use and never again. *Italics* mark the pivotal word in a contrast — *which*
layer, *one* pair, *many* pairs. Backticks mark anything the reader could type.

**One idea per paragraph, three sentences at most.** A long explanation is a
table that has not been written yet.

**Say the number in the sentence.** "IIA 0.96 at L0–L2, decaying to 0.14 by
L13" — not "high early and low late".

**Comments in a document carry the why.** `"description"` is the field for it;
JSON has no comments, which is the reason the field exists.

**Failure modes are content.** Say what a bad result looks like and what
causes it, in the same voice as the good one — a reader who gets the bad result
is exactly the reader who needs the demo.

**Be honest at the point of the claim.** Where a choice is imprecise, a null is
a mathematical identity rather than a bug, or a figure predates the document
below it, say so in the sentence that makes the claim — not in a footnote and
not nowhere.

### Not this

| don't | because |
|---|---|
| "simply", "just", "of course" | if it were simple the demo would not exist |
| "as we can see" | say what is seen |
| a figure with no claim | a picture is not a finding |
| a number with no floor | 0.5 is a triumph or a coin flip, and the reader cannot tell |
| a result with no question | Results mirrors Experimental design, one for one |
| Python that reimplements a document | the document *is* the experiment; run it |

## 6. Checklist

Before opening the PR:

- [ ] every JSON in the demo passes `causalab validate … --data --data-root <root>`
- [ ] every `explain` block is pasted output, not typed by hand
- [ ] every document is inlined verbatim, beside a link to the file it copies
- [ ] `Results` has one subsection per `Experimental design` question, same order
- [ ] every number in prose has a floor, a chance level, or a unit
- [ ] every figure caption says what produced it
- [ ] the header's `Reproduced` field is true
- [ ] `Limits` is not empty

`tests/demos/test_demos.py` checks the mechanical half of this list — the
documents, the links, the inlined copies, the quoted digests and the section
skeleton. The rest is review.
