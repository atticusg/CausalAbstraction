# Step 1 — Behavioral analysis

Establish what the model actually does on the task, from the outside, before
looking inside it. Nothing downstream is interpretable without this: an
intervention that changes a behavior the model was never reliably producing tells
you nothing, and a mechanism localized on a task the model solves by shortcut is a
mechanism for the shortcut.

Three questions, in order.

## Can the model solve the task?

Find a prompt the model reliably solves, and know how reliably.

Start from the task and examples you were given. If there are none, write several
representative prompts yourself. Greedy-decode them and check the outputs against
the intended answers.

**Decide about in-context demonstrations before you start.** Few-shot examples let
a model pattern-match from the demonstrations instead of running the algorithm you
want to study, which quietly changes what the whole investigation is about. Prefer
zero-shot. If you use few-shot, say so in the roadmap and treat any mechanism you
find as a mechanism for the few-shot condition until shown otherwise.

Vary phrasing, format, and token choices until the model is reliably correct.
Things that routinely matter more than they should: whether the answer follows a
space, whether the answer is a single token under this tokenizer, whether the
prompt ends with a colon or a newline, and whether the answer space is
first-token-distinct.

**If representative variations fail, stop.** Report what you tried and what the
model did instead. Do not proceed to internals on a prompt the model cannot solve
— you will be localizing a failure.

Record: the working prompt verbatim, the accuracy over a reasonable sample, and
the variations that did and didn't work.

## What systematic errors does it make?

Accuracy is a summary; the errors are the data. A model at 80% that fails
uniformly at random is telling you something very different from a model at 80%
that fails on exactly the inputs where two operands collide.

Sort the failures. Look for:

- **Structure in the inputs that fail.** Group by every input variable you have. A
  failure rate that varies sharply with one variable is a lead — it usually means
  the model's algorithm is right for part of the input space and wrong outside it.
- **Structure in the wrong answers.** What does it say instead? A model that
  answers with the *other* operand is doing something specific and findable. A
  model that answers with the most frequent value in the training distribution is
  falling back on a prior.
- **Off-format outputs.** Answers outside the intended answer space are usually a
  prompt problem, not a model problem — go back and fix the prompt.
- **Position effects.** Same content, different position in the prompt, different
  accuracy. Very common, and it constrains where the mechanism can be.

Errors are also where the interesting hypotheses come from. The cheapest route to
a good hypothesis is often "the model gets these right and those wrong — what
algorithm has exactly that failure boundary?"

Record a **specimens table**: concrete inputs, the model's actual output, the
expected output, and which error class each belongs to. Aggregate rates alone are
not enough — the specimens are what the reader checks your reasoning against.

## For multi-token generations, what kind of output is it producing?

Most causal-abstraction machinery reads one position and scores one token. When
the behavior of interest is a longer generation, you need a characterization of
the output before you can pick a readout.

- **Does the behavior have a single decisive token?** Often it does — the first
  token of the answer, a yes/no, a chosen letter. If so, establish that it is
  genuinely decisive (the rest of the generation follows from it) and score that
  token. This is the case that makes everything downstream easy, so look for it
  first.
- **If not, what is the space of outputs?** Cluster the generations. Are there a
  few recurring modes? A length effect? A format the model falls into? Name the
  categories and how you will assign a generation to one — that assignment
  function is what any later metric will be built on.
- **Does the property you care about appear at a stable position?** If the
  decisive content moves around between generations, positional interventions will
  smear across it. Either find an anchor, or restrict the task until one exists.

A behavior you cannot score consistently is a behavior you cannot test, so this
question decides whether the rest of the protocol can run at all. If the answer is
that there is no stable readout, the right move is usually to narrow the task
until there is one — and to say in the roadmap that you did.

## Typical units of work

- One unit per prompt-format candidate you evaluate.
- One unit per error-structure hypothesis you check (by input variable, by
  position, by answer identity).
- One unit for the output characterization, if the behavior is multi-token.

## Task quality

If you will be building a causalab task package from this behavior, it has to
satisfy five properties — answer-space granularity, grading totality, input
determinism, single-token decoding, and value coverage. They are cheaper to check
now than after the task is written. See
[`../../implementation/setup-task/instructions/task_quality_objectives.md`](../../implementation/setup-task/instructions/task_quality_objectives.md).

## Handoff

You leave this step with: a working prompt, an accuracy figure, a specimens table,
a characterization of the errors, and a decision about what will be scored. Update
`ROADMAP.md`, then go to
[`../exploratory-experimentation/exploratory-experimentation.md`](../exploratory-experimentation/exploratory-experimentation.md).

> **Execution: stub.** Batch greedy decoding over a prompt list used to run through
> the Hydra `exploration` analysis in `probe` mode, which the protocol refactor
> retired. On the document layer this is a protocol with a `lm_head` read at the
> final position and a `match` metric against the answer column, but the shape of
> a "just generate and show me" invocation is not settled. Fill in once causalab
> stabilizes.
