# Step 1 — Behavioral analysis

Establish how the model behaves before looking inside it. Otherwise, you may study
a shortcut instead of the intended behavior.

## Can the model solve the task?

Find a prompt that the model solves reliably and measure its accuracy. If the model
fails representative variations, stop rather than studying a task it cannot solve.

- Decide whether to include examples in the prompt **before** you start. These
  examples may let the model match a pattern instead of using the algorithm under
  study. Prefer a prompt with no examples. If you include examples, limit every
  later claim to that prompt condition.
- Vary phrasing, format, and token choices. What usually matters: whether the
  answer follows a space, whether it is a single token under this tokenizer, and
  whether every possible answer begins with a different token.
- Record the working prompt verbatim, the accuracy over a reasonable sample, and
  which variations failed.

## What systematic errors does it make?

Accuracy hides error structure. A model that fails randomly behaves differently
from one that fails whenever two operands have the same value.

- **Group failures by every input variable.** A failure rate that swings with one
  variable usually means the algorithm is right for part of the input space.
- **Look at the wrong answers, not just the rate.** Answering with the other
  operand is a specific, findable behavior; answering with the most frequent value
  is a prior.
- **Check whether position affects accuracy.** If the same content produces
  different accuracy in different positions, that constrains where the mechanism
  can be.
- **Off-format outputs are a prompt problem.** Go fix the prompt.
- Record a table of examples with the concrete input, actual output, expected
  output, and type of error. A reader cannot check aggregate rates alone.

## For multi-token generations, what kind of output is it producing?

Most causal methods score one token at one position. For longer outputs, first
define a consistent score.

- **Look for a single decisive token first** — the first token of the answer, a
  yes/no, a chosen letter. If the rest of the generation follows from it, score it
  and the rest of the protocol gets easy.
- **Otherwise, name the output modes** and the function that assigns a generation
  to one. Any later metric is built on that function.
- **Check the decisive content sits at a stable position.** If it moves, positional
  interventions smear across it — find an anchor or narrow the task until one
  exists, and say in the roadmap that you did.

## Typical units of work

- One per prompt-format candidate evaluated.
- One per error-structure hypothesis checked (by input variable, position, answer
  identity).
- One for the output characterization, if the behavior is multi-token.

## Handoff

You leave with a working prompt, measured accuracy, an examples table, a summary of
the errors, and a scoring rule. If you
will build a causalab task from this behavior, check it against the five task
quality objectives now rather than after the task is written —
[`../../implementation/setup-task/instructions/task_quality_objectives.md`](../../implementation/setup-task/instructions/task_quality_objectives.md).

Update `ROADMAP.md`, then go to
[`../exploratory-experimentation/exploratory-experimentation.md`](../exploratory-experimentation/exploratory-experimentation.md).

> **Execution: stub.** Batch greedy decoding over a prompt list ran through the
> Hydra `exploration` analysis in `probe` mode, which the protocol refactor
> retired. A protocol document can read `lm_head` at the final position and use a
> `match` metric to compare it with the answer column. However, causalab does not
> yet define a command that simply generates and displays the answers. Fill in this
> section once causalab stabilizes.
