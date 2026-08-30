# Step 1 — Behavioral analysis

Establish how the model behaves before looking inside it. Otherwise, you may study
a shortcut instead of the intended behavior.

## Can the model solve the task?

First find a prompt under which the model actually exhibits the behavior. Do not
launch any later phase until this is established and the behavioral analysis is
complete.

1. Write several plausible prompt formats, usually about ten. Vary wording,
   formatting, answer position, and other small choices that could affect whether
   the model understands the task.
2. Evaluate the candidates on comparable examples. These evaluations may run in
   parallel as separate small GPU jobs within the same experiment. No work from a
   later phase may run alongside them.
3. Inspect the generated outputs as well as the aggregate scores. Confirm that a
   high score reflects the intended behavior rather than a formatting artifact,
   memorized pattern, or shortcut.
4. Select one prompt and report its performance. Performance must be well above
   random chance and should normally be about 90% or higher. If performance is
   lower, do not proceed unless the failures reveal a stable, systematic behavior
   that becomes the explicit subject of the research question.
5. If no prompt produces either reliable task performance or a clear systematic
   behavior, stop and revise the task or research question.

- Decide whether to include examples in the prompt **before** you start. These
  examples may let the model match a pattern instead of using the algorithm under
  study. Prefer a prompt with no examples. If you include examples, limit every
  later claim to that prompt condition.
- Vary phrasing, format, and token choices. What usually matters: whether the
  answer follows a space, whether it is a single token under this tokenizer, and
  whether every possible answer begins with a different token.
- Record every candidate prompt verbatim, its evaluation result, which prompt was
  selected, and the selected prompt's accuracy over a reasonable sample.

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
- Preserve every error example with the concrete input, actual output, and
  expected output. Group the errors and report the short list of recurring
  patterns. A reader cannot check aggregate rates alone.

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

## Final report

Write `result/behavioral-analysis.html` as a minimal, self-contained interactive
report. Use [`REPORT_TEMPLATE.html`](REPORT_TEMPLATE.html) as its structure. The
report must contain only the following:

1. **Prompt examples.** Put this first. Provide one tab for every prompt format
   that was evaluated. Each tab must show the exact prompt text and four or five
   complete examples, including the exact input sent to the model and the model's
   output.
2. **Prompt results.** Provide one simple table whose rows correspond to the same
   prompt tabs. Report the number of evaluated examples and the success rate for
   each prompt.
3. **Selected prompt.** State only which prompt was selected.
4. **Error examples.** Provide a minimal browser for moving through every error.
   Each entry must show the exact input, expected output, and actual model output.
5. **Error patterns.** End with a short bullet list of the recurring patterns in
   the selected prompt's errors.

Use direct technical language. Do not add a narrative summary, implementation
log, causal interpretation, unrelated plots, or additional sections.

## Handoff

You leave with a selected prompt, measured performance, the interactive behavioral
report, a summary of the errors, and a scoring rule. If you
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
