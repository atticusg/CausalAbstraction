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

## Decompose multi-token outputs

Most causal methods explain one next-token prediction at one location. If the
behavior generates several meaningful tokens, complete
[`output-decomposition.md`](output-decomposition.md) before exploration.

For a standardized sequence, such as several numbers or fixed semantic fields,
reuse one parent behavioral setup and create a child investigation for each
meaningful output token or semantic slot. For free-form text, do not branch on
every literal token. Define a smaller set of scientific subquestions that each
ends at one next-token prediction and has a semantic target that can be aligned
across examples.

The primary analysis conditions on the correct preceding output tokens. Treat
those tokens as explicit input variables for the current target. Follow-up
investigations may replace the correct prefix with the model's own generated
prefix to test how earlier errors change the computation.

## Typical units of work

- One per prompt-format candidate evaluated.
- One per error-structure hypothesis checked (by input variable, position, answer
  identity).
- One for output decomposition, if the behavior is multi-token.

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
report, a summary of the errors, a scoring rule, and `OUTPUT_TARGETS.md` when the
generation contains several meaningful tokens. If you
will build a causalab task from this behavior, check it against the five task
quality objectives now rather than after the task is written —
[`../../implementation/setup-task/instructions/task_quality_objectives.md`](../../implementation/setup-task/instructions/task_quality_objectives.md).

Update `ROADMAP.md`, then go to
[`../exploratory-experimentation/exploratory-experimentation.md`](../exploratory-experimentation/exploratory-experimentation.md).

## Execution

There is no "generate and display" command, and none is needed: a
**no-intervention protocol document** does both halves of this step. One
document per prompt format, `--set` for nothing that matters, one table per
format.

**The score.** Read `lm_head` at the last prompt position and grade it with
`match` against the answer column:

```json
"sites":   {"lm_head": {"component": "lm_head"}},
"reads":   {"logits": {"site": "lm_head", "pos": -1, "model": "original", "input": "base"}},
"metrics": {
  "acc":  {"kind": "match",  "of": "logits", "expected": "base_answer_forms",
           "mode": "first_token", "token_form": "space_prefixed"},
  "top5": {"kind": "top_k",  "of": "logits", "k": 5, "by": "prob"}
}
```

- `token_form` is **required** and is a real choice. `space_prefixed` is right
  when the answer follows a space in the prompt; pin `bare` when the prompt
  already ends in a space, or the answer is punctuation. Getting it wrong reads
  a flat 0.000 at every layer with no error — it is the single most common way
  this step reports a false negative.
- `mode: "first_token"` is what grades a multi-token answer; `expected` may name
  an answer-**forms** column, so a task's equivalent surface forms are task
  data, not a document-side string transform.
- `top_k` with `by: "prob"` is the "inspect the outputs" half. It is what tells
  you the model's top token is `"\n"` rather than an answer at all — which is
  not hypothetical: on Qwen3.6-35B-A3B the shipped `MCQA` prompt and the shipped
  `natural_domains_arithmetic` integer prompt each scored `base_acc` **0.000**,
  for two independent reasons, and only the top-k view says which.

**The generations.** When the behavior is more than one token, decode and read
the text back with a continuation-frame read
(`causalab/configs/protocols/probe_variable.json` is the worked shape, minus its
steer):

```json
"positions": {"continuation": {"generated": {"max_new_tokens": 8}, "all": true}},
"reads":     {"steps": {"site": "lm_head", "pos": "continuation",
                        "model": "original", "input": "base"}},
"metrics":   {"said": {"kind": "decode", "of": "steps"},
              "per_step": {"kind": "top_k", "of": "steps", "k": 1, "by": "prob"}}
```

`decode` reduces ids the greedy decode already produced, so the text costs no
extra vocabulary projection, and a row that generated nothing yields a null
value with `matched: false` rather than failing the run.

**The error browser** is then a script step over the saved tables: join `acc`
with the row's `input`, `base_answer` and the `top5` entry, and keep every row
where `acc` is 0.

> **Step 1 is load-bearing, not ceremonial.** Both measured failures above are
> failures of the *prompt*, found in an hour of CPU-scale work, and each would
> otherwise have been discovered as an unexplainable null result several GPU
> days into exploration. Do not let a later phase start before a prompt clears
> the gate.
