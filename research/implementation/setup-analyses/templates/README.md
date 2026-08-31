# Workflow script template

`experiment.json` measures an expected-answer logit. `workflow.json` runs that
protocol and then invokes the repository-local `summarize.py` script, which writes
a one-row summary table.

Copy both files into a research directory, update the protocol path and expected
columns, then run `causalab validate` and `causalab explain` before execution.

`token_form` is **required** on every metric that names a string answer, and
`"space_prefixed"` here is a choice, not boilerplate: it is right because the
answer follows a space in the prompt. Change it to `"bare"` whenever the prompt
already ends in a space, or the answer is punctuation — the space-prefixed form
would then name a token the model never emits.
