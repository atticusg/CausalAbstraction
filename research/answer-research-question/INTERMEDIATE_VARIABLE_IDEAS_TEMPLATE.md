# Ideas for intermediate variables: {research question}

Create this file as `$WORKDIR/INTERMEDIATE_VARIABLE_IDEAS.md` before exploration.
It is a live list of possible internal variables, including ideas proposed before
there is evidence for them. An entry is a research lead, not a result.

## Task and output targets

- **Controlled task:** {how examples are generated and which parts may vary}
- **Output target:** {the one token or semantic next-token subquestion examined by
  this investigation}
- **Prefix condition:** {correct preceding output tokens for the primary analysis}
- **Relevant input variables:** {tokens, spans, and preceding output tokens}

## Ideas

| Variable | Plain definition | Possible values | Causal role | Output target | Possible layers, components, and token locations | Why plausible | Variables it could be confused with | PCA labels to inspect | Symbols or tokens for logit lens | Status | Evidence and next test |
|---|---|---|---|---|---|---|---|---|---|---|---|
| {variable} | {one quantity or state} | {…} | {parents, transformation, children} | {…} | {…} | {behavioral or algorithmic reason} | {input, output, or intermediate alternatives} | {…} | {…} | untested | {…} |

Use these statuses:

- `untested` — a plausible idea without internal evidence;
- `supported by exploration` — PCA, logit lens, or interventions motivate a
  formal hypothesis, but do not establish the variable;
- `testing` — its datasets or six hypothesis tests are in progress;
- `supported` — hypothesis testing distinguishes it from its plausible
  alternatives on held-out data;
- `rejected` — a valid test contradicted it;
- `needs revision` — the variable or its counterfactual datasets were not defined
  sharply enough to test.

Do not delete rejected ideas. Add evidence and change the status while preserving
the earlier rationale. When an idea informs PCA or logit lens, record the exact
labels, symbols, token locations, and report cells used to inspect it.
