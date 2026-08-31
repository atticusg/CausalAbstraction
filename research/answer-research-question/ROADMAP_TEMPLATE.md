# Roadmap: {research question, one line}

_Copy this into `$WORKDIR/ROADMAP.md` before running anything. Update it at the
end of every step. See [`answer-research-question.md`](answer-research-question.md)._

## The question

- **Question:** {the question in a form that has an answer — not a topic}
- **Model:** {model key and revision}
- **Task / behavior:** {what the model is being asked to do}
- **Goal for intermediate variables:** {what meaningful internal variables this work
  may identify and what evidence would make one worth testing}
- **Layer-by-layer causal account:** {what must be explained about the input
  tokens, every attention and MLP layer, the intermediate variables, and output}
- **What changes once we know:** {what decision, follow-up, or claim this enables.
  If nothing, say so and reconsider the question.}
- **What would make us stop:** {the result that kills this line of work}

## Plan

Use one block for each step. Before the step begins, describe what you expect and
divide the work into the units that you will track.

The steps are blocking phases, not independent work streams. Complete behavioral
analysis before launching any work from a later phase. After that, launch work
only from the current phase. Independent units within that phase may run in
parallel, but the next phase remains blocked until the current phase is complete.

```
behavioral analysis             required first; blocks every later phase
        │
        │ reuse its selected prompt, model, code, and evaluation setup
        ▼
identify critical tokens        required before every exploratory experiment
        │
        ▼
exploratory experimentation     trace every input variable and the output:
        ├── logit lens
        ├── PCA
        └── six intervention experiments:
            ├── residual stream patching ──▶ residual stream DAS
            ├── attention output patching ─▶ attention head DBM
            └── MLP output patching ───────▶ MLP neuron DBM
        │
        │ each follow-up waits only for its parent patching experiment
        │ all initial and applicable follow-up experiments must finish
        ▼
hypothesis generation           one dataset-design experiment per variable
        │
        ▼
hypothesis testing              six experiments per intermediate variable
        │                       run concurrently when dependencies allow
        │                       │
        │ strong result         └── revise and return to an earlier phase
        ▼
generalize results              prompt templates, related tasks, wild text
        │
        ▼
save results
```

### 1. Behavioral analysis

- **Expect:** {…}
- **Units:** {…}

### 2. Exploratory experimentation

- **Critical locations:** {tokens and spans that must be fixed before launch}
- **Variables traced:** {every relevant input token or span, plus the output}
- **Initial experiments:** {logit lens, PCA, residual stream patching, attention
  output patching, and MLP output patching}
- **Gated follow-ups:** {which patching results would launch DAS or DBM}
- **Completion:** {all required reports and the updated table of candidate variables}

### 3. Hypothesis generation

- **Expect:** {the explicit intermediate variables and competing causal models
  suggested by exploration}
- **Units:** {one dataset-design experiment per proposed intermediate variable}
- **Output:** {one HTML report and machine-readable result per variable}

### 4. Hypothesis testing

- **Expect:** {how each tested intermediate variable appears across residual
  streams, attention, and MLPs, and what alternatives remain}
- **Units:** {six experiments per intermediate variable: three complete-output
  patching experiments, residual stream DAS, attention head DBM, and MLP neuron
  DBM}
- **Seeds:** {three recorded seeds for every DAS and DBM fit}

### 5. Generalize results

- **Expect:** {the widest layer-by-layer causal account that survives}
- **Units:** {new prompt templates, related tasks, and in-the-wild next-token
  prediction}

### 6. Save results

- **Expect:** {what the deliverable is and who reads it}

## Candidate intermediate variables

Start this ledger during exploratory experimentation. Keep competing candidates
separate and update it through hypothesis generation and testing.

| Candidate variable | Possible values | Possible neural locations | Evidence for | Evidence against | Distinguishing experiment | Status |
|---|---|---|---|---|---|---|
| {plain definition} | {…} | {layers, tokens, components} | {…} | {…} | {…} | active |

Use these statuses: `active`, `rejected`, `merged`, or `promoted to causal model`.
Do not delete rejected candidates; preserving them prevents the pipeline from
repeating the same unsupported guess.

## Layer-by-layer causal account

Start this table during exploration and update it after hypothesis testing and
every generalization experiment. Include every model layer. Leave unsupported
cells as `unresolved`; do not fill gaps with guesses.

| Layer | Relevant token positions | Residual stream | Attention contribution | MLP contribution | Variables supported here | Evidence | Unresolved questions |
|---|---|---|---|---|---|---|---|
| embedding | {…} | {…} | not applicable | not applicable | {input variables} | {artifact links} | {…} |
| 0 | {…} | {…} | {…} | {…} | {…} | {artifact links} | {…} |
| … | {…} | {…} | {…} | {…} | {…} | {artifact links} | {…} |
| output | {final position} | {…} | not applicable | not applicable | {output variable} | {artifact links} | {…} |

For every entry, distinguish direct intervention evidence from decodability or
correlation. Record where information moves between token positions as well as
where it exists.

## Revision log

Append an entry after every step. Do not edit earlier entries. The log should
preserve the difference between what you expected and what happened.

### {date} — after {step}

- **Produced:** {what the step actually returned}
- **Differs from plan:** {how, or "as expected"}
- **Plan changes:** {what in the blocks above is now wrong, and what replaces it}
- **Routing:** {only after hypothesis testing — which door was taken and why}
