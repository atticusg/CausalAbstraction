# Roadmap: {research question, one line}

_Copy this into `$WORKDIR/ROADMAP.md` before running anything. Update it at the
end of every step. See [`answer-research-question.md`](answer-research-question.md)._

_Also create `INTERMEDIATE_VARIABLE_IDEAS.md` and `REPORT_PLAN.md` from the
templates beside this file._

## The question

- **Question:** {the question in a form that has an answer — not a topic}
- **Model:** {model key and revision}
- **Task / behavior:** {what the model is being asked to do}
- **Controlled task:** {how a natural or templatic prompt becomes examples with
  named variables and auditable token or span locations}
- **Nearby counterfactual principle:** {which minimal, legible input changes will
  reveal causal structure and where broader changes are necessary}
- **Output targets:** {one token or semantic next-token subquestion per child
  investigation; include the prefix used for each target}
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

Behavioral analysis and exploration are strict gates. After exploration, track
dependencies per intermediate variable: hypothesis generation for many variables
may run in parallel, and testing for one variable may begin while generation
continues for another. Generalization waits for selection of the main claims, and
final synthesis waits for their generalization results.

```
behavioral analysis             required first; establish controlled task
        │
        │ reuse its selected prompt, model, code, and evaluation setup
        ▼
decompose output                create meaningful next-token child investigations
        │
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
hypothesis loop                 many variables proceed concurrently
        ├── generation A ──▶ six tests A ──┐
        ├── generation B ──▶ six tests B ──┤ revise and repeat
        └── generation C ──▶ six tests C ──┘
        │
        │ select main claims
        ▼
generalize main claims          prompt templates, related tasks, natural text
        │
        ▼
final LaTeX synthesis
```

### 1. Behavioral analysis

- **Expect:** {…}
- **Units:** {…}
- **Output decomposition:** {shared parent setup and meaningful child targets, or
  why the task has one output target}
- **Prefix policy:** {correct prefix for primary analysis; selected free-generation
  follow-ups}

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
- **Claims selected:** {subset of main claims for which these tests are meaningful}

### 6. Save results

- **Expect:** {`report.tex`, compiled `report.pdf`, figures, and a machine-readable
  index of every experiment}

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
