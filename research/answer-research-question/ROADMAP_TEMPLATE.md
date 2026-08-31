# Roadmap: {research question, one line}

_Copy this into `$WORKDIR/ROADMAP.md` before running anything. Update it at the
end of every step. See [`answer-research-question.md`](answer-research-question.md)._

## The question

- **Question:** {the question in a form that has an answer — not a topic}
- **Model:** {model key and revision}
- **Task / behavior:** {what the model is being asked to do}
- **Goal for intermediate variables:** {what meaningful internal variables this work
  may identify and what evidence would make one worth testing}
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
exploratory experimentation     launch five initial experiments in parallel:
        ├── logit lens
        ├── PCA
        ├── residual stream patching ──▶ residual stream DAS
        ├── attention output patching ─▶ attention head DBM
        └── MLP output patching ───────▶ MLP neuron DBM
        │
        │ each follow-up waits only for its parent patching experiment
        │ all initial and applicable follow-up experiments must finish
        ▼
hypothesis generation           independent units may run in parallel
        │
        ▼
hypothesis testing              independent tests may run in parallel
        │                       │
        │ strong result         └── revise and return to an earlier phase
        ▼
generalize results              independent checks may run in parallel
        │
        ▼
save results
```

### 1. Behavioral analysis

- **Expect:** {…}
- **Units:** {…}

### 2. Exploratory experimentation

- **Critical locations:** {tokens and spans that must be fixed before launch}
- **Initial experiments:** {logit lens, PCA, residual stream patching, attention
  output patching, and MLP output patching}
- **Gated follow-ups:** {which patching results would launch DAS or DBM}
- **Completion:** {all required reports and the updated table of candidate variables}

### 3. Hypothesis generation

- **Expect:** {the explicit intermediate variables and competing causal models
  suggested by exploration}
- **Units:** {…}

### 4. Hypothesis testing

- **Expect:** {the counterfactual dataset for each tested intermediate variable,
  and what result would count as strong evidence}
- **Units:** {one per contested alternative, roughly}

### 5. Generalize results

- **Expect:** {the widest claim you think will survive}
- **Units:** {…}

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

## Revision log

Append an entry after every step. Do not edit earlier entries. The log should
preserve the difference between what you expected and what happened.

### {date} — after {step}

- **Produced:** {what the step actually returned}
- **Differs from plan:** {how, or "as expected"}
- **Plan changes:** {what in the blocks above is now wrong, and what replaces it}
- **Routing:** {only after hypothesis testing — which door was taken and why}
