# Roadmap: {research question, one line}

_Copy this into `$WORKDIR/ROADMAP.md` before running anything. Update it at the
end of every step. See [`answer-research-question.md`](answer-research-question.md)._

## The question

- **Question:** {the question in a form that has an answer — not a topic}
- **Model:** {model key and revision}
- **Task / behavior:** {what the model is being asked to do}
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
exploratory experimentation     launch these three experiments in parallel:
        ├── logit lens
        ├── PCA
        └── counterfactual patching
        │
        │ all three must finish
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

- **Logit lens:** {what it would show if the initial guess were right}
- **PCA:** {what it would show if the initial guess were right}
- **Counterfactual patching:** {what it would show if the initial guess were right}

### 3. Hypothesis generation

- **Expect:** {the shape of the causal model you anticipate}
- **Units:** {…}

### 4. Hypothesis testing

- **Expect:** {the test, and what result would count as strong positive}
- **Units:** {one per contested alternative, roughly}

### 5. Generalize results

- **Expect:** {the widest claim you think will survive}
- **Units:** {…}

### 6. Save results

- **Expect:** {what the deliverable is and who reads it}

## Revision log

Append an entry after every step. Do not edit earlier entries. The log should
preserve the difference between what you expected and what happened.

### {date} — after {step}

- **Produced:** {what the step actually returned}
- **Differs from plan:** {how, or "as expected"}
- **Plan changes:** {what in the blocks above is now wrong, and what replaces it}
- **Routing:** {only after hypothesis testing — which door was taken and why}
