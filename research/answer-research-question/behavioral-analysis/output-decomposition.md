# Decompose a multi-token output

Complete this step after selecting the prompt and before identifying critical
tokens. The purpose is to turn a longer generation into scientific questions that
each examine one next-token prediction.

The pipeline does not require the original prompt to look rigidly templated. It
does require each child investigation to have a target that can be located,
scored, and changed through understandable counterfactual inputs. A natural prompt
may therefore be placed inside a controlled example generator that records its
variables and token locations.

## Choose the kind of decomposition

### Standardized outputs

Use one child investigation per meaningful token or semantic slot when the output
has a stable interpretation. Examples include each number in a sequence, each
label in a structured answer, or each field in a fixed-format sentence.

Do not assume that one explanation covers every output position. A sequence of
five numbers creates five targets unless two positions genuinely answer the same
causal question and that equivalence is itself tested.

### Free-form outputs

Do not create a child investigation for every token in a story, poem, instruction
set, or ordinary sentence. Instead, define subquestions with one next-token target
each. A useful subquestion has:

- a semantic role that is meaningful across several examples;
- a rule that locates the target token in each example;
- a scoring rule for the target prediction;
- input variables that could causally affect it; and
- nearby or narrowly designed counterfactual inputs that preserve the rest of the
  context well enough to interpret the change.

Examples may align by semantic role rather than absolute token index. Record both
the role and the resolved index for every example. If a semantic role spans
several tokenizer tokens, either choose one prediction within it or create
separate child investigations for its meaningful token predictions.

If no stable role and valid counterfactual can be defined, narrow the research
question. Do not use a broad generation-level score as a substitute for a
next-token target.

## Prefix conditions

Use the correct preceding output tokens for the primary child investigation. This
isolates the computation for the current target from mistakes made earlier in the
generation. Treat every preceding output token that could affect the target as an
explicit input variable.

After the controlled investigation, create a follow-up condition using the
model's own generated prefix. Compare it with the correct-prefix condition to
measure error propagation and determine whether the same mechanism still applies.
Do not mix the two prefix conditions in one aggregate score.

## Shared parent and child investigations

The parent behavioral setup owns the selected prompt, model revision, tokenizer,
example generator, and general scoring code. Create child directories only when
the output targets answer meaningfully different causal questions:

```text
$WORKDIR/
├── OUTPUT_TARGETS.md
└── output-targets/
    ├── {semantic-target-a}/
    │   └── ROADMAP.md
    └── {semantic-target-b}/
        └── ROADMAP.md
```

Each child roadmap references the shared parent artifacts and records its target,
prefix condition, input variables, token-location rule, counterfactual design,
exploration, hypotheses, tests, and generalization. Do not copy and silently
modify the parent model or tokenizer setup.

## Required artifact

Write `$WORKDIR/OUTPUT_TARGETS.md` with one row per child investigation:

| Semantic target | Example next token | Position rule | Prefix condition | Preceding output tokens treated as inputs | Scoring rule | Counterfactual plan | Child roadmap | Status |
|---|---|---|---|---|---|---|---|---|
| {…} | {…} | {role and resolved index} | correct prefix | {…} | {…} | {nearby and narrow edits} | {path} | ready |

Use `proposed`, `ready`, `active`, `complete`, or `rejected` as the status. A child
is ready only when representative examples resolve to the intended target, its
prefix is explicit, and at least one interpretable counterfactual plan exists.

## Completion gate

Output decomposition is complete when every meaningful standardized output slot
has a child investigation, or when every selected free-form subquestion has a
stable semantic target. Verify the target and prefix tokens with the actual
tokenizer on representative examples. Exploration remains blocked until this
artifact is complete.
