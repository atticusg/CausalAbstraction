# Demos

One markdown file per research question, with the documents that answer it. The
format — sections, header table, voice, checklist — is
[`docs/demos.md`](../docs/demos.md).

These replace the notebooks the protocol refactor retired. A notebook had to
carry its own execution; a document does not, so what is left is the experiment.

## Onboarding

Three demos on one multiple-choice task, in order. Each takes something concrete
from the one before it.

| | demo | question | needs a GPU |
|---|---|---|---|
| 01 | [Define the task](onboarding_tutorial/01_define.md) | does this counterfactual dataset tell two variables apart? | no |
| 02 | [Trace one pair](onboarding_tutorial/02_trace.md) | for one pair, which cells carry the answer symbol? | yes, a small one |
| 03 | [Localize the variable](onboarding_tutorial/03_localize.md) | across 64 pairs, which cell carries it most reliably? | yes, a small one |

**Start with 01.** It runs on a laptop in under a second and it is the demo that
decides whether the other two measure anything.

## Worked research

| demo | question | needs |
|---|---|---|
| [Weekdays geometry](weekdays_geometry/weekdays_geometry.md) | where and how is the answer day represented, and what does the model say between two answers? | one GPU ≥40 GB |

A **workflow demo**: four research questions as ten steps, each RQ's answer
feeding the next one's document.

## Running any of them

Every demo's "Run it" section holds the three commands, with real output pasted
in. The first two are pure — no weights, no network, no accelerator — so the
shape of a run is checkable before it is booked:

```bash
uv run causalab validate <doc> --data-root <demo>/data --data
uv run causalab explain  <doc> --data-root <demo>/data
```

Each demo carries its own `data/`, with the `<ref>.manifest.json` sidecar that
records how the table was built.

## Adding one

Read [`docs/demos.md`](../docs/demos.md), then copy the closest existing demo's
skeleton. `tests/demos/test_demos.py` checks the mechanical half of the format:
that every document validates, every link resolves, and every demo has the seven
sections in order.
