# Demos

One markdown file per research question, with the documents that answer it. The
format — sections, header table, voice, checklist — is
[`docs/demos.md`](../docs/demos.md).

These replace the notebooks the protocol refactor retired. A notebook had to
carry its own execution; a document does not, so what is left is the experiment.

## Onboarding

Three demos on one multiple-choice task, in order. Each takes something concrete
from the one before it.

| | demo | question | needs a GPU | measured |
|---|---|---|---|---|
| 01 | [Define the task](onboarding_tutorial/01_define.md) | does this counterfactual dataset tell two variables apart? | no | < 1 s, CPU |
| 02 | [Trace one pair](onboarding_tutorial/02_trace.md) | for one pair, which cells carry the answer symbol? | yes, a small one | 49 s, 1×H100 |
| 03 | [Localize the variable](onboarding_tutorial/03_localize.md) | across 64 pairs, which cell carries it most reliably? | yes, a small one | 39 s, 1×H100 |

**Start with 01.** It runs on a laptop in under a second and it is the demo that
decides whether the other two measure anything.

## Worked research

| demo | question | needs | measured |
|---|---|---|---|
| [Weekdays geometry](weekdays_geometry/weekdays_geometry.md) | where and how is the answer day represented, and what does the model say between two answers? | one GPU ≥40 GB | 132 s, 1×H100 |

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

The third command needs the accelerator. Note that `--dtype` applies to a
**protocol** document only — a workflow's steps each declare their own
realization, so `causalab run <workflow> --dtype bf16` is refused. Every demo
document already pins its dtype, so there is nothing to pass:

```bash
uv run causalab run <protocol> --data-root <demo>/data --out runs/<name> --device cuda --dtype bf16
uv run causalab run <workflow> --data-root <demo>/data --out runs           --device cuda
```

Each demo carries its own `data/`, with the `<ref>.manifest.json` sidecar that
records how the table was built.

## Adding one

Read [`docs/demos.md`](../docs/demos.md), then copy the closest existing demo's
skeleton. `tests/demos/test_demos.py` checks the mechanical half of the format:
that every document validates, every link resolves, every document is inlined in
its demo byte for byte, and every demo has the seven sections in order.

**Which document shape?** A **protocol** if the product is a number — one
document, one campaign, tables you read. A **workflow** if the product is a
figure, or a value a later step consumes: the figure script and the `select`
script both read the step record that only a workflow writes. The smallest
useful workflow is one protocol step plus one script step, which is a legitimate
shape rather than a workaround — 02 ships exactly that beside its protocol, and
both 02 and 03 inline the workflow next to the figure it produces
([02's](onboarding_tutorial/02_trace.md#results) ·
[03's](onboarding_tutorial/03_localize.md#q1--yes)), so the picture and its
generation are read together.
