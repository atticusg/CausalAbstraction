# pipelines

Research protocols that run **on** causalab, as opposed to the specs in
[`../docs/`](../docs/) that describe what causalab *is*.

Open here if you have a question about a model's internals and want to know how to
go about answering it. Open `../docs/` if you need the normative format of an
intervention protocol or a workflow document.

## Start here

**[`answer-research-question/`](answer-research-question/answer-research-question.md)**
— the main protocol. Six steps, from establishing the behavior through to a saved,
scoped claim:

1. [Behavioral analysis](answer-research-question/behavioral-analysis/behavioral-analysis.md)
2. [Exploratory experimentation](answer-research-question/exploratory-experimentation/exploratory-experimentation.md)
3. [Hypothesis generation](answer-research-question/hypothesis-generation/hypothesis-generation.md)
4. [Hypothesis testing](answer-research-question/hypothesis-testing/hypothesis-testing.md)
5. [Generalize results](answer-research-question/generalize-results/generalize-results.md)
6. [Save results](answer-research-question/save-results/save-results.md)

The outer document owns the flow chart, the roadmap you write before running
anything, and the routing rules that decide where you go when a test comes back
negative. Read it first.

## Reference trees

Entered from within the steps, not steps themselves:

- [`implementation/`](implementation/) — working against the causalab codebase:
  building a task package, methods and analyses, running experiments, the analysis
  catalog, common failures, and the localization report format.
- [`explore-subspace/`](explore-subspace/) — verifying the description of a
  subspace you were handed, before designing causal experiments against it.
- [`subspace-causal-analysis-pipeline/`](subspace-causal-analysis-pipeline/) — the
  older end-to-end pipeline for adjudicating whether a *given* subspace is causal.

**All three predate the protocol refactor and are substantially stale.** Each
carries a banner saying so. They are kept because the science in them is still
correct and there is no replacement; the invocations in them are not.

## Execution stubs

The protocol refactor (PR #20) retired the Hydra runner, the `analyses/` tree,
`methods/` as Python, and SLURM dispatch. The `answer-research-question/` documents
were rewritten around that: they keep the science and mark each place where the
concrete invocation used to be with

> **Execution: stub.** …

Current stubs, and what each is waiting on:

| Document | Waiting on |
|---|---|
| `behavioral-analysis/behavioral-analysis.md` | A batch greedy-decode / "just run these prompts" shape on the document layer. |
| `exploratory-experimentation/methods/logit-lens.md` | No logit-lens primitive. Plausibly a residual-stream swap into the final block plus a `top_k` metric, unwritten and unvalidated. |
| `exploratory-experimentation/methods/probing.md` | No first-class probing. The closed metric vocabulary is over `lm_head` reads, so a probe's own loss is not expressible. |
| `exploratory-experimentation/methods/ablation.md` | The mechanism exists; the swept knockout document, the corpus-mean harvest, and band expansion do not. |
| `exploratory-experimentation/methods/steering.md` | The mechanism exists; the harvest-and-difference step producing the direction, and coherence tracking, do not. |
| `exploratory-experimentation/methods/attribution.md` | No attribution support at all — needs new protocol vocabulary, not just a document. |
| `exploratory-experimentation/methods/feature-labels.md` | The `sae` featurizer exists; nothing on the labeling side, and the handoff format is unspecified. |
| `hypothesis-generation/hypothesis-generation.md` | `can_distinguish_with_dataset` and `find_live_paths` still exist as library calls; the `develop_hypothesis` harness around them does not. |
| `save-results/save-results.md` | How an investigation's results are persisted and promoted is not settled. |

`hypothesis-testing/` has no stub: it runs on the shipped method presets in
`causalab/configs/methods/` and the worked workflow in
`causalab/configs/workflows/weekdays_8b.json`.
