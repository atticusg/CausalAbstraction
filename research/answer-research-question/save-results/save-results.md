# Step 6 — Save results

Do this once, at the end of the research session, after the selected main claims
have completed their applicable generalization experiments. The model should
synthesize the investigation into a concise final handoff. Do not produce the
final report after the first successful hypothesis test.

Read `ROADMAP.md`, `INTERMEDIATE_VARIABLE_IDEAS.md`, `REPORT_PLAN.md`, every phase
report, and the experiment artifacts. Produce:

```text
$WORKDIR/result/
├── report.md
├── report.html
├── experiment-index.json
└── figures/
```

`report.md` is the source of truth and `report.html` is the self-contained
rendering of it — one file, figures inlined or referenced under `figures/`, no
external assets. Copy every referenced figure into `figures/`. The
machine-readable index must identify every experiment by its canonical identity,
method, output target, variable, status, report, run artifacts, and figure
paths. Use the Silico experiment `eid` when one exists and the CausaLab document
or point digest for CausaLab runs.

> **Why not LaTeX.** This contract used to require `report.tex` plus a
> `report.pdf` that "must compile from it without manual edits". **There is no
> LaTeX toolchain on the cluster** — no `pdflatex`, `xelatex`, `latexmk`,
> `tectonic` or `lualatex`, on login or compute nodes — and the cluster is the
> only environment these runs execute in, so the contract required a deliverable
> the executor could not produce. `report.md` + `report.html` +
> `experiment-index.json` satisfies it. If a venue needs LaTeX, convert at that
> point, on a machine that has it; do not make a machine-checkable contract
> depend on a toolchain the machine lacks.

## Main report

Tell the shortest supported layer-by-layer causal story. Explain how the relevant
inputs enter, what attention and MLP layers contribute, which intermediate
variables have sharp evidence, how the current next-token output forms, and how
the main claims generalize.

Use only the experiments that materially determine what the reader should
believe. Include no more than 20 experiments and no more than 20 figures in the
main report. Ten or fewer is preferable when it tells the same story. Each figure
must support one explicit claim and link back to its experiment artifacts.

Include these sections:

1. The controlled task, output target, and nearby counterfactual strategy.
2. The concise layer-by-layer causal account.
3. The small set of decisive hypothesis tests.
4. Generalization of the main claims.
5. Boundaries, unresolved layers, and failed generalizations.

## Appendix

Index every completed experiment, including null results, failed controls,
rejected hypotheses, and superseded tests. The appendix may contain many more
figures, including roughly 100 when the investigation warrants them. Give a full
figure only to results that add distinct evidence; summarize repetitive sweeps in
tables with artifact references.

The appendix must make it possible to audit how the concise story was selected.
It must not turn speculative ideas or failed controls into supporting evidence.
