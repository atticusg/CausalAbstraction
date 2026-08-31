# Running report plan: {research question}

Create this file as `$WORKDIR/REPORT_PLAN.md` before the first experiment. Update
it throughout behavioral analysis, exploration, hypothesis generation, testing,
and generalization. It is the plan for the final handoff, not the handoff itself.

## Current concise story

Write the shortest causal account currently supported. State what happens to the
relevant inputs, which intermediate variables have evidence, how attention and
MLP layers contribute, and how the output forms. Mark every unsupported link as a
gap.

## Candidate main claims

| Claim | Scope | Best supporting experiment | Best figure | Important alternative | Missing evidence | Generalization needed | Status |
|---|---|---|---|---|---|---|---|
| {plain claim} | {output target, prompt, model, data} | {experiment identity and artifact} | {figure path or planned view} | {…} | {…} | {prompt, related task, or natural text} | candidate |

Use `candidate`, `main claim`, `appendix only`, `rejected`, or `unresolved` as the
status. Generalize only the main claims, or the subset for which a particular
generalization experiment is meaningful.

## Main report plan

The main report should contain only the experiments and figures needed to explain
the final causal story. Keep both counts at 20 or fewer, and prefer a smaller set
when it carries the same evidence.

| Order | Claim or transition | Experiment | Figure | What the reader should conclude | Ready? |
|---|---|---|---|---|---|
| {…} | {…} | {experiment identity} | {figure path} | {one conclusion} | no |

## Appendix plan

Index every completed experiment, including null results, failed controls, and
superseded tests. Give a full appendix figure to a result only when it adds
distinct evidence. Represent repetitive sweeps with a table and artifact link.

| Experiment | Output target | Variable | Method | Result | Figure or table | Why main or appendix | Artifact |
|---|---|---|---|---|---|---|---|
| {canonical experiment identity} | {…} | {…} | {…} | {positive, null, failed control, superseded} | {…} | appendix | {path} |

## Final handoff checklist

- `result/report.tex` contains the concise causal story and no more than 20 main
  figures.
- `result/report.pdf` compiles from `report.tex` without manual edits.
- `result/figures/` contains every main and appendix figure referenced by LaTeX.
- `result/experiment-index.json` identifies every experiment and its artifacts.
- The appendix indexes every completed experiment and may contain many more
  figures, including roughly 100 when the investigation warrants them.
- Every main claim links to sharp intervention evidence and states its scope.
- Generalization results appear before the final conclusions.
- Unresolved layers and failed generalizations remain visible.
