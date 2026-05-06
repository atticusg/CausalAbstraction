---
name: replicate-paper
description: Replicate results from research papers. Extracts methodology and the paper's experiment list, then orchestrates the standard research pipeline (/plan-experiment → /run-experiment → /interpret-experiment) with paper-specific comparison and extension prompts threaded through. Use when the user wants to reproduce or extend published results.
args: <paper_path_or_url_or_description>
---

# Replicate Paper Skill

Thin orchestrator for replicating results from research papers. Elicits *replication-specific* context (paper, experiment list, paper-reported numbers for comparison, extension suggestions) and hands off to the standard pipeline:

```
/replicate-paper  ──►  /plan-experiment  ──►  /run-experiment  ──►  /interpret-experiment
   (interactive)        (interactive)          (autonomous)         (autonomous)
```

Approval gates concentrate in this skill's planning phase. Once `/plan-experiment` begins, it owns interactivity. From `/run-experiment` onward the pipeline is autonomous.

## Required Reading

Before running this skill, read:
- `.claude/skills/research-session/CONVENTIONS.md` — research-session layout and active-session protocol.

If no active session exists (`agent_logs/.current` missing or stale), instruct the user to run `/research-session` first and stop. Suggested topic naming: `agent_logs/{YYYY-MM-DD}--replicate-{paper}--{adj-noun}/`.

## Step 0a: Resolve the Active Session

```bash
if [ ! -f agent_logs/.current ]; then
    echo "No active research session. Run /research-session first." >&2
    exit 1
fi
SESSION_NAME="$(cat agent_logs/.current)"
SESSION_DIR="agent_logs/${SESSION_NAME}"
if [ ! -d "${SESSION_DIR}" ]; then
    echo "Active session marker points at missing directory ${SESSION_DIR}. Run /research-session." >&2
    exit 1
fi
```

In autonomous mode (`CAUSALAB_AUTONOMOUS=1`), the harness pre-creates the session and writes `.current` before invoking the skill. The skill assumes `${SESSION_DIR}` is valid and proceeds.

## Step 0b: One Session, One Plan Gate

Refuse to overwrite or sibling-add to an existing plan. See `.claude/skills/research-session/CONVENTIONS.md` §One session, one plan.

```bash
PLAN_FILES=()
[ -f "${SESSION_DIR}/plan/RESEARCH_OBJECTIVE.md" ] && PLAN_FILES+=("RESEARCH_OBJECTIVE.md")
[ -f "${SESSION_DIR}/plan/PLAN.md" ]               && PLAN_FILES+=("PLAN.md")
[ -f "${SESSION_DIR}/plan/paper_context.md" ]      && PLAN_FILES+=("paper_context.md")
```

If `PLAN_FILES` is non-empty:

**Interactive mode** — surface the conflict and stop:

> "Session `${SESSION_NAME}` already contains a plan: {list `PLAN_FILES`}.
> One session, one plan (see `.claude/skills/research-session/CONVENTIONS.md`).
>
> Options:
>   1. **Switch sessions** — run `/research-session` to start a fresh session,
>      then re-invoke `/replicate-paper` there.
>   2. **Hold off** — keep the existing plan, exit without changes.
>
> Which do you want?"

On choice 1: print "Run `/research-session` then re-invoke `/replicate-paper`." and stop.
On choice 2 (or any non-1 response): print "Keeping existing plan." and stop.

Do not delete or modify the existing plan files under any circumstance.

**Autonomous mode** (`CAUSALAB_AUTONOMOUS=1`) — exit non-zero:

```
Error: session ${SESSION_NAME} already contains a plan ({list PLAN_FILES}).
The harness must allocate a fresh session per plan. Aborting.
```

If `PLAN_FILES` is empty, proceed.

## Lifecycle map

| Artifact | Location | Written by |
|---|---|---|
| Downloaded paper sources (PDFs, datasets) | `${SESSION_DIR}/paper_data/` | this skill |
| Free-form objective draft for `/plan-experiment` | `${SESSION_DIR}/plan/objective_input_draft.md` | this skill |
| Paper-specific comparison + extension data (sidecar) | `${SESSION_DIR}/plan/paper_context.md` | this skill |
| Approval log | `${SESSION_DIR}/plan/approval.log` | this skill (autonomous and interactive) |
| `RESEARCH_OBJECTIVE.md`, `PLAN.md` | `${SESSION_DIR}/plan/` | `/plan-experiment` |
| Resolved configs, run logs | `${SESSION_DIR}/run/` | `/run-experiment` |
| Raw experiment outputs | `${SESSION_DIR}/artifacts/{task}/{model}/...` | `/run-experiment` |
| Final report (with paper-comparison section) | `${SESSION_DIR}/result/REPORT.md` | `/interpret-experiment` |
| Issues encountered | `${SESSION_DIR}/issues.md` | `/document-issues` |

Paper sources are session-scoped. Each replication session re-fetches its own copy. Before re-fetching, it is reasonable to check whether a recent `agent_logs/*--replicate-{paper}--*` session has it locally and copy from there.

## Autonomous vs Interactive Mode

```bash
echo $CAUSALAB_AUTONOMOUS
```

- **`"1"`**: extract methodology, default to "all experiments selected", write `objective_input_draft.md` and `paper_context.md` without prompting, log decisions to `${SESSION_DIR}/plan/approval.log`, hand off.
- **Empty / unset**: walk through Steps 2–4 interactively with the user.

## Issue Tracking

Use `/document-issues` for any failures, methodology mismatches, or surprises encountered while extracting the paper. Issues land in `${SESSION_DIR}/issues.md`.

---

## Step 1: Get Paper

Accept the paper via:
1. **PDF path** — read the PDF and extract methodology.
2. **URL** — fetch and parse the paper.
3. **User description** — the user describes the paper's key claims and methods.

### Find the paper's code/data repository

Search for the paper's GitHub repository or data release:
1. Check the PDF for links to code repositories, supplementary materials, or dataset URLs.
2. Search GitHub using the paper title, author names, or keywords (e.g., `gh search repos "{paper_title}"` or `WebSearch`).
3. Check arXiv — papers often link to code in the abstract or "Code" tab.
4. Check PapersWithCode for associated repositories.

If a repository is found, present it to the user (interactive only):
> "I found the paper's code repository at {url}. It contains {description — data, code, configs, etc.}. Should I use their data/code as the basis for our replication?"

In autonomous mode, default to "yes, use the paper's repo if available". Log the decision.

Clone or download relevant data files to `${SESSION_DIR}/paper_data/`. Note the path — it will be referenced in `paper_context.md` so `/plan-experiment` §B can wire the dataset.

---

## Step 2: Extract Methodology and Enumerate Experiments

From the paper, extract two structures.

### 2a. Methodology table

| Field | Description |
|-------|-------------|
| **Citation** | Authors, year, title, link |
| **Task** | What task does the paper study? (e.g., indirect object identification, addition) |
| **Model(s)** | Which models? (e.g., GPT-2 small, Pythia-1B) |
| **Causal model** | What variables and relationships does the paper assume? |
| **Intervention method** | What technique? (activation patching, DAS, causal scrubbing, etc.) |
| **Metrics** | How does the paper measure success? (accuracy, KL divergence, logit diff, etc.) |
| **Dataset** | How many examples? How were they generated? |
| **Data availability** | Is the dataset publicly available? (GitHub repo, HuggingFace, supplementary materials) |

### 2b. Experiment enumeration

List **every distinct experimental result** reported in the paper, indexed for selection in Step 3:

| # | Experiment | Section/Figure | Models | Metric | Reported finding |
|---|------------|----------------|--------|--------|------------------|
| 1 | {short label} | §X.Y, Fig N | {model(s)} | {metric} | {one-line headline} |
| 2 | ... | ... | ... | ... | ... |

Each row is one quantitative claim — typically one figure, one table, or one result block in the paper.

### Approval checkpoint

**Interactive:**
> "Here's the methodology + experiment list extracted from the paper:
> {methodology_table}
> {experiment_table}
>
> Is this accurate? Anything missing?"

Iterate until the user accepts.

**Autonomous:** skip the prompt. Append to `${SESSION_DIR}/plan/approval.log`:
```
[step-2] methodology + experiment list extracted, no approval gate (autonomous)
```

---

## Step 3: Select Experiments to Replicate

**Interactive:** show the numbered experiment list from Step 2b and ask:
> "Which experiments should we replicate? You can answer with:
> - `all` — replicate every experiment in the list
> - a comma-separated list of indices — e.g. `1,3,5`
> - a free-form filter — e.g. `only the patching experiments`
>
> Which do you want?"

Resolve the answer to a concrete list of indices. Echo the resolved list back for confirmation.

**Autonomous:** default to "all". Append to `${SESSION_DIR}/plan/approval.log`:
```
[step-3] selected experiments: all (autonomous default)
```

The selection is the input to Step 4.

---

## Step 4: Draft `objective_input_draft.md` and `paper_context.md`

Compose two artifacts under `${SESSION_DIR}/plan/`.

### 4a. `objective_input_draft.md` — feeds `/plan-experiment` Step 1

This is the free-form objective blob `/plan-experiment` will consume. Structure:

```markdown
# Objective

Replicate {selected experiments by short label} from {paper citation}.

# Motivation

{2–4 sentences. Why this paper? Why these experiments? What changes downstream
if findings (don't) reproduce? Phrase from the user's perspective — why they
asked for this replication. Do not let causalab's available analyses shape this
section (per `/plan-experiment` motivation rule).}

# Scope boundaries

- Paper experiments **not** being replicated this session: {list other rows}
- Out-of-scope models / variables: {anything the user excluded}

# Success criteria (recommended)

{Paper-reported findings as concrete, measurable thresholds. Be quantitative
where the paper is. Examples:
- "Locate finds the same (layer, position) cell as the paper's Fig 2 within ±1 layer."
- "Subspace k=4 reconstructs ≥ 80% of the patching effect, matching paper's Tbl 2."}

# Hypotheses (recommended)

{Paper's qualitative claims as falsifiable predictions. Each paired with what
would falsify it. Examples:
- "**H1.** Same circuit emerges (L9H9 dominant). *Falsified if* L9H9 ranks
   below the median across heads in layer 9."}

# Paper data location

`${SESSION_DIR}/paper_data/` contains {what was downloaded — PDF, dataset files,
repo clone}. The paper's dataset is at `{path within paper_data}`. Use this as
the dataset source for `/plan-experiment` §B (task spec) when applicable.
```

`/plan-experiment`'s motivation rule expects user-stated motivation. In replication, the user's "motivation" is "I want to know if this paper's claim X reproduces under our tools" — phrase it that way. Do **not** infer motivation from causalab's tool inventory.

### 4b. `paper_context.md` — sidecar consumed by `/interpret-experiment`

```markdown
# Paper Context — {citation}

## Citation

{authors, year, title, link to PDF/arXiv}

## Selected experiments

| # | Experiment | Paper section | Reported finding |
|---|------------|---------------|------------------|
| {only the rows the user selected in Step 3} |

## Reported findings (for comparison)

For each selected experiment, list the specific quantitative claims that should
be checked side-by-side against our replication.

| Component / variable | Paper's metric | Paper's value | Source (§ / Fig / Tbl) |
|---|---|---|---|
| L9H9 | logit-diff drop | 0.45 | §4.1, Fig 2 |
| ... | ... | ... | ... |

## Comparison expectations

- For each row above, `/interpret-experiment` should produce a side-by-side
  row in REPORT.md §7: paper's value vs replication's measured value, with the
  artifact path cited for the latter.
- Discrepancy categories to consider when annotating mismatches:
  **methodology** (different intervention method / metric),
  **model** (different checkpoint / tokenizer),
  **dataset** (different templates / filtering),
  **genuine** (our tools reveal something different).

## Suggested extensions

*(carried into REPORT.md "Suggested next steps" by `/interpret-experiment`,
prefixed with "From the paper:")*

- {extension 1: e.g., run the same experiments on a model the paper didn't test}
- {extension 2: e.g., paper used patching → try DAS for distributed representations}
- {extension 3: e.g., broaden scope from one variable to all causal variables}
- ...
```

### Approval checkpoint

**Interactive:**
> "Here's the objective draft + paper context sidecar.
> {objective_input_draft.md content}
> {paper_context.md content}
>
> Approve to hand off to `/plan-experiment`, edit, or cancel?"

Iterate until the user accepts. Log to `${SESSION_DIR}/plan/approval.log`.

**Autonomous:** write both files and append to `approval.log`:
```
[step-4] objective draft + paper_context written, no approval gate (autonomous)
```

---

## Step 5: Hand Off to `/plan-experiment`

Invoke `/plan-experiment` with the draft path as its argument:

```
Skill(plan-experiment, args="${SESSION_DIR}/plan/objective_input_draft.md")
```

`/plan-experiment` will:
- Copy the draft to its canonical `${SESSION_DIR}/plan/objective_input.md` (its Step 1 — provenance capture).
- Walk the user through `RESEARCH_OBJECTIVE.md` finalization (its Step 2 — interactive).
- Characterize the task into PLAN.md §B — covers what the old `/replicate-paper` Step 4 ("Check for Existing Task") used to do.
- Design the analysis DAG into PLAN.md §D — covers what the old Step 3 ("Map to Causalab") used to do.
- Choose sweep / cache strategy into PLAN.md §F — covers what the old Step 5 ("Design Replication Plan") used to do.
- Block on its own approval gates.

This skill **waits** for `/plan-experiment` to return. Do not invoke `/run-experiment` until the user has approved the plan.

---

## Step 6: Hand Off to `/run-experiment`

After `/plan-experiment` returns successfully (RESEARCH_OBJECTIVE.md and PLAN.md present, approval.log shows approval), invoke `/run-experiment`:

```
Skill(run-experiment)
```

`/run-experiment` will autonomously:
- Optionally invoke `/setup-task`, `/setup-methods`, `/setup-analyses` per PLAN.md §B and §D markers (the latter two accept all custom-node specs in one batch each).
- Materialize the runner config(s) listed in PLAN.md §F under `${SESSION_DIR}/code/configs/runners/`.
- Execute (covers what the old Step 6 "Execute" used to do).
- Auto-invoke `/interpret-experiment` at its Step 9.

`/replicate-paper` does **not** prompt during this phase. The pipeline is fully autonomous from this point.

---

## Step 7: Final Hand-Off

Once `/run-experiment` returns, print:

```
Replication pipeline complete.

Session:    ${SESSION_DIR}
Plan:       ${SESSION_DIR}/plan/{RESEARCH_OBJECTIVE.md, PLAN.md, paper_context.md}
Artifacts:  ${SESSION_DIR}/artifacts/
Report:     ${SESSION_DIR}/result/REPORT.md

The report includes a "Paper comparison" section (§7) populated from
paper_context.md, with side-by-side paper vs replication values, and
"Suggested next steps" prefixed with extensions from the paper.
```

This skill writes no result file of its own — the consolidated REPORT.md from `/interpret-experiment` is the canonical output.

---

## Important Notes

### What this skill does NOT do

- **Does not write `RESEARCH_OBJECTIVE.md` or `PLAN.md`.** Those are owned by `/plan-experiment`.
- **Does not write `replication_plan.md`.** That artifact is obsolete.
- **Does not run experiments, write runner configs, or call `/setup-task` directly.** Those are owned by `/run-experiment`.
- **Does not write the final report.** That is owned by `/interpret-experiment`, which reads `paper_context.md` opportunistically and renders REPORT.md §7 (paper comparison) and the augmented "Suggested next steps".
- **Does not prompt during Steps 5–7.** Approval gates only exist in Steps 2 and 4 (interactive mode). After hand-off, `/plan-experiment` brings its own interactive gates; `/run-experiment` and `/interpret-experiment` are autonomous.

### Restrictions

- Do not modify core library files.
- Do not skip Step 2b (experiment enumeration). The user's selection in Step 3 depends on it.
- Do not skip Step 4b (`paper_context.md`). Without it, `/interpret-experiment` cannot produce the paper-comparison section.
- Do not duplicate writes to `${SESSION_DIR}/plan/RESEARCH_OBJECTIVE.md` or `PLAN.md`. If you need to amend the objective after `/plan-experiment` has run, route the user to a fresh session — see CONVENTIONS.md §"One session, one plan".
- Confine writes to `${SESSION_DIR}/paper_data/`, `${SESSION_DIR}/plan/objective_input_draft.md`, `${SESSION_DIR}/plan/paper_context.md`, and `${SESSION_DIR}/plan/approval.log`.
