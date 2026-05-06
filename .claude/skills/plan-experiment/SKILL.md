---
name: plan-experiment
description: Crystallize a research objective into a detailed experiment plan. Produces RESEARCH_OBJECTIVE.md (objective, hypotheses, success criteria) and PLAN.md (task spec, analysis DAG, sweep strategy, expected artifacts) in the active session's plan/ dir. Use this BEFORE /run-experiment whenever a new investigation starts. Input is either a free-form objective markdown blob, a path to such a file, or nothing (interactive elicitation).
args: <optional_path_to_objective_markdown>
---

# Plan Experiment Skill

The first phase of the research-session workflow (plan → run → interpret). Turns a fuzzy research idea into two artifacts that downstream skills consume:

- `${SESSION_DIR}/plan/RESEARCH_OBJECTIVE.md` — objective, motivation, scope, success criteria, hypotheses.
- `${SESSION_DIR}/plan/PLAN.md` — task spec, neural surface, analysis DAG with per-node IO contracts and pre-flight gates, sweep & cache strategy, expected artifact tree.

`/run-experiment` reads both, materializes runner config(s), and executes. `/interpret-experiment` reads the artifacts plus this plan as the interpretation lens.

## Required Reading

Before running this skill, read:

1. `.claude/skills/research-session/CONVENTIONS.md` — research-session layout and active-session protocol.
2. `.claude/skills/plan-experiment/ANALYSIS_GUIDE.md` — analysis dependency DAG, per-analysis decision matrix, auto-discovery rules.
3. `ARCHITECTURE.md` §2 (config system), §3 invariants 5/7/12 (Hydra-only defaults, `experiment_root` routing, dataset-param placement), §5 (config-knob interactions).

If no active session exists (`agent_logs/.current` missing or stale), instruct the user to run `/research-session` first and stop.

## Autonomous vs Interactive Mode

```bash
echo $CAUSALAB_AUTONOMOUS
```

- **`"1"`**: read the seed file referenced in the workflow, fill all fields without prompting, write both files, log decisions to `${SESSION_DIR}/plan/approval.log`. The five non-autonomous task-spec decisions (output_token_mode, single-token filtering, low-accuracy fallback, space-variant tokenization, custom CF distributions — see `instructions/characterize_task.md`) are surfaced *for visibility*, not gated on user input; defaults are taken and logged.
- **Empty / unset**: walk through the steps interactively, getting user approval at the explicit checkpoints below.

## Issue Tracking

Use `/document-issues` whenever you hit a blocker or surprise. Issues land in `${SESSION_DIR}/issues.md`.

---

## Step 0: Resolve the Active Session

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

In autonomous mode, the harness pre-creates the session and `.current` before invoking. Trust them.

---

## Step 0.5: One Session, One Plan Gate

Refuse to overwrite or sibling-add to an existing plan. See `.claude/skills/research-session/CONVENTIONS.md` §One session, one plan.

```bash
PLAN_FILES=()
[ -f "${SESSION_DIR}/plan/RESEARCH_OBJECTIVE.md" ] && PLAN_FILES+=("RESEARCH_OBJECTIVE.md")
[ -f "${SESSION_DIR}/plan/PLAN.md" ]               && PLAN_FILES+=("PLAN.md")
```

`plan/paper_context.md` is **not** in `PLAN_FILES`: it's a replication sidecar written by `/replicate-paper` immediately before this skill is invoked, and its presence is the expected entry state when `/plan-experiment` is reached via the paper-replication chain. The gate scopes to the real plan artifacts only.

If `PLAN_FILES` is non-empty:

**Interactive mode** — surface the conflict and stop:

> "Session `${SESSION_NAME}` already contains a plan: {list `PLAN_FILES`}.
> One session, one plan (see `.claude/skills/research-session/CONVENTIONS.md`).
>
> Options:
>   1. **Switch sessions** — run `/research-session` to start a fresh session,
>      then re-invoke `/plan-experiment` there.
>   2. **Hold off** — keep the existing plan, exit without changes.
>
> Which do you want?"

On choice 1: print "Run `/research-session` then re-invoke `/plan-experiment`." and stop.
On choice 2 (or any non-1 response): print "Keeping existing plan." and stop.

Do not delete or modify the existing plan files under any circumstance.

**Autonomous mode** (`CAUSALAB_AUTONOMOUS=1`) — exit non-zero:

```
Error: session ${SESSION_NAME} already contains a plan ({list PLAN_FILES}).
The harness must allocate a fresh session per plan. Aborting.
```

The harness owns session allocation in autonomous mode; the skill refuses to guess.

If `PLAN_FILES` is empty, proceed to Step 1.

---

## Step 1: Read Input

The skill accepts three input shapes:

1. **Argument is a path to a markdown file** — read it as the user's free-form objective draft.
2. **Argument is a markdown blob** (multi-line text) — read it directly.
3. **No argument** — interactive elicitation: ask the user for a one-paragraph statement of what they want to learn.

In all cases, capture the raw input verbatim at `${SESSION_DIR}/plan/objective_input.md` for provenance.

---

## Step 2: Crystallize the Objective → `plan/RESEARCH_OBJECTIVE.md`

> **Tool-agnostic step.** Steps 1–2 stay close to the user's *original intent*. Do not let causalab's existing analyses/methods (`causalab/analyses/`, `causalab/methods/`) shape the objective or motivation — they are a library of pre-implemented tools, not the research question. Tool selection happens in Step 4 only. Phrase objective and motivation in scientific/empirical terms; avoid analysis names (`locate`, `subspace`, `path_steering`, …) here.

Read `RESEARCH_OBJECTIVE_TEMPLATE.md`. Fill:

- Session pointer
- Objective (one sentence — convert "what we'll do" into "what we want to know"; reflect the user's wording)
- Motivation (2–4 sentences derived from the user input — see motivation rule below)
- Scope boundaries (what is *not* in scope this session)

### Motivation rule

The motivation must come from the user, not from the toolset.

- **If the user input clearly states the motivation** (a paper, prior result, hunch, downstream consequence): summarize it in 2–4 sentences using the user's framing.
- **If the motivation is underspecified, vague, or absent**: in interactive mode, ask the user explicitly — e.g. *"Why does this matter now? What earlier result, paper, or downstream decision is driving the question?"* Do not infer a motivation from the available analyses. In autonomous mode, write `(motivation underspecified — user did not state)` and log the gap to `${SESSION_DIR}/plan/approval.log`; do not fabricate.

Then **recommend** (do not invent):

- **Success criteria** — concrete enough that the conclusion paragraph could be written today, leaving only numbers blank.
- **Hypotheses** — concrete, falsifiable predictions, each paired with what would falsify it.

If the user opts out of either recommendation, write `(not specified)` — never fabricate.

#### 🔒 APPROVAL CHECKPOINT (interactive only)

> "Here is the crystallized research objective. Approve, edit, or cancel?"

Show the file, accept edits, repeat until approved. Log the final approval to `${SESSION_DIR}/plan/approval.log`.

In autonomous mode: write and proceed. Log "objective written, no approval gate (autonomous)".

---

## Step 3: Characterize the Task → `plan/PLAN.md` §B

Follow `instructions/characterize_task.md` to fill §B of `PLAN.md`. Two paths:

- **Task package exists** at `causalab/tasks/{task_name}/`: read `causal_models.py`, `counterfactuals.py`, `token_positions.py`, `config.py`, plus `set_up_task.md` if present. Summarize into §B.
- **Task package does not exist**: fill §B as a `/setup-task`-compatible spec covering all 8 fields the survey identified (variables/value spaces, mechanisms, templates, counterfactuals, token positions, checker/metrics, output_token_mode, MAX_*_TOKENS). Flag the five non-autonomous decisions inline so the user (or autonomous harness) approves them now, before runs start.

§B is the only section of `PLAN.md` that `/run-experiment` may pass verbatim to `/setup-task` if the task does not yet exist.

---

## Step 4: Design the Analysis DAG → `plan/PLAN.md` §D

Follow `instructions/design_dag.md`. Walk the dependency DAG from `ANALYSIS_GUIDE.md` and choose nodes that answer the research questions implied by the objective.

For each chosen node, fill the per-node card defined in `PLAN_TEMPLATE.md` §D:

- scoped research question
- upstream artifact files (with auto-discovery notes)
- downstream artifact files (paths under `${experiment_root}/...`)
- non-default knobs vs `causalab/configs/analysis/<name>.yaml`
- **pre-flight check** — the minimal finding required before any dependent node runs. These act as gates, not predictions.
- runtime + GPU footprint estimate

Method-level granularity is adaptive: surface methods only when method choice **is** the research question (e.g. "compare DAS vs DBM on `day`"). Otherwise stay at the analysis level.

Also draw the DAG diagram and any cross-analysis post-steps.

---

## Step 5: Sweep & Cache Strategy → `plan/PLAN.md` §F

Follow `instructions/plan_sweep.md`. Decide:

- **Single config**: §F lists one runner-config name. No `sweep_id`. Skip the sweep subsection.
- **Sweep (≥ 2 configs sharing this plan)**: declare a `sweep_id` slug. All members share `experiment_root = ${SESSION_DIR}/artifacts/{task}/{model}/{sweep_id}`. Document expected cache reuse per node and explicitly verify no two members write to the same `_subdir`.

The sweep instructions enforce the safety check from the cache-behavior survey: configs varying along axes already encoded in `_subdir` (k_features, smoothness, method) coexist cleanly; configs varying along axes not in `_subdir` (most importantly `target_variables` for `locate`) **must not** share an `experiment_root` and the skill emits a warning.

---

## Step 6: Compose `plan/PLAN.md` and Review Checkpoint

Render the full `PLAN.md` from `PLAN_TEMPLATE.md` with §B–F filled. Then surface to the user:

1. **Hypotheses + success criteria** (from `RESEARCH_OBJECTIVE.md`) — does the user agree these are the right tests?
2. **Sweep strategy** (§F) — does the user accept the cache-sharing plan, or do they want isolated `experiment_root`s?
3. **Compute estimate** (§C) — does the wall-time fit the user's window?

#### 🔒 APPROVAL CHECKPOINT (interactive only)

> "Here is the full plan. Approve to hand off to `/run-experiment`, edit, or cancel?"

Repeat until approved. Log to `${SESSION_DIR}/plan/approval.log`.

In autonomous mode: write the file, log "plan written, no approval gate (autonomous)", and proceed.

---

## Step 7: Hand Off

Print:

```
Plan complete. Two files written:
  - ${SESSION_DIR}/plan/RESEARCH_OBJECTIVE.md
  - ${SESSION_DIR}/plan/PLAN.md

Next: /run-experiment
```

`/run-experiment` reads both files at its Step 1 (after task validation in Step 2). It will materialize the runner config(s) listed in §F and execute.

---

## Important Notes

### What this skill does NOT do

- **Does not predict experiment results.** The plan declares pre-flight checks (gates) but never sketches expected heatmaps, expected manifold shapes, or expected scores. Predictions live in `RESEARCH_OBJECTIVE.md` hypotheses (user-stated) only.
- **Does not write the runner-config YAML.** YAML materialization is `/run-experiment` Step 3 — the plan only names the configs and lists per-node knobs.
- **Does not invoke `/setup-task`.** `/run-experiment` invokes `/setup-task` later, using §B as the spec input. The plan only ensures §B is rich enough for that to be autonomous.
- **Does not invent hypotheses or success criteria.** If the user does not state them, write `(not specified)`.
- **Does not let the toolset frame the objective.** Steps 1–2 are tool-agnostic. Do not infer or rephrase objective/motivation from `causalab/analyses/` or `causalab/methods/`. If motivation is unclear, ask the user (interactive) or mark it `(motivation underspecified)` (autonomous) — never reverse-engineer it from the available analyses.
- **Does not implement missing methods or analyses.** If a §D node references an analysis missing from `causalab/analyses/`, or a method missing from `causalab/methods/`, mark the node `custom` and record what is missing. After plan approval and before `/run-experiment`, run `/setup-methods` and/or `/setup-analyses` — those skills scaffold the missing pieces under `${SESSION_DIR}/code/` (the runner picks them up via session-code path injection) and accept multiple specs at once. The plan does not gate on this; implementation happens between plan approval and the first run.

### Restrictions

- Do not write Python or YAML — only markdown into `${SESSION_DIR}/plan/`.
- Do not edit files outside `${SESSION_DIR}/plan/` and `${SESSION_DIR}/issues.md`.
- Do not skip the review checkpoint in interactive mode. The checkpoint is the design's single guarantee that the user has confronted hypotheses + sweep risk before compute is spent.
- One session, one plan: refuse to write if the session already contains a plan-level artifact (see Step 0.5 and `.claude/skills/research-session/CONVENTIONS.md` §One session, one plan). Never delete or overwrite an existing plan; route the user to a fresh session instead.
