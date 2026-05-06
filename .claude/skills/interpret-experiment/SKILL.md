---
name: interpret-experiment
description: Auto-invoked after /run-experiment finishes. Reads the active session's plan and produced artifacts, then writes a single consolidated result/REPORT.md grounded in the planned objective, success criteria, and hypotheses. Inspects shipped *and* session-local analyses (under ${SESSION_DIR}/code/). Runs autonomously without approval gates. Triggers: "interpret experiment", "interpret results", "write report", "analyze results", "explain what the experiment found".
---

# Interpret Experiment Skill

Reads the active session's plan and produced artifacts and writes one consolidated `result/REPORT.md`. Adapts to whatever was planned in `/plan-experiment` and whatever ran in `/run-experiment` — including session-local methods and analyses scaffolded by `/setup-methods` and `/setup-analyses`. Cross-session comparison happens only when the plan declares it.

This skill is the closing step of the research workflow. It is invoked automatically at the end of `/run-experiment` and runs autonomously — no approval checkpoints.

## Required Reading

Before running this skill, read:
- `.claude/skills/research-session/CONVENTIONS.md` — research-session layout and active-session protocol.

If no active session exists (`agent_logs/.current` missing or stale), instruct the user to run `/research-session` first and stop.

## Step 0: Resolve the active session

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

In autonomous mode (`CAUSALAB_AUTONOMOUS=1`) the harness pre-creates the session and writes `.current`. The skill assumes `${SESSION_DIR}` is valid and proceeds. Output lands at `${SESSION_DIR}/result/REPORT.md` with embedded figures under `${SESSION_DIR}/result/figures/`. Issues go to `${SESSION_DIR}/issues.md`.

## Step 1: Discover what was planned

Read the plan-level artifacts. These are the source of truth for what the session set out to learn.

1. **`${SESSION_DIR}/plan/RESEARCH_OBJECTIVE.md`** — extract:
   - Objective (one sentence)
   - Motivation
   - Scope boundaries
   - Success criteria (may be `(not specified)`)
   - Hypotheses (may be `(not specified)`)

2. **`${SESSION_DIR}/plan/PLAN.md`** — extract:
   - §B causal model & dataset
   - §C neural surface (models, compute budget)
   - §D analysis-chain DAG with per-node research questions and pre-flight checks
   - §E risk register & contingency
   - §F sweep strategy and expected artifact tree

If either file is missing or contains `(not specified)` placeholders, continue but record the gap in §8 Caveats of the report.

3. **Cross-session reuse detection** — scan `RESEARCH_OBJECTIVE.md` and `PLAN.md` for explicit reuse declarations (per `CONVENTIONS.md` § "Cross-session reuse"). Examples: "reuse activations from session X", "use the global baseline", or an `experiment_root` pointing into another `agent_logs/{other_session}/` path. For each referenced session, locate its `result/REPORT.md` and `artifacts/` subtree — these become the right-hand side of the cross-session comparison in Step 5.

4. **Paper context detection (replication sessions only)** — check for `${SESSION_DIR}/plan/paper_context.md`. If present, parse:
   - The "Reported findings (for comparison)" table → list of `(component, paper_metric, paper_value, source)` tuples. These become the rows of REPORT.md §7 "Paper comparison".
   - The "Suggested extensions" bullets → list of strings to append to REPORT.md "Suggested next steps", prefixed with "From the paper:".
   - The citation, for inclusion in the §7 header.
   Flag the session as a replication session for use in Step 7. If the file is absent, the session is not a replication and §7 renders as `*(not a replication session)*`.

## Step 2: Discover what actually ran

The plan describes intent; the resolved config describes reality. Read the latter.

1. **`${SESSION_DIR}/run/<runner>_resolved.yaml`** — the canonical record. Enumerate:
   - Which analyses ran (the `defaults` list and per-analysis sub-blocks)
   - Each analysis's resolved `_output_dir`
   - Task parameters (`task.*`), model config, seed, sweep axes
   - Any non-default knobs

2. **Session-local code inventory**:
   - List `${SESSION_DIR}/code/methods/` — for each method, read `set_up_method.md` (§1 purpose, §2 surface) to understand what it computes.
   - List `${SESSION_DIR}/code/analyses/` — for each analysis, read `README.md` (§ Outputs → Interpretation, Saved artifacts table) to understand its output schema.
   - List `${SESSION_DIR}/code/configs/analysis/` and `${SESSION_DIR}/code/configs/runners/` to confirm the wiring.

3. **Diff plan vs. reality**: compare the analyses listed in PLAN.md §D and the artifact tree from §F against what `<runner>_resolved.yaml` declares and what actually exists under `${SESSION_DIR}/artifacts/`. Note unplanned runs and missing planned runs.

## Step 3: Inventory artifacts

Walk `${SESSION_DIR}/artifacts/{task}/{model}/[{sweep_id}/]` and catalog every analysis subdirectory's outputs. Discovery is dynamic — do **not** rely on a hard-coded list of analyses.

For each analysis subdirectory present:

- **Shipped analysis** (name appears under `causalab/analyses/<name>/`): read `causalab/analyses/<name>/README.md`, locate the "Saved artifacts" table, use it as the schema for the on-disk outputs. Load every JSON / metadata file the table calls out.
- **Session-local analysis** (name appears under `${SESSION_DIR}/code/analyses/<name>/`): same procedure, using the session-local README.

Load structured numbers (JSON, metadata) into memory. Note image / HTML / tensor paths for later embedding. If the on-disk contents don't match the README's Saved artifacts table, record the discrepancy for §8 Caveats.

## Step 4: Score against success criteria & hypotheses

Match plan to evidence.

1. **Success criteria** — for each criterion in `RESEARCH_OBJECTIVE.md`: locate the bearing artifact(s) and record `met` / `unmet` / `inconclusive` with the supporting numbers. A criterion is `inconclusive` only when the artifacts that would settle it are missing or partial — never use it to dodge a verdict.

2. **Hypotheses** — for each hypothesis: cite the bearing artifact(s) and record `confirmed` / `falsified` / `inconclusive`. If the hypothesis specified a falsification condition (e.g., "score < 0.5 at every layer"), evaluate it explicitly and quote the numbers.

3. **Pre-flight gates** — for each PLAN.md §D node with a pre-flight check, record whether the gate was met. A failed gate upstream of a downstream node casts doubt on that node's results — flag this so the report doesn't over-interpret polluted output.

If the plan contains no success criteria or hypotheses, skip the per-criterion / per-hypothesis verdicts but still apply pre-flight checks. Note the gap in §8 Caveats.

## Step 5: Adaptive cross-run / cross-session comparison

Trigger on any of:
- The session ran a sweep (PLAN.md §F declares a `sweep_id`) — the artifact tree has multiple sibling subdirs under `{task}/{model}/{sweep_id}/`.
- PLAN.md / RESEARCH_OBJECTIVE.md declares cross-session reuse (Step 1.3).
- Multiple `{model}` or `{task}` subdirs exist directly under `${SESSION_DIR}/artifacts/`.

Otherwise skip §7 of the report.

When triggered:

1. **Axes come from the plan**, not from a fixed list. PLAN.md §F names the sweep axis (e.g., `subspace.k_features`, `model`, `task.variant`); use it. For cross-session comparison, the axis is "this session vs. session X".
2. **Statistics come from the plan too**. PLAN.md §D pre-flight gates and `RESEARCH_OBJECTIVE.md` hypotheses say what to look at — best_cell location, score deltas, KL deltas, accuracy deltas. Don't apply a generic comparison checklist; report only what the plan motivates.
3. **Significance** — if a success criterion gives a numeric threshold, use it. Otherwise report raw deltas and let the reader judge. Avoid hard-coded thresholds like "delta > 0.15 is significant".
4. **Cross-session lineage** — when comparing against another session, cite that session's `result/REPORT.md` so the provenance is legible.

## Step 6: Synthesize the interpretation

Anchor on the `Objective` line from `RESEARCH_OBJECTIVE.md`. Write a direct answer to it using the evidence collected in Steps 3–5.

Layer on circuit-style reasoning **only where the relevant artifacts exist and the plan motivated the question**. Skip questions whose required artifacts weren't run. Examples (apply à la carte, not as a checklist):
- Where does information enter? — ask only if `locate/` ran.
- Where is computation happening? — ask only if `locate/` shows a layer trend across cells.
- What is the representation geometry? — ask only if `subspace/` or `activation_manifold/` ran.
- Is the representation causally faithful? — ask only if `path_steering/` ran.
- What attention patterns support the circuit? — ask only if `attention_pattern/` ran.

If session-local methods or analyses ran, derive the question to ask from their `set_up_method.md` §1 / `README.md` "Research question" — those are the planned questions for those primitives.

## Step 7: Render the report

Open `REPORT_TEMPLATE.md` (next to this SKILL.md). Fill in each section. Write the result to `${SESSION_DIR}/result/REPORT.md`.

All template sections are required; if a section is N/A (e.g. no hypotheses given), write `(not specified)` verbatim — do not omit headers. This keeps reports uniform across sessions.

**Paper comparison (§7) — replication sessions only.** If Step 1.4 loaded `plan/paper_context.md`:

- Build the side-by-side table: one row per `(component, paper_metric, paper_value, source)` tuple from `paper_context.md`. Fill the "Our value" column with the matching number from `${SESSION_DIR}/artifacts/...`, citing the artifact path. Mark `Match?` as `yes` / `partial` / `no` and assign a discrepancy category (`methodology` / `model` / `dataset` / `genuine`) per the rubric in `paper_context.md`.
- Add 1–2 sentences of discrepancy annotation per non-`yes` row, naming the category.
- Append the "Suggested extensions" bullets from `paper_context.md` to §10 "Suggested next steps", prefixed with **`From the paper:`**, so they remain distinguishable from session-derived next steps.

If `paper_context.md` is absent, render §7 as `*(not a replication session)*` and proceed.

**Embedding figures**:
- Copy or symlink curated PDFs / PNGs from `artifacts/` into `${SESSION_DIR}/result/figures/`. Reference via relative markdown image links: `![caption](figures/locate_var_x.pdf)`.
- HTML interactive viz: link only, do not try to embed.
- Large tables: render the headline as a markdown table and save the full data as a CSV next to the figure; link the CSV.

**Citing claims**: every numeric claim in the report must cite the artifact path it came from. Mirror the rule from the prior version of this skill — interpretation without provenance is not useful.

## Step 8: Issue tracking

If anything went wrong during the run or this interpretation — missing artifacts, failed pre-flight gates, README/disk mismatches, surprise outputs — invoke the `document-issues` skill to append entries to `${SESSION_DIR}/issues.md`. Issues do **not** go in `REPORT.md`; they live in the session-level issue log per CONVENTIONS.

## Restrictions

- Do NOT modify artifact files or score files.
- Do NOT re-run analyses — this skill is for interpretation only.
- Do NOT modify shipped causalab library files.
- Do NOT write outside `${SESSION_DIR}/result/` and `${SESSION_DIR}/issues.md` (and `figures/` underneath `result/`).
- Do NOT prompt the user for approval — this skill runs autonomously.
