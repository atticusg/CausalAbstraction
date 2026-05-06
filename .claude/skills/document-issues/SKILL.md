---
name: document-issues
description: Proactively create/update issues.md to document failures, confusions, oddities, and workarounds. Use this skill automatically whenever you encounter problems during work.
---

# Document Issues Skill

**Purpose:** Improve the agent environment by documenting everything that goes wrong or seems weird. This data helps make the environment clearer for future agents.

## Required Reading

Before running this skill, read:
- `.claude/skills/research-session/CONVENTIONS.md` — research-session layout and active-session protocol.

If no active session exists (`agent_logs/.current` missing or stale), instruct the user to run `/research-session` first and stop.

## When to Use This Skill (Auto-Suggestion)

Use this skill proactively whenever you encounter:

1. **Failures** - Errors, exceptions, commands that didn't work, even trivial ones
2. **Confusions** - Unclear instructions, misleading templates, ambiguous documentation
3. **Oddities** - Unexpected results, weird outputs, things that don't make sense
4. **Workarounds** - Things you had to figure out that should have been explained

**Important mindset:** We're designing an agent environment, not assigning blame. If you made a mistake, that's valuable data - it means the environment should be clearer.

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

## File Location

Issues for the active session land in `${SESSION_DIR}/issues.md` (top level — see `.claude/skills/research-session/CONVENTIONS.md`). The path is resolved via `agent_logs/.current`.

`${SESSION_DIR}` is shared across `setup-task`, `plan-experiment`, `setup-methods`, `setup-analyses`, `run-experiment`, `interpret-experiment`, and `replicate-paper` within one research session, so all issues for the session land in one place. Example: `agent_logs/2026-05-03--locate-weekdays--pensive-mongoose/issues.md`.

## Format

Use a simple markdown list. Each issue should include:

```markdown
# Issues

- **[Category]** Brief description of what happened
  - Context: What you were trying to do
  - What went wrong or was confusing
  - How you worked around it (if applicable)
```

### Categories

- `[ERROR]` - Actual errors, exceptions, failed commands
- `[CONFUSION]` - Unclear documentation, ambiguous instructions
- `[UNEXPECTED]` - Results that don't match expectations, weird behavior
- `[WORKAROUND]` - Solutions you discovered that should be documented elsewhere
- `[SUGGESTION]` - Ideas for improvement based on experience

## Example issues.md

```markdown
# Issues

- **[ERROR]** Import failed for `causalab.tasks.ioi`
  - Context: Trying to run experiment after creating task files
  - Got `ModuleNotFoundError: No module named 'causalab.tasks.ioi'`
  - The directory was named `IOI` (uppercase) but import uses lowercase

- **[CONFUSION]** Unclear which template file to use for token positions
  - Context: Creating token_positions.py for new task
  - Instructions mention `.claude/skills/setup-task/templates/token_positions.py` but also reference patterns in existing tasks
  - Ended up copying from existing task which worked better

- **[UNEXPECTED]** Model accuracy showed 0% on valid counterfactuals
  - Context: Running residual stream experiment on two_digit_addition
  - Expected ~95% accuracy based on similar tasks
  - Not sure if this is a bug or expected behavior for this model

- **[WORKAROUND]** Had to manually add `__init__.py` to get imports working
  - Context: Task files were created but wouldn't import
  - The setup-task skill doesn't always create __init__.py
  - Should be documented or automated

- **[SUGGESTION]** Error messages from checker.py are cryptic
  - Context: Debugging why counterfactuals were marked invalid
  - The error just says "Invalid" with no details
  - Would help to include which validation failed
```

## Workflow

1. **As you work:** When you hit any problem, make a mental note
2. **Before finishing:** Create or update `issues.md` with all issues encountered
3. **Be thorough:** Include even small things - patterns emerge from details
4. **Don't filter:** Document it even if you're not sure it's a "real" issue

## Key Principles

- **Document your mistakes** - If you misunderstood something, that's environment feedback
- **Include confusion** - Even if you eventually figured it out, the confusion itself is data
- **Note weird results** - Even without explanation, documenting oddities helps
- **Err on the side of documenting** - More data is better; filtering can happen later
