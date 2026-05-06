---
name: development-session
description: Load development-mode context for codebase work — engineering principles and environment conventions. Invoke at the start of any codebase task (bug fixes, refactors, dependency bumps, skill edits, doc/typo fixes, tooling/infra changes). Do NOT invoke for research workflows — use /research-session instead.
---

# Development Session

Loads engineering principles and environment conventions that apply to all codebase work in this repo. These are concise reminders, not rules — use judgment.

For codebase architecture, layering rules, and invariants, see [`ARCHITECTURE.md`](../../../ARCHITECTURE.md).

## Principles

**Question the approach, not just the implementation.** Before improving a cache, ask if caching is the right solution. Before handling an edge case, ask if that case can actually occur. Before adding backwards compatibility, check if the old version is still used. Before fixing code to match a test, consider whether the test should change instead.

**Understand root causes.** When something breaks, dig into WHY — what led to this bug, why didn't the type checker catch it, is this a symptom of a deeper design problem?

**Simplify where possible.** Remove flags by detecting conditions automatically. Split functions that do too much. Use libraries instead of reimplementing.

**Be pragmatic.** Not everything needs to be fixed immediately. If something is a "hint for the future" rather than a required change, say so.

**Think about implications.** What do the results mean? How does this approach compare to alternatives? What conclusions should we draw?

**Check the abstraction level.** Is there too much functionality in one place? Are we reimplementing something a library already does?

## Environment

**Always use the `uv` virtual environment.** Run Python via `uv run` by default (e.g. `uv run pytest`, `uv run python -c '...'`). The repo's lockfile pins versions; ad-hoc system Python skips that.
