## Mode detection

At the first user message of a session, you MUST determine which session skill to invoke. Exactly one of `/research-session` or `/development-session` must be invoked before any other work — no substantive action happens until a session is active.

- **Research mode** (`/research-session`) — research-flavored work: running experiments, analyzing results, setting up tasks, replicating papers, exploring datasets, interpreting model internals.
- **Development mode** (`/development-session`) — codebase work: bug fixes, refactors, dependency bumps, editing skills, doc/typo fixes, infrastructure changes, tooling work.

Decision rule:

- **Intent is clear** → invoke the corresponding session skill directly and silently. Do not announce the classification.
- **Intent is ambiguous, unclear, or unspecified** → explicitly ask the user which mode applies. Do not guess, and do not proceed until the user answers.

## Pointers

- **Codebase architecture, layering rules, invariants** → [`ARCHITECTURE.md`](ARCHITECTURE.md) (root).
- **Artifact serialization** → see ARCHITECTURE.md "Artifact serialization policy".
- **Research-session conventions** (session dir layout, active-session protocol) → [`.claude/skills/research-session/CONVENTIONS.md`](.claude/skills/research-session/CONVENTIONS.md).
- **Per-skill behavior** → `.claude/skills/<name>/SKILL.md`.

## Skill routing

If the user's request maps to a skill, do NOT launch Explore agents or do manual codebase exploration. Instead, immediately ask to exit plan mode (via ExitPlanMode) so you can invoke the skill. Skills handle their own discovery, planning, and user approval — plan mode is redundant for skill-eligible requests. Only bypass a skill when the user explicitly asks for something outside the skill's scope or requests a manual approach.
