"""Load a project-local ``.env`` into the process environment.

The experiment runner calls :func:`load_project_dotenv` at entry so a ``.env``
at the repository root is picked up automatically — no manual ``export``, no
shell ``source``, and no hand-parsing it with ``sed``. This is the single
canonical place credentials such as ``OPENROUTER_API_KEY`` (the
subspace-characterization LLM judge) enter the process.

Resolution order, first hit wins:

1. ``.env`` found by walking up from the current working directory — honours
   the directory the runner was launched from / sbatch's submit dir, and a git
   worktree nested under the main checkout that owns the (untracked) ``.env``.
2. ``.env`` found by walking up from this module's location in the source tree
   — a stable fallback when cwd is elsewhere.

Values already present in the environment are never overwritten
(``override=False``), so an explicit ``export`` or sbatch ``--export`` always
wins over the file.
"""

from __future__ import annotations

import logging

from dotenv import find_dotenv, load_dotenv

logger = logging.getLogger(__name__)

__all__ = ["load_project_dotenv"]


def load_project_dotenv() -> str | None:
    """Load the project ``.env`` if one is found; return its path or ``None``.

    Idempotent and safe to call when no ``.env`` exists (e.g. in CI) — it logs
    and returns ``None`` rather than raising. Never overrides an already-set
    variable.
    """
    dotenv_path = find_dotenv(usecwd=True) or find_dotenv()
    if not dotenv_path:
        logger.debug("No .env found; relying on the ambient environment.")
        return None
    load_dotenv(dotenv_path, override=False)
    logger.info(
        "Loaded environment from %s (existing vars take precedence)", dotenv_path
    )
    return dotenv_path
