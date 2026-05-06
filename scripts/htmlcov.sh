#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
uv run pytest --cov=causalab --cov-report=term-missing --cov-report=html:htmlcov --durations=0 "$@"
uv run python tools/coverage_module_index.py htmlcov
