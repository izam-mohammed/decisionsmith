#!/usr/bin/env sh
# Run one integration's tests and examples in its own environment (frameworks never share one).
#   scripts/integration.sh langchain [venv-dir]
set -e
name="$1"
module=$(printf %s "$name" | tr - _)
venv="${2:-.venvs/$name}"
export UV_PROJECT_ENVIRONMENT="$venv"
uv sync --locked --no-default-groups --group "int-$name" --extra laya
DS_EXAMPLES="04-integrations/$name" uv run --no-sync pytest -o addopts="" -q -p no:warnings \
  "tests/integrations/test_$module.py" tests/test_examples.py --cov="decisionsmith.integrations.$module" --cov-branch --cov-report=json:"$venv/coverage.json"
uv run --no-sync python scripts/check_coverage.py --file "$venv/coverage.json" --only "integrations/$module.py"
