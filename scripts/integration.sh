#!/usr/bin/env sh
# Run one integration's tests and examples in its own environment (frameworks never share one).
#   scripts/integration.sh langchain [venv-dir]
# The environment gets decisionsmith[laya] plus the pinned group `int-<name>` from tests/integrations/pyproject.toml.
set -e
name="$1"
module=$(printf %s "$name" | tr - _)
venv="${2:-.venvs/$name}"
[ -x "$venv/bin/python" ] || uv venv -q --python 3.12 "$venv"
VIRTUAL_ENV="$venv" uv pip install -q --torch-backend cpu -e ".[laya]" \
  --group "tests/integrations/pyproject.toml:int-$name"
DS_EXAMPLES="04-integrations/$name" "$venv/bin/python" -m pytest -o addopts="" -q -p no:warnings \
  "tests/integrations/test_$module.py" tests/test_examples.py --cov=decisionsmith --cov-branch \
  --cov-report=json:"$venv/coverage.json"
"$venv/bin/python" scripts/check_coverage.py --file "$venv/coverage.json" --only "integrations/$module.py"
