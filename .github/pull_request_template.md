<!-- Title: Conventional Commits, e.g. `feat(integrations): add langgraph route_on`. The squash merge reuses it. -->

## What and why



## How it was tested

<!-- Commands you ran and their results. -->

```bash
uv run pytest
uv run python scripts/check_coverage.py --min 95 --skip integrations/
uv run pre-commit run --all-files
```

## Checklist

- [ ] Tests pass (`uv run pytest`, plus `scripts/integration.sh <name>` for any integration touched)
- [ ] Coverage gate passes (>= 95% line + branch per module)
- [ ] Ruff lint + format and mypy pass (`uv run pre-commit run --all-files`)
- [ ] Docs and examples updated for any behaviour change
- [ ] No planning or marketing files added
- [ ] uv only in every install and run instruction

## Linked issue

<!-- Closes #... or "none" -->
