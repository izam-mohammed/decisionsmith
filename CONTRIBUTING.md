# Contributing to decisionsmith

Thanks for helping. Bug reports, docs fixes, examples and new integrations are all welcome. For a larger change,
please open an issue first so we can agree on the shape before you write the code.

By taking part you agree to the [Code of Conduct](CODE_OF_CONDUCT.md). Security problems go through
[private reporting](SECURITY.md), not public issues.

## Set up

decisionsmith uses [uv](https://docs.astral.sh/uv/) for everything (no pip, no conda):

```bash
git clone https://github.com/izam-mohammed/decisionsmith.git
cd decisionsmith
uv sync --extra laya --extra anthropic --extra mcp
uv run pre-commit install
```

## Tests and checks

```bash
uv run pytest
uv run python scripts/check_coverage.py --min 95 --skip integrations/
uv run pre-commit run --all-files
```

- Tests run offline: no network, no GPU, no API keys. Use `decisionsmith.testing.FakeEngine`, mocked HTTP and
  the tiny Laya checkpoint.
- Each module needs at least 95% line and branch coverage.
- CI runs on Linux, macOS and Windows with Python 3.10 to 3.13, so avoid hard-coded `/` in path assertions.
- Every Python block in `README.md` and `docs/*.md` runs in the docs test. A block that cannot run carries a
  `<!-- no-test: reason -->` comment.
- pre-commit runs Ruff (lint and format), mypy, the uv lock check and the commit message check.

## Keep it simple

The public API is small on purpose. Please discuss new public names or options in an issue before adding them.
The core dependencies stay `pydantic` and `httpx`, and `import decisionsmith` must not load anything heavy.

## Commits and pull requests

- Work on a branch (`feat/...`, `fix/...`, `docs/...`, `chore/...`) and open a pull request against `main`.
- Commit messages and pull request titles follow [Conventional Commits](https://www.conventionalcommits.org/):
  `type(scope): summary`, imperative, lowercase, no trailing period, at most 72 characters. Types: `feat`, `fix`,
  `docs`, `test`, `refactor`, `perf`, `build`, `ci`, `chore`, `revert`. Example:
  `feat(integrations): add langgraph route_on and guard_node`.
- Fill in the pull request template: what and why, how you tested it, and the checklist.
- Pull requests are squash merged, so the title becomes the commit message on `main`.
- Update the docs and examples for any change in behaviour, and add a line to `CHANGELOG.md` under
  `## Unreleased`.
- No AI attribution in commits, pull requests, code or docs (no `Co-Authored-By` lines for tools, no generated-by
  footers).

## Adding an integration

Each integration with another framework is:

1. one lazily imported module, `src/decisionsmith/integrations/<name>.py`, built on `integrations/_base.py`
2. its own optional extra in `pyproject.toml`, e.g. `decisionsmith[<name>]`
3. a pinned `int-<name>` dependency group in `tests/integrations/pyproject.toml` with the framework versions it is
   tested against
4. tests in `tests/integrations/test_<name>.py` and runnable examples in `examples/04-integrations/<name>/`
5. a section in `docs/integrations.md`

Run it in its own environment (frameworks never share one):

```bash
scripts/integration.sh <name>
```

The integrations workflow runs only the integrations your pull request touches, and all of them when a core file
changes.

## Licence

decisionsmith is Apache-2.0. By contributing you agree that your contribution is licensed under the same terms.
