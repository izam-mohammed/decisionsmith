# decisionsmith: what every change passes

Run everything: there is no single command yet. Today a change passes the
three commands in `CONTRIBUTING.md` § *Tests and checks*, locally, and the
`ci` and `integrations` workflows on its pull request.
`plans/01-quality-gates.md` lands one check command and the shared
security gate; this file is updated in that plan.

Recorded 22/10/2026 from what the repository already runs on `719ff73`.
`CONTRIBUTING.md` stays the contributor-facing summary; this file is the
full record.

## Toolchain

| What | Version | Pinned in |
| --- | --- | --- |
| Python (development) | 3.12 | `.python-version` |
| Python (supported) | 3.10 to 3.13 | `pyproject.toml` (`requires-python`), the CI matrix |
| uv | 0.8 line for the build backend (`uv_build>=0.8.0,<0.9`) | `uv.lock`, committed; `uv sync --locked` in CI |
| Integration environments | one pinned group per framework | `tests/integrations/pyproject.toml` |

## Checks

| Check | Tool | Config | Command | Passing means |
| --- | --- | --- | --- | --- |
| Format | Ruff | `pyproject.toml` `[tool.ruff]` | `uv run ruff format --check .` | no diff |
| Lint | Ruff (E, F, W, I, B, UP, SIM, RUF, ERA) | `pyproject.toml` `[tool.ruff.lint]` | `uv run ruff check .` | no findings |
| Types | mypy | `pyproject.toml` `[tool.mypy]` (`check_untyped_defs`) | `uv run mypy` | no errors |
| Unit tests | pytest | `pyproject.toml` `[tool.pytest.ini_options]` | `uv run pytest --cov --cov-report=json` | all pass, offline |
| Coverage | coverage.py, line and branch | `pyproject.toml` `[tool.coverage]`, `scripts/check_coverage.py` | `uv run python scripts/check_coverage.py --min 95 --skip integrations/` | every module at 95 % or more |
| Docs blocks | pytest | `tests/test_docs.py` | in `uv run pytest` | every `python` block in `README.md` and `docs/*.md` runs, or carries a `no-test` reason |
| Integrations | pytest per framework | `tests/integrations/pyproject.toml` | `scripts/integration.sh <name>` | that framework's tests and examples pass |
| Lock file | uv-lock hook | `.pre-commit-config.yaml` | `uv run pre-commit run --all-files` | `uv.lock` matches `pyproject.toml` |
| Commit message | conventional-pre-commit | `.pre-commit-config.yaml` | commit-msg hook | Conventional Commits |
| Static analysis | CodeQL default setup | repository settings | runs on push and PR | no new alert |
| Supply chain posture | OpenSSF Scorecard | `.github/workflows/scorecard.yml` | weekly and on push to `main` | results published |
| Dependency updates | Dependabot, 7-day cooldown | `.github/dependabot.yml` | weekly | PRs opened |
| Dependency audit | none yet | | | `plans/01-quality-gates.md` |
| Secret scan | none yet | | | `plans/01-quality-gates.md` |
| Build | uv | `pyproject.toml` | `uv build` | a wheel and sdist (run in `release.yml`) |

## Tests

Tests live in `tests/`, one `test_<area>.py` per area. They run offline:
no network, no GPU, no API keys; use `decisionsmith.testing.FakeEngine`,
mocked HTTP and the tiny Laya checkpoint. Tests needing real weights or the
network are marked `slow` and skipped by default. Integration tests live in
`tests/integrations/test_<name>.py` and run only in their own environment.
One test: `uv run pytest tests/test_cli.py -k doctor`.

## Constraints

- A library on PyPI: the public API is small on purpose, the core
  dependencies stay `pydantic` and `httpx`, and `import decisionsmith`
  loads nothing heavy (`CONTRIBUTING.md`).
- Runs on Linux, macOS and Windows: no hard-coded `/` in path assertions.
- Releases publish on a `v*` tag (`.github/workflows/release.yml`): never
  tag without meaning to publish.

## Conventions

- Branches `feat/...`, `fix/...`, `docs/...`, `chore/...`; PRs against
  `main`, squash merged.
- Conventional Commits for commits and PR titles, lowercase, at most 72
  characters, checked by `.github/workflows/pr-title.yml`.
- Docs and examples change with behaviour; a line under `## Unreleased` in
  `CHANGELOG.md` per change.
- No AI attribution anywhere.

## CI

`.github/workflows/ci.yml` runs lint, format, mypy, then tests with the
coverage floor on 3 operating systems and 4 Python versions for every pull
request. `.github/workflows/integrations.yml` runs the integrations a PR
touches. `.github/workflows/pr-title.yml` checks the title. All read-only
tokens, actions pinned by commit SHA.

## Decided

- Ruff for lint and format, mypy for types, pytest with a 95 % per-module
  floor: chosen at the start (`d7c8db3`), recorded here, not re-decided.
- uv for environments and the build: `CONTRIBUTING.md`.
