# one command checks a change, and every pull request is scanned for secrets and vulnerable dependencies

## The ask

Every repository needs proper lint and format, tests, a
coverage floor of 90 % (target 95 %), and security checks (dependency
audit, secret scan, static analysis), all run by one local check command and
by CI on every pull request, reusing the shared security gate in
`izam-mohammed/.github` rather than inventing a parallel one.

Built so far: `docs/built/BUILT-2026-10-22-1827.md`.

Once the tree was read, most of this is already met in decisionsmith:

- Lint, format and types: Ruff and mypy run in `.github/workflows/ci.yml`
  and in pre-commit.
- Tests and coverage: pytest on 3 operating systems and 4 Python versions,
  and `scripts/check_coverage.py --min 95` fails any module under 95 % line
  and branch coverage. The floor is above the asked 90 % and equal to the
  95 % target; on PR `#28` (CI run `39766592137`) every module outside
  `integrations/` measured 99.2 % or more. This plan keeps that floor and
  does not touch it.
- Static analysis: CodeQL default setup runs on every push and PR.

What is missing is narrower:

- **No secret scan** anywhere: no gitleaks in pre-commit or CI.
- **No dependency audit** on pull requests: Dependabot opens update PRs,
  but nothing fails or reports when a locked version carries an advisory.
- **No single check command**: `CONTRIBUTING.md` lists three commands, run
  by hand.
- **No workflow audit**: the workflows are hardened by hand (SHA pins,
  read-only tokens) but nothing checks them.

The shared gate (`izam-mohammed/.github`, `security-scan.yml`) covers the
first, second and fourth in CI: gitleaks (blocking), osv-scanner, semgrep
and zizmor (reporting), with a sticky PR comment.

## Alternatives

### Decision 1: how pull requests get the secret scan, audit and SAST (two-way door)

The brief says to reuse the shared gate. It is still compared, because a
weaker or slower option should not win by default.

Searched 2026-10-22: "gitleaks vs trufflehog vs betterleaks secret scanning
speed benchmark", "semgrep vs bandit vs CodeQL python SAST false positives
speed 2026"; read `izam-mohammed/.github` `README.md` and
`security-scan.yml` at `ff8da46`, and `repo-template`'s
`.github/workflows/security.yml`.

| Must | Baseline: CodeQL and Dependabot only | Shared gate (`security-scan.yml`) | Own jobs in `ci.yml` |
| --- | --- | --- | --- |
| A leaked secret blocks the merge | no (nothing scans for secrets) | yes (stage 1, gitleaks, blocking) | yes |
| Needs no repository secret | yes | yes (README: callers pass none) | yes |
| **Survives** | no | **yes** | **yes** |

| Criterion | Weight | Shared gate | Own jobs in `ci.yml` |
| --- | ---: | ---: | ---: |
| Performance efficiency | 35 | 4 | 4 |
| Fit to the ask | 25 | 5 | 4 |
| Security | 20 | 5 | 4 |
| Maintainability of the result | 20 | 5 | 2 |
| **Weighted total** | **100** | **4.65** | **3.60** |

- Efficiency 4 both: the same tools (gitleaks, osv-scanner, semgrep) in
  parallel jobs with 10 to 20 minute timeouts; no published timing for this
  repository, so neither scores 5. Reviewers describe gitleaks as faster
  than TruffleHog because it does no live verification (appsecsanta.com,
  trunk.io, 2026; no figures given).
- Shared gate fit 5: it adds zizmor for the workflows and a sticky PR
  report; own jobs would have to rebuild both.
- Shared gate security 5: zizmor-clean, SHA-pinned, no secrets passed
  (its `README.md`); own jobs start unaudited.
- Maintainability 5 against 2: one pinned `uses:` line here, tool versions
  bumped once in `izam-mohammed/.github` for every repository; own jobs
  are a second copy that drifts.

**Chosen: the shared gate.** It meets every must, reports more and keeps one
copy of the scanners for all repositories. Revisit if a stage it only
reports on must block here before the gate's owner tightens it.

### Decision 2: which dependency audit the local check runs (two-way door)

CI's audit is osv-scanner (the gate's stage 2). The local check should find
the same advisories.

Searched 2026-10-22: "pip-audit vs osv-scanner uv.lock support 2026"; the
Astral blog on `uv audit` (preview, 2026), the Google security blog on
OSV-Scanner v2 (03/2025, adds `uv.lock`).

| Criterion | osv-scanner | pip-audit | `uv audit` (preview) |
| --- | --- | --- | --- |
| Must: reads `uv.lock` | yes (v2) | not confirmed for `uv.lock` | yes |
| Must: still maintained, stable | yes | yes | no (preview, may change) |
| Performance efficiency | one Go binary, no environment | needs an environment resolved | Astral: 4 to 10 times faster than pip-audit, not like for like |
| Compatibility | the same tool and database as CI's stage 2 | a different tool from CI | different tool, same OSV data |

**Chosen: osv-scanner.** It reads the lock file, runs as one binary and
matches what CI reports. Revisit when `uv audit` leaves preview.

### Decision 3: the one check command (two-way door)

`CONTRIBUTING.md`'s three commands become one. The house style
(`project-setup` `SKILL.md`: "Task runner: Makefile, same target names in
every repo") names a Makefile.

Searched 2026-10-22: "Makefile vs justfile vs nox task runner python
project check command"; the charmhub make-vs-just comparison, the
cookiecutter-robust-python task automation page.

| Criterion | Weight | `Makefile` | `justfile` | `scripts/check.sh` |
| --- | ---: | ---: | ---: | ---: |
| Performance efficiency | 30 | 5 | 5 | 5 |
| Compatibility | 40 | 5 | 3 | 4 |
| Portability | 30 | 4 | 5 | 3 |
| **Weighted total** | **100** | **4.70** | **4.20** | **4.00** |

- Efficiency 5 all: each only starts the same tools.
- Compatibility: a Makefile with `check`, `lint`, `test`, `test-cov`,
  `security` keeps the target names of `project-setup/templates/Makefile.python`;
  `just` adds a binary to install; a shell script has no named targets.
- Portability: `make` is not on stock Windows, `just` is (single binary,
  charmhub comparison); CI's Windows tests keep running their commands
  directly, so the check command is a contributor tool.

**Chosen: a `Makefile`.** It keeps the house target names and adds nothing
to install on macOS or Linux. Revisit if Windows contributors need the one
command.

### Decision 4: security lint rules in the local check (two-way door)

semgrep runs in CI through the gate and CodeQL already runs. Locally the
question is whether to add a fast Python security linter.

Searched 2026-10-22: "semgrep vs bandit vs CodeQL python SAST false
positives speed 2026"; aicodereview.cc "Semgrep vs Bandit" (2026), the
`project-setup` `SKILL.md` note that Ruff's `S` rules replace bandit.

| Criterion | Ruff `S` rules | bandit | semgrep locally |
| --- | --- | --- | --- |
| Performance efficiency | inside the Ruff run already made | seconds, a second pass | about 10 s median (secondary source) |
| Fit to the ask | the bandit rule set | the original | broadest rules |
| Compatibility | no new tool, one config | new dependency | a second copy of CI's stage 3 |

**Chosen: Ruff `S` rules**, with `S101` ignored under `tests/`. No new tool,
no extra pass, and semgrep and CodeQL still run on the PR. Revisit if the
gate's semgrep finds a class of issue `S` misses.

## The decision

Standards gap, no new behaviour for library users. The usable thing is the
check command itself: Phase 1 lands `make check` and the user runs it.

- The shared gate is called from a new `.github/workflows/security.yml`
  with `ecosystem: python`, pinned to the `izam-mohammed/.github` commit
  current at build time (`ff8da46` on 22/10/2026) (decision 1).
- `make check` runs, in order: `ruff format --check`, `ruff check`
  (with `S` added, decision 4), `mypy`, `pytest --cov`, the coverage floor
  (95 %, unchanged), `osv-scanner` on `uv.lock` (decision 2) and
  `gitleaks dir` (decisions 1 and 3).
- gitleaks joins `.pre-commit-config.yaml`, as in `izam-mohammed/.github`'s
  own pre-commit file.

Dependencies and permissions: no Python package. `osv-scanner` and
`gitleaks` are developer binaries named in `docs/standards.md` with their
install line; the CI job gets `contents: read` and `pull-requests: write`
for the gate's PR comment, nothing more.

Standards: `docs/standards.md` (today the three `CONTRIBUTING.md`
commands; after Phase 1, `make check`).

## Where it lands

| Path | New or edit | Phase | What changes | Firm? |
| --- | --- | --- | --- | --- |
| `Makefile` | new | 1 | `check`, `lint`, `format`, `test`, `test-cov`, `security` targets | expected |
| `pyproject.toml` | edit | 1 | `S` in `[tool.ruff.lint] select`, `S101` ignored for `tests/*` | expected |
| `.github/workflows/security.yml` | new | 2 | calls the shared gate, `ecosystem: python` | fixed (the gate's documented caller) |
| `.pre-commit-config.yaml` | edit | 2 | the gitleaks hook | expected |
| `docs/standards.md` | edit | 1, 2 | `make check` and the security rows | reuse |
| `CONTRIBUTING.md` | edit | 1 | *Tests and checks* points at `make check` | reuse |

*Reused, not rewritten:*

- `izam-mohammed/.github` `.github/workflows/security-scan.yml`: the whole
  CI security stage; nothing in this repository copies it.
- `scripts/check_coverage.py`: the per-module floor, called by
  `make test-cov` unchanged.
- `project-setup/templates/Makefile.python`: target names and the
  `security` target's tools; its 70 % floor and `app/` paths are not
  copied.
- Searched for an existing task runner (`Makefile`, `justfile`,
  `noxfile.py`, `tox.ini`, `scripts/check*`): none.

## Security

What is protected: the repository's history and the PyPI release from
leaked credentials, and users from a release built on a vulnerable locked
dependency. The new entry point is one workflow run on `pull_request` and a
weekly schedule.

| Threat | Here | Handled by | Proved by |
| --- | --- | --- | --- |
| Information disclosure | a token committed in a PR | gitleaks, stage 1, blocks the merge; the pre-commit hook stops it locally | a throwaway branch with a fake `AKIA`-shaped key fails stage 1 (Phase 2) |
| Tampering | a compromised shared workflow | the `uses:` line pinned to a full commit SHA; Dependabot's `github-actions` updates bump it | zizmor (stage 4) reports no unpinned use |
| Elevation of privilege | the gate given more than it needs | `permissions: {}` at the top, `contents: read` and `pull-requests: write` on the job, no `secrets: inherit` | zizmor reports no excessive permission |

No secret is added. Checklist: CIS-style least privilege for CI and SLSA
build integrity, as the gate's `README.md` follows.

## Declined

- CodeQL and Dependabot alone (decision 1): fail the must that a leaked
  secret blocks the merge.
- Own scan jobs in `ci.yml` (decision 1): lose on maintainability, a second
  copy of the gate.
- pip-audit (decision 2): `uv.lock` support not confirmed; a different tool
  from CI.
- `uv audit` (decision 2): still a preview.
- `justfile` and `scripts/check.sh` (decision 3): lose on compatibility
  with the house target names.
- bandit and local semgrep (decision 4): a second pass or a second copy of
  CI for what Ruff's `S` rules already cover locally.
- Changing the coverage floor: it is already 95 %, above the asked 90 %.

## What the tests hold

The existing suite is unchanged and must stay green under `make check`. The
new rules are proved by running them: `make check` exits 0 on a clean tree
and non-zero with a planted fake secret; the gate's stage 1 fails on the
same fake secret in a throwaway PR and passes once it is removed.

## Open decisions

- The gate's stages 2 to 4 only report. Default: leave them reporting here
  and note the first PR's counts in `docs/standards.md`; tightening a stage
  is a change in `izam-mohammed/.github`, not here.

## Phases

### Phase 1: the rough cut: `make check` (needs you: try it)

- [ ] `make check` runs format check, lint, mypy, pytest with coverage, the
  95 % floor, osv-scanner on `uv.lock` and gitleaks, stopping at the first
  failure (expected in `Makefile`), with `S` added to Ruff's rules and any
  finding it raises fixed or ignored with a reason (`pyproject.toml`).
- [ ] `docs/standards.md` and `CONTRIBUTING.md` name `make check` as the one
  command, with the install line for `osv-scanner` and `gitleaks`.
- [ ] Check: `make check` exits 0 on this branch, and exits non-zero after
  adding a file holding a fake `AKIA`-shaped key (then removed).
- [ ] You try it: run `make check` in your checkout and say whether the
  output and the time it takes are what you want from the one command.

### Phase 2: every pull request runs the shared gate

- [ ] `.github/workflows/security.yml` calls
  `izam-mohammed/.github/.github/workflows/security-scan.yml` pinned to a
  full commit SHA with `ecosystem: python`, `permissions: {}` at the top
  and `contents: read`, `pull-requests: write` on the job.
- [ ] The gitleaks hook added to `.pre-commit-config.yaml`, and the
  security rows of `docs/standards.md` filled in (gate, stages, what
  blocks).
- [ ] Check: `make check` passes; on the PR, the Security workflow's four
  stages run, stage 1 passes and the sticky comment shows the counts, and a
  throwaway branch with a fake key fails stage 1.

### Phase 3: open the PR

- [ ] The branch pushed and its PR opened against the branch it was cut
  from (or, if one is already open for this branch, its body brought up to
  date) with a Conventional Commit title for the whole change, a body naming
  this plan, what changed and why, and the checks run with their result.
  The PR's URL printed as the evidence. Opened, never merged.
