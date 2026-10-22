# Dependabot keeps the pre-commit hooks current as well as the actions and the uv lock

## The ask

The owner asked, on 22/10/2026, to "have dependabot in all of my repos", for
"proper security.md file for all of my repos", and for a proper licence on
every product repository.

This repository already has all three: `LICENSE` and `NOTICE` (Apache-2.0,
matching `license = "Apache-2.0"` in `pyproject.toml`), `SECURITY.md`, and
`.github/dependabot.yml` covering `github-actions` and `uv`. It is the model
the owner's other repositories copy. One ecosystem is missing:
`.pre-commit-config.yaml` pins three hook repositories by tag (uv-pre-commit,
ruff-pre-commit, conventional-pre-commit) and nothing updates them.

`tests/integrations/pyproject.toml` holds PEP 735 dependency groups, one per
framework, pinned on purpose to the versions each integration is tested
against, with versions that cannot share one environment (its header says
so). It has no lock file.

## Alternatives

### Decision 1: which bot keeps dependencies current? (two-way door)

Dependabot already runs here; the pre-commit hooks need the same.

Searched 22/10/2026: "dependabot pre-commit ecosystem", "renovate
pre-commit manager"; GitHub's Dependabot options reference (docs.github.com,
supported ecosystems list: `pre-commit` is a `package-ecosystem` value that
reads hook revisions from git tags), Renovate's `pre-commit` manager docs.

| Must | Baseline: Dependabot | Renovate |
| --- | --- | --- |
| Updates `rev:` pins in `.pre-commit-config.yaml` | yes | yes |
| Runs with no app install or token to manage | yes | no (needs the Renovate app) |
| **Survives** | **yes** | no |

| Criterion | Weight | Baseline: Dependabot |
| --- | ---: | ---: |
| Performance efficiency | 40 | 4 |
| Reliability | 30 | 4 |
| Compatibility | 30 | 5 |
| **Weighted total** | **100** | **4.30** |

- Dependabot compatibility 5: it already runs here, so the pre-commit
  entry is one more block in the same file.

**Chosen: Dependabot.** It already runs here and supports `pre-commit`.
Revisit if its pre-commit support misses a hook repository.

## The decision

Add a `pre-commit` entry to `.github/dependabot.yml` (weekly,
`cooldown: default-days: 7`, one group, prefix `chore(deps)`), in the
file's existing shape (Decision 1). The `github-actions` and `uv` entries
stay as they are. `SECURITY.md` and `LICENSE` are already right and stay
as they are. Dependencies and permissions: none.

`tests/integrations/pyproject.toml` gets no entry: its pins are the tested
versions, chosen by the integration matrix, and a grouped update across
groups that pin conflicting versions would be one unmergeable PR.

Security: no code, input, secret or permission changes; the change adds
one config block.

## Where it lands

| Path | New or edit | Phase | What changes | Firm? |
| --- | --- | --- | --- | --- |
| `.github/dependabot.yml` | edit | 2 | adds the `pre-commit` ecosystem | fixed |

*Reused, not rewritten:*

- `.github/dependabot.yml`: extended, not replaced.

## Declined

- Renovate: needs an app install, and Dependabot already runs here.
- Dependabot on `tests/integrations/`: the pins are deliberate and
  conflict across groups.

## What the tests hold

The YAML parses; GitHub accepts the Dependabot config on the PR.

## Phases

### Phase 1: topics

- [x] The existing topics kept as they are (`classification`,
  `claude-code-plugin`, `distillation`, `fine-tuning`, `jev`, `langchain`,
  `laya`, `llamaindex`, `llm`, `machine-learning`, `mcp-server`, `python`,
  `system-one`): all thirteen are accurate, so none is removed; check:
  `gh repo view --json repositoryTopics` lists them.

### Phase 2: Dependabot

- [x] `.github/dependabot.yml` gains a `pre-commit` entry beside the
  `github-actions` and `uv` ones; check: the file parses with
  `yaml.safe_load`.

### Phase 3: security policy

- [x] `SECURITY.md` read against this repository and kept as it is: it
  names the package, CLI, MCP server and release workflow, and the private
  reporting link; check: the link points at this repository.

### Phase 4: open the PR

- [x] The branch pushed and its PR opened against `main` (or, if one is
  already open for this branch, its body brought up to date) with the
  title `chore: add pre-commit to dependabot`, a body naming this plan,
  what changed and why, and the checks run with their result. The PR's
  URL printed as the evidence. Opened, never merged.
