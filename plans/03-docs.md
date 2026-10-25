# decisionsmith has a documentation site whose API, CLI and MCP reference is generated from the code

## The ask

A tool that needs detailed documentation gets it, planned in its own
repository.
ragrank is the model: a Sphinx site on Read the Docs with getting started,
guides and an API reference (`/home/izam/fun/ragrank/docs/`).

Built so far: `docs/built/BUILT-2026-10-22-1827.md` (`main` is still on
`719ff73`, the commit it names).

Detailed docs are required here. decisionsmith is a library on PyPI with a
public Python API (`decisionsmith/__init__.py` `__all__`), a CLI with ten
subcommands (`decisionsmith.cli:parser`), an MCP server
(`decisionsmith.mcp.TOOLS`), extras per framework and a model folder
format. The words already exist: twelve guide pages in `docs/` (about 1,200
lines), indexed by `docs/README.md` and run as code by `tests/test_docs.py`.
What is missing is a site to read them in, with navigation and search, and
any reference generated from the code: today the CLI flags, the MCP tools
and the API signatures are described only by hand, so they can drift.

`plans/02-site.md` lands the landing page (`site/index.html`). This plan is
the reference behind it: the site says what decisionsmith is, the docs say
how to use every part of it.

## Alternatives

### Decision 1: which generator builds the docs (two-way door)

The sources stay Markdown in `docs/` whichever generator renders them, so
moving later means rewriting one config file and the reference directives.
The generator must render the existing pages, generate the API reference
from docstrings written in Markdown style (single backticks, indented
examples), and generate the CLI reference from the argparse parser.

Searched 2026-10-22: "Material for MkDocs maintenance mode Zensical 2026",
"Sphinx vs MkDocs vs Zensical python documentation 2026 build speed",
"Zensical mkdocstrings support API reference python", "sphinx-autodoc2 MyST
docstrings markdown maintained release 2026", "sphinx-argparse generate CLI
reference from argparse parser documentation", "Starlight Astro docs
performance zero JavaScript Pagefind search"; the Material for MkDocs blog
(maintenance mode announced 11/11/2025), the Zensical docs on mkdocstrings,
the sphinx-autodoc2 docs, the sphinx-argparse-cli PyPI page, the Scientific
Python development guide on docs, and the PyPI JSON API for every
package's latest release date (read 22/10/2026).

| Must | Sphinx + MyST + autodoc (ragrank's) | Zensical + mkdocstrings | MkDocs + Material + mkdocstrings | Sphinx + sphinx-autodoc2 | pdoc | Starlight | Baseline: Markdown on GitHub |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Still maintained | yes (Sphinx 9.2.0, 21/10/2026) | yes (0.0.71, 16/10/2026) | no (MkDocs 1.6.1 is from 30/08/2024; Material in maintenance mode since 11/11/2025) | no (0.5.0, 27/11/2023) | yes | yes | yes |
| API reference generated from the Python code | yes | yes | yes | yes | yes | no (no Python API generator) | no |
| Guides with navigation beside the reference | yes | yes | yes | yes | no (API pages only) | yes | no |
| **Survives** | **yes** | **yes** | no | no | no | no | no |

| Criterion | Weight | Sphinx + MyST + autodoc | Zensical + mkdocstrings |
| --- | ---: | ---: | ---: |
| Performance efficiency | 35 | 3 | 3 |
| Fit to the ask | 25 | 3 | 4 |
| Reliability | 20 | 5 | 2 |
| Compatibility | 10 | 5 | 3 |
| Reversibility | 10 | 4 | 4 |
| **Weighted total** | **100** | **3.70** | **3.15** |

- Efficiency 3 for both: no published benchmark was found comparing build
  time or page weight; Zensical's speed is a stated aim, not a measurement
  (Scientific Python guide). Phase 1 records the build time and the size
  of the built quickstart page, so the score can be replaced by a number.
- Fit: Sphinx's autodoc reads docstrings as reStructuredText, so the
  Markdown-style single backticks need `default_role = "code"` and an
  indented example renders as a quote, not code (3). mkdocstrings renders
  docstrings as Markdown natively, but its Zensical support is preliminary
  (backlinks missing, per the Zensical docs) and no argparse renderer for
  Zensical was found, so the CLI page would need its own script (4).
  Sphinx has `sphinx-argparse-cli` (1.24.0, 11/10/2026) for the CLI.
- Reliability: Sphinx has been the Python default since 2008 and MyST-Parser
  released 5.1.0 on 13/05/2026; Zensical is a 0.0.x series whose own docs
  say its API reference support will be rethought (2).
- Compatibility: ragrank, the model, is Sphinx with MyST and Furo; the
  `docs/*.md` pages are plain Markdown either way.

Sensitivity: moving 10 points from reliability to efficiency changes
nothing while both efficiency scores are equal; moving 10 points from
reliability to fit gives 3.50 against 3.35, and Sphinx still holds. A
measured efficiency gap of two points in Zensical's favour would flip it.
Risk: autodoc output for Markdown-style docstrings reads poorly; limited by
fixing those docstrings in place in Phase 2, which also improves `help()`.

**Chosen: Sphinx with MyST-Parser, autodoc, sphinx-argparse-cli and the
Furo theme.** It renders the existing pages and the CLI from the parser
today, and it is the stable choice the model repository already uses.
Revisit if Zensical leaves 0.0.x with full mkdocstrings support, or if the
Phase 1 measurement shows a large efficiency gap.

### Decision 2: where the docs are published (two-way door)

`plans/02-site.md` decision 2 publishes `site/` with GitHub Pages from an
Actions workflow (with an open decision on Vercel). The repository is
public, so every option below is free.

Searched 2026-10-22: "Read the Docs Community private repository pricing
Business plan", "Read the Docs Addons injected JavaScript flyout ads
community hosting pull request previews", "Cloudflare Workers static assets
free plan limits requests files 2026"; the Read the Docs pricing page, the
Read the Docs Addons blog post (04/2024), Cloudflare's Workers limits page,
and `plans/02-site.md` for GitHub Pages.

| Criterion | Weight | Read the Docs Community (ragrank's) | GitHub Pages under `/docs/` beside the site | Cloudflare Pages |
| --- | ---: | ---: | ---: | ---: |
| Performance efficiency | 35 | 3 | 4 | 5 |
| Security | 25 | 3 | 5 | 3 |
| Fit to the ask | 20 | 5 | 3 | 3 |
| Running cost and operability | 20 | 4 | 5 | 3 |
| **Weighted total** | **100** | **3.60** | **4.25** | **3.70** |

- Efficiency: Read the Docs Community is ad-supported and injects its
  Addons script into every page (Read the Docs pricing page and Addons
  post), 3; GitHub Pages serves the static build from its CDN with a soft
  100 GB monthly bandwidth limit (`plans/02-site.md`), 4; Cloudflare static
  assets are unmetered (Cloudflare limits page), 5. Unmeasured.
- Security: Pages deploys with the workflow's own token and no third-party
  script on the page; Read the Docs runs its own script and ads on every
  page and needs its GitHub app; Cloudflare needs an API token as a secret.
- Fit: Read the Docs gives versioned docs per release and pull request
  previews (5); Pages and Cloudflare serve one version unless the workflow
  builds more (3).
- Operability: Pages adds one build step to the workflow
  `plans/02-site.md` already lands; the others are another account.

**Chosen: GitHub Pages, the docs built into `/docs/` of the same
deployment as `site/`.** The landing page is the root, the reference is
`/docs/`, one origin, one workflow, no third-party script. ragrank's host,
Read the Docs, loses on efficiency and security (3.60 against 4.25), so
this goes to the user (`## Open decisions`). If `plans/02-site.md` moves to
Vercel, the docs follow the site to the same host under `/docs/`. Revisit
if readers need docs per released version.

## The decision

The docs are a Sphinx site built from `docs/` (decision 1): `docs/conf.py`
with MyST-Parser for the Markdown pages, autodoc for the public API with
`default_role = "code"`, `sphinx-argparse-cli` on `decisionsmith.cli:parser`
for the CLI, and Furo as the theme. The MCP tool list is rendered from
`decisionsmith.mcp.TOOLS` by a small local directive, so a tool added to
the server appears in the docs without anyone writing it down. The
existing pages keep their names and stay where they are; `docs/README.md`
stays as the GitHub index and links the site. `docs/standards.md`,
`docs/built/` and `docs/roadmap.md` are excluded from the build.

The build runs as `make docs` (`sphinx-build -W --keep-going -n`, warnings
and broken internal references fail it), which `make check` runs. CI also
runs `sphinx-build -b linkcheck` for external links, outside `make check`
so the local check stays offline like the tests. It is published by the
Pages workflow from `plans/02-site.md`, which gains a docs build step and
serves the result at `/docs/` (decision 2).

Dependencies: a new `docs` dependency group in `pyproject.toml` with
`sphinx`, `myst-parser`, `furo` and `sphinx-argparse-cli`, locked in
`uv.lock`, never in the published wheel's requirements. No new secret,
network host or permission beyond the Pages deployment
`plans/02-site.md` already names.

Standards: `docs/standards.md` (`make check` from
`plans/01-quality-gates.md`: format, lint, types, tests with coverage,
audit, secret scan), gaining a docs build row in Phase 4.

## Where it lands

| Path | New or edit | Phase | What changes | Firm? |
| --- | --- | --- | --- | --- |
| `docs/conf.py` | new | 1 | Sphinx config: MyST, autodoc, argparse-cli, Furo, excludes | expected |
| `docs/index.md` | new | 1 | the docs home and the table of contents | expected |
| `pyproject.toml`, `uv.lock` | edit | 1 | the `docs` dependency group; `Documentation` URL in Phase 5 | expected |
| `docs/reference/api.md` | new | 2 | autodoc of everything in `decisionsmith.__all__` | expected |
| `docs/reference/cli.md` | new | 2 | `sphinx-argparse-cli` on `decisionsmith.cli:parser` | expected |
| `docs/reference/mcp.md`, `docs/_ext/mcp_tools.py` | new | 2 | the MCP tool list from `decisionsmith.mcp.TOOLS` | expected |
| `src/decisionsmith/*.py` | edit | 2 | docstrings the reference renders badly, fixed in place | expected |
| `docs/*.md` | edit | 3 | cross-links between guides and the reference | reuse |
| `Makefile` | edit | 4 | `docs` target, run by `check` (the file lands in `plans/01-quality-gates.md`) | expected |
| `.github/workflows/ci.yml` | edit | 4 | the docs build and linkcheck job | reuse |
| `.github/workflows/site.yml` | edit | 5 | build the docs into `/docs/` of the Pages artefact (the file lands in `plans/02-site.md`) | expected |
| `README.md`, `docs/README.md`, `site/index.html` | edit | 5 | link the published docs (`site/index.html` lands in `plans/02-site.md`) | expected |
| `docs/standards.md`, `CONTRIBUTING.md` | edit | 4 | the docs build as a check, and how to run it | reuse |

*Reused, not rewritten:*

- `docs/*.md`: the twelve guide pages are the content; nothing is moved or
  rewritten to fit the generator.
- `tests/test_docs.py`: already runs every `python` block in `docs/*.md`
  offline, so the guides' examples cannot drift; it is kept as is, and new
  reference pages hold no runnable blocks of their own.
- `examples/README.md` and `notebooks/`: linked from the docs as the
  examples gallery, not copied in.
- `decisionsmith.cli:parser` and `decisionsmith.mcp.TOOLS`: the sources the
  CLI and MCP reference are generated from.
- `/home/izam/fun/ragrank/docs/docs/conf.py`: the MyST extension list and
  Furo setup, as a starting point.
- Searched for an existing docs config (`conf.py`, `mkdocs.yml`,
  `zensical.toml`, `.readthedocs.yaml`, `docs/_build`): none.

## Security

The docs are a static site with no input. What is at stake is the build
(new dependencies, and autodoc imports the package to read it) and what
gets published.

| Threat | Here | Handled by | Proved by |
| --- | --- | --- | --- |
| Tampering | a compromised docs dependency runs code in the build | the `docs` group is locked in `uv.lock` and covered by the osv-scanner audit in `make check` (`plans/01-quality-gates.md`); the docs job has `contents: read` only | `make check` audit stage passes; zizmor in the shared gate reports no excess permission |
| Information disclosure | `docs/built/`, `docs/standards.md` or the local `docs/roadmap.md` published | `exclude_patterns` in `docs/conf.py` | a check in Phase 1 that the built HTML holds no page from those paths |
| Elevation of privilege | the deploy job gains more than it needs | the docs step joins the existing build job of `site.yml`; only the deploy job keeps `pages: write`, `id-token: write` | zizmor reports no finding for `site.yml` |

No personal data and no secret. No analytics or third-party script on the
pages (decision 2). Checklist: SLSA build guidance for the CI build, and
the OWASP Top 10 (current edition) for a static site, which reduces to no
third-party content.

## Open decisions

- Host (decision 2): GitHub Pages under `/docs/` won over ragrank's Read
  the Docs, 4.25 against 3.60, on efficiency and security (no injected
  script or ads). Default: GitHub Pages, unless the user picks Read the
  Docs before Phase 5.
- Versioned docs per release: default, one version (latest `main`); revisit
  with the host if readers ask for older versions.

## Declined

- MkDocs with Material and mkdocstrings (decision 1): failed the
  maintained must; MkDocs' last release is 1.6.1 (30/08/2024) and Material
  is in maintenance mode since 11/11/2025.
- Sphinx with sphinx-autodoc2 (decision 1): failed the maintained must,
  last release 27/11/2023, though it reads Markdown docstrings natively.
- pdoc (decision 1): failed the must for guides with navigation.
- Starlight (decision 1): failed the must for a Python API reference.
- Markdown on GitHub only (decision 1): the baseline; no generated
  reference, no search.
- Zensical with mkdocstrings (decision 1): lost on reliability, a 0.0.x
  series with preliminary API reference support.
- Read the Docs Community (decision 2): lost on efficiency and security,
  an injected script and ads on every page; put to the user as ragrank's
  host.
- Cloudflare Pages (decision 2): lost on security and operability, an API
  token and another account for no gain at this traffic.

## What the tests hold

`make docs` fails on any Sphinx warning, so a broken cross-reference, a
missing page in the table of contents, or a docstring autodoc cannot parse
fails the check. The reference pages are generated at build time from
`decisionsmith.__all__`, `decisionsmith.cli:parser` and
`decisionsmith.mcp.TOOLS`, so they cannot fall behind the code: a renamed
flag or a new tool changes the page in the same build. `tests/test_docs.py`
keeps running every example in the guides. The CI linkcheck catches dead
external links.

## Repository knowledge

Read root `KNOWLEDGE.md` when present before each phase. Follow the global and
repository instructions for recording useful development findings.

## Phases

### Phase 1: the rough cut: the docs build locally (needs you: try it)

- [ ] `uv run --group docs sphinx-build docs docs/_build/html` builds a site
  with a home page, `quickstart.md` as getting started and `guide.md` as
  the first real page, Furo theme, light and dark (expected in
  `docs/conf.py`, `docs/index.md`, the `docs` group in `pyproject.toml`).
- [ ] `docs/standards.md`, `docs/built/` and `docs/roadmap.md` are
  excluded, checked by `docs/_build/html/standards.html` and
  `docs/_build/html/built/` not existing; `docs/_build/` is git-ignored.
- [ ] Check: `make check` passes; the build exits 0 with no warning; the
  build time and the size of `docs/_build/html/quickstart.html` are
  recorded in the PR body as decision 1's efficiency baseline.
- [ ] You try it: run the build, open `docs/_build/html/index.html`, read
  the quickstart and the guide in light and dark and at phone width, and
  say whether this is the docs you want to grow.

### Phase 2: reference generated from the code

- [ ] `docs/reference/api.md` documents every name in `decisionsmith.__all__`
  with autodoc, signatures and docstrings, and docstrings that render
  badly are fixed in place (expected in `src/decisionsmith/*.py`).
- [ ] `docs/reference/cli.md` renders every `decisionsmith` subcommand and
  flag from `decisionsmith.cli:parser` with `sphinx-argparse-cli`.
- [ ] `docs/reference/mcp.md` lists every tool in `decisionsmith.mcp.TOOLS`
  with its signature and docstring through a local directive (expected in
  `docs/_ext/mcp_tools.py`), so a new tool appears without a docs edit.
- [ ] Check: `make check` passes; the build has no warning; adding a dummy
  flag to `parser()` locally shows it on the CLI page, then is reverted.

### Phase 3: guides and examples

- [ ] Every page in `docs/README.md` is in the table of contents, grouped as
  getting started (`quickstart`), guides (`finetune`, `golden`, `collect`,
  `evaluate`, `save-and-load`, `teachers`, `guide`, `integrations`,
  `agents`, `agent-build`) and help (`faq`), with links from each guide to
  the reference entries it names.
- [ ] An examples page links `examples/README.md` and `notebooks/` on
  GitHub, one line per example, so nothing is copied.
- [ ] Check: `make check` passes, including `tests/test_docs.py` on every
  page; the build has no warning.

### Phase 4: the docs build is a check

- [ ] `make docs` runs `sphinx-build -W --keep-going -n`, and `make check`
  runs it after the tests (expected in `Makefile`); `docs/standards.md`
  and `CONTRIBUTING.md` gain the row and the command.
- [ ] CI builds the docs the same way and runs `sphinx-build -b linkcheck`
  on every pull request, with `contents: read` only (expected in
  `.github/workflows/ci.yml`).
- [ ] Check: `make check` passes; it fails after adding a link to a page
  that does not exist, then passes again once it is removed; the CI job is
  green on the PR.

### Phase 5: published at `/docs/`

- [ ] The Pages workflow from `plans/02-site.md` builds the docs into
  `/docs/` of the same artefact as `site/`, actions pinned by SHA (expected
  in `.github/workflows/site.yml`), on pushes to `main` touching `docs/**`,
  `src/**` or `site/**`.
- [ ] `README.md`, `docs/README.md` and `site/index.html` link the
  published docs, and `pyproject.toml` gains a `Documentation` URL.
- [ ] Check: `make check` passes; zizmor in the shared gate reports no
  finding for `site.yml`; after merge, `/docs/` serves the home page and
  the CLI reference.

### Phase 6: open the PR

- [ ] The branch pushed and its PR opened against the branch it was cut
  from (or, if one is already open for this branch, its body brought up to
  date) with a Conventional Commit title for the whole change, a body
  naming this plan, what changed and why, and the checks run with their
  result. The PR's URL printed as the evidence. Opened, never merged.
