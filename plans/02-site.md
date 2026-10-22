# decisionsmith has a one-page site that says what it does and how to start

## The ask

The user wants every repository to carry a `site/` folder holding a
presentable one-page site for users, modelled on
`/home/izam/fun/mac-notch/site/index.html`: one self-contained HTML file,
no scripts, no fonts fetched, one screen with the pitch, one call to
action and links to the releases and the README.

Built so far: `docs/built/BUILT-2026-10-22-1827.md`.

decisionsmith has no `site/`. Its repository homepage field points at the
PyPI page (`gh repo view`), and its README already carries the pitch, the
install line and the first example, plus a themed banner in `assets/`.
The site is for a person deciding whether to try the library: what it does
(an LLM teaches a fast model, which takes over field by field), the one
install line, the three-line example, and links to the docs, PyPI and
GitHub.

How mac-notch's site is published was looked for and not found: no Pages
workflow, no `vercel.json`, and GitHub Pages is not enabled on that
repository (`gh api repos/izam-mohammed/mac-notch/pages` returns 404). So
publishing is decided here, not copied.

## Alternatives

### Decision 1: how the page is built (two-way door)

Searched 2026-10-22: "single page landing site plain HTML vs Astro vs
MkDocs project homepage performance"; the Astro docs (`docs.astro.build`),
the MkDocs comparison on stackshare.io, and mac-notch's `site/index.html`
(7,769 bytes, no scripts).

| Criterion | Weight | Hand-written `site/index.html` | Astro | MkDocs Material |
| --- | ---: | ---: | ---: | ---: |
| Performance efficiency | 40 | 5 | 5 | 3 |
| Fit to the ask | 30 | 5 | 4 | 2 |
| Compatibility | 15 | 5 | 3 | 3 |
| Security | 15 | 5 | 3 | 3 |
| **Weighted total** | **100** | **5.00** | **4.10** | **2.70** |

- Efficiency: one HTML file with inline CSS and no script is one request;
  Astro's output for a page with no islands is the same HTML (Astro docs),
  so both 5; MkDocs Material ships its theme's JavaScript and search index,
  3. No measurement was made; a Lighthouse run in Phase 1 records it.
- Fit: the ask is one page like mac-notch's; MkDocs is built for many
  pages of documentation with navigation, 2.
- Compatibility: mac-notch's site is a hand-written file; Astro and MkDocs
  add a Node or Python build to a library repository.
- Security: no build dependency at all against a dependency tree to audit.

**Chosen: a hand-written `site/index.html`.** It is the reference's shape,
the fastest page possible and adds no dependency. Revisit if the site grows
past one page or should publish `docs/` too.

### Decision 2: where the site is published (two-way door)

Searched 2026-10-22: "GitHub Pages vs Cloudflare Pages vs Vercel static
site 2026 comparison bandwidth limits", "GitHub Pages deploy from folder
actions/upload-pages-artifact deploy-pages subdirectory site"; the
`actions/deploy-pages` README, and secondary comparisons (guptadeepak.com,
bootstrap.build, 2026) for limits, to be confirmed on each provider's own
page.

| Must | GitHub Pages (Actions) | Cloudflare Pages | Vercel (house default, `project-setup`) |
| --- | --- | --- | --- |
| Free for a public open-source project | yes | yes | yes (Hobby, non-commercial) |
| Publishes only `site/`, not the repository | yes (`upload-pages-artifact` `path: site`) | yes (output directory) | yes (root directory) |
| **Survives** | **yes** | **yes** | **yes** |

| Criterion | Weight | GitHub Pages (Actions) | Cloudflare Pages | Vercel |
| --- | ---: | ---: | ---: | ---: |
| Performance efficiency | 35 | 4 | 5 | 5 |
| Security | 25 | 5 | 3 | 3 |
| Running cost and operability | 20 | 5 | 4 | 4 |
| Reliability | 20 | 4 | 4 | 4 |
| **Weighted total** | **100** | **4.45** | **4.10** | **4.10** |

- Efficiency: all three serve static files from a CDN; secondary sources
  give GitHub Pages a 100 GB per month soft bandwidth limit, Cloudflare
  Pages unmetered static bandwidth and Vercel Hobby 100 GB. A one-page
  site of a few kilobytes is far inside every limit; Pages scores 4 for the
  soft cap, unmeasured otherwise.
- Security: GitHub Pages deploys with the workflow's own OIDC token
  (`pages: write`, `id-token: write`) and no third-party access to the
  repository; Cloudflare and Vercel need either their GitHub app with
  repository access or an API token stored as a repository secret.
- Operability: Pages lives in one workflow file in this repository and
  needs no other account; the others add a project on another service.

Sensitivity: moving 10 points from security to efficiency gives 4.35
against 4.30; GitHub Pages holds. Security carries the result.

**Chosen: GitHub Pages, deployed by an Actions workflow from `site/`.**
It needs no secret and no outside account. The house default for hosting
is Vercel (`project-setup` `SKILL.md`), so this goes to the user before
Phase 2 (`## Open decisions`). Revisit if the site needs server-side code.

## The decision

The site is one file, `site/index.html`, written by hand in the reference's
shape (decision 1): the pitch and the A+ joke of the banner, the install
line, the three-line example, what it does in four short rows (teacher,
shadow, finetune, switch per field), and links to PyPI, the docs and
GitHub. Light and dark through `prefers-color-scheme`. No script, no
external font, no tracking. The banner SVGs in `assets/` may be inlined
or copied into `site/`; the page links nowhere it does not need.

It is published by `.github/workflows/site.yml` on pushes to `main` that
touch `site/**`, with `actions/upload-pages-artifact` (`path: site`) and
`actions/deploy-pages`, both pinned by commit SHA (decision 2). The
repository's homepage field stays on PyPI unless the user asks otherwise;
the README links the site.

Dependencies and permissions: no package. GitHub Pages enabled on the
repository with source "GitHub Actions" (a settings change the user makes
or approves); the deploy job gets `pages: write` and `id-token: write`
only.

Standards: `docs/standards.md` (`make check` once
`plans/01-quality-gates.md` has landed, which this plan follows in
`PLAN.md`).

## Where it lands

| Path | New or edit | Phase | What changes | Firm? |
| --- | --- | --- | --- | --- |
| `site/index.html` | new | 1 | the page | expected |
| `.github/workflows/site.yml` | new | 2 | build-free deploy of `site/` to Pages | expected |
| `README.md` | edit | 2 | one link to the site under the banner | expected |
| `docs/standards.md` | edit | 2 | a row for the site's checks and how it is published | reuse |

*Reused, not rewritten:*

- `README.md`: the pitch, install line and first example are copied from
  it, not rewritten, so the two say the same thing.
- `assets/banner-dark.svg`, `assets/banner-light.svg`: the visual identity;
  no new artwork.
- `/home/izam/fun/mac-notch/site/index.html`: the layout pattern (one
  frame, header, centred main, footer, inline CSS), not its colours or
  copy.
- Searched for an existing site or Pages config (`site/`, `docs/index.html`,
  `mkdocs.yml`, `.github/workflows/*pages*`): none.

## Security

What is protected: the repository and the published page. The page takes
no input and runs no script. The new entry point is the deploy workflow.

| Threat | Here | Handled by | Proved by |
| --- | --- | --- | --- |
| Tampering | a pull request from a fork changes what is published | the workflow runs only on `push` to `main`, never on `pull_request` | zizmor (the gate's stage 4) reports no dangerous trigger |
| Elevation of privilege | the deploy job given more than it needs | `permissions: {}` at the top; `pages: write`, `id-token: write` on the deploy job only; actions pinned by SHA | zizmor reports no excessive permission or unpinned action |
| Information disclosure | the page leaks more than the README | the page holds only what `README.md` already publishes; no analytics, no external request | a check that `site/index.html` has no `<script` and no `http` URL outside the link list |

No personal data, no secret. Checklist: OWASP Top 10 (current edition) for
a static page, which reduces to no third-party content.

## Rollback

Disable GitHub Pages in the repository settings, or revert the commit that
added `site.yml`; the README link is removed in the same revert. No data is
kept anywhere.

## Open decisions

- Hosting (decision 2): GitHub Pages won over the house default, Vercel,
  4.45 against 4.10, on security (no third-party app or token). Default:
  GitHub Pages, unless the user picks Vercel before Phase 2.
- Whether the repository homepage field moves from PyPI to the site.
  Default: it stays on PyPI.

## Declined

- Astro (decision 1): no faster for one page, and adds a Node build to a
  Python library.
- MkDocs Material (decision 1): built for multi-page docs; ships theme
  JavaScript a one-pager does not need.
- Cloudflare Pages and Vercel (decision 2): lose on security, a
  third-party app or API token with repository access. Vercel is the house
  default and is put to the user.

## What the tests hold

The page is static, so the checks are on the file: it is valid HTML
(`npx --yes html-validate site/index.html` or the W3C validator, recorded
in the phase), holds no `<script` and no external resource, and the
install line and example match `README.md`. The deploy is checked by
loading the published URL.

## Phases

### Phase 1: the rough cut: the page, opened locally (needs you: try it)

- [ ] `site/index.html` says what decisionsmith does, shows
  `uv add "decisionsmith[all]"` and the three-line harness example from
  `README.md`, and links to PyPI, `docs/README.md` and GitHub, in one
  self-contained file with inline CSS, light and dark.
- [ ] Check: `make check` passes; `grep -c '<script' site/index.html`
  prints 0; a Lighthouse run on the file records performance and
  accessibility in the PR body.
- [ ] You try it: open `site/index.html` in a browser, in light and dark
  mode and at phone width, and say whether it is the page you would send
  someone.

### Phase 2: published

- [ ] `.github/workflows/site.yml` deploys `site/` to GitHub Pages on
  pushes to `main` touching `site/**`, actions pinned by SHA, least
  permissions, and `README.md` links the published URL.
- [ ] `docs/standards.md` gains how the site is published and checked.
- [ ] Check: `make check` passes; the Security workflow's zizmor stage
  reports no finding for `site.yml`; after merge, the Pages URL serves the
  page (needs Pages enabled with source "GitHub Actions").

### Phase 3: open the PR

- [ ] The branch pushed and its PR opened against the branch it was cut
  from (or, if one is already open for this branch, its body brought up to
  date) with a Conventional Commit title for the whole change, a body naming
  this plan, what changed and why, and the checks run with their result.
  The PR's URL printed as the evidence. Opened, never merged.
