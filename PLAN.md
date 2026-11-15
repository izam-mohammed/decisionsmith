# decisionsmith: what is left to build, in order

The library is usable and released (`docs/built/BUILT-2026-10-22-1827.md`).
What is left is repository standards. The README already has its themed
banner (`assets/`, PR `#24`), so no README plan is written.

### Phase 1: one check command and the shared security gate

- [ ] `plans/01-quality-gates.md` - `make check` runs every check, and every
  pull request runs the shared secret scan, audit and SAST gate.

### Phase 2: a one-page site

- [ ] `plans/02-site.md` - `site/index.html`, published by GitHub Pages.

### Phase 3: detailed documentation

- [ ] `plans/03-docs.md` - a Sphinx docs site at `/docs/` beside the landing
  page, with the API, CLI and MCP reference generated from the code.
