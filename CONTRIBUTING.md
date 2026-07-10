# Contributing

Thanks for your interest in improving Cleopatra! This guide describes how the
project actually works so your change lands smoothly.

Please also read our [Code of Conduct](CODE_OF_CONDUCT.md) and follow it in all
your interactions with the project.

## Before you start — is it in scope?

Cleopatra is a **matplotlib-only convenience layer over in-memory NumPy data**.
Before proposing or building a feature, read [`SCOPE.md`](SCOPE.md): it defines
what belongs here (the glyph family, colour/colorbar/legend pipeline, animation
export, the fixed-public-dataset basemap helpers) and what does not (non-matplotlib
backends, user-data file I/O, GIS/geoprocessing, interactive GUIs, modelling, heavy
new dependencies). If a request falls outside that scope, it will be declined or
redirected rather than merged.

For anything non-trivial, open an issue to discuss the change before you invest
time in a pull request.

## Development setup

Cleopatra uses [uv](https://docs.astral.sh/uv/) for environment and dependency
management.

```bash
git clone https://github.com/serapeum-org/cleopatra.git
cd cleopatra

# create the environment and install the package with the dev + docs tooling
uv sync

# include the optional web-tile / basemap extra when working on geo features
uv sync --extra tiles
```

Run the test suite with pytest:

```bash
uv run pytest
```

Notebook-based docs examples are validated in CI; if you touch a public API,
update the matching example under `docs/notebooks/` and make sure it still
executes.

## Commit messages — Conventional Commits

**The version number and `docs/change-log.md` are generated automatically** by
[commitizen](https://commitizen-tools.github.io/commitizen/) from the commit
history — do **not** bump versions or edit the changelog by hand. Release notes
come entirely from well-formed commit messages, so write
[Conventional Commits](https://www.conventionalcommits.org/):

- `feat: ...` — a new feature (minor version bump)
- `fix: ...` — a bug fix (patch version bump)
- `docs: ...`, `test: ...`, `refactor: ...`, `chore: ...` — no version bump
- `feat!: ...` or a `BREAKING CHANGE:` footer — a breaking change (major bump)

## Pull request process

1. Work on a feature branch, not `main`.
2. Keep the change in scope (see above) and covered by tests; run `uv run pytest`
   locally.
3. Update the relevant documentation: the API reference page and/or example
   notebook under `docs/`, and the README/`docs/index.md` overview if you add a
   user-facing feature. Do not edit `docs/change-log.md` (auto-generated).
4. Use Conventional Commit messages so the release tooling can pick up your change.
5. Open the pull request against `main` and fill in the template. A maintainer will
   review and merge once CI is green.

## Code of conduct

This project and everyone participating in it is governed by the
[Cleopatra Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected
to uphold it; please report unacceptable behavior to the project maintainers.
