# Summary

Round-3 review of PR #274 (`refactor/grouped-render-params` vs `main`). This branch replaces the flat
per-glyph style/scale/classify/contour/cell keyword arguments with grouped parameter objects
(`ColorScaling`, `Contour`, `CellValues`, `DataStyle`, `Classify`), extracts the norm-building logic into
`cleopatra.styling.scaling.ColorScaling`, and migrates docs/notebooks/tests.

- The three recent, previously-unreviewed refactors were verified behaviour-preserving: the `build_norm`
  per-kind dispatch table (`_NORM_BUILDERS`), the unnested `extend` if/elif/else, and the
  `pytest.raises` setup-hoisting in the tests all reproduce the original semantics exactly (norm construction
  for all 5 kinds, `cbar_kw["extend"]` values, tick handling). Doctests + full impacted suite pass.
- The recent `kde_glyph` `_UNSET_HILLSHADE` sentinel fix and the `mesh_glyph` `levels`-in-`MESH_DEFAULT_OPTIONS`
  fix are correct and coherent.
- **New correctness gap found**: the error-path rollback that KDE gained (commit `33c196f`) was **not**
  extended to the four primitive glyphs (`Flow`/`Vector`/`Scatter`/`Polygon`) or fully to `ArrayGlyph`. A
  failed `plot(...)` that passes a group object now permanently mutates `default_options`, poisoning later
  plain `plot()` calls on the same glyph instance. Reproduced empirically (see M1/M2).

**Triage** (from workflow step 3):

- **Subsystems touched**:
  - `src/cleopatra/styling/scaling.py` (new module — norm building) — behavior/API.
  - `src/cleopatra/styling/params.py` (new module — group objects) — behavior/API.
  - `src/cleopatra/glyphs/base/glyph.py` — API (loose-kwarg rejection, `_merge_group_params`, norm delegation).
  - `src/cleopatra/glyphs/**` (array, mesh, vector, flow, polygon, scatter, kde) — public API (plot/animate
    signatures), behavior.
  - `src/cleopatra/templates.py` — behavior (publication_map style forwarding).
  - `tests/**`, `docs/notebooks/**` — tests, docs.
- **High-risk surfaces**: public API break (all colour-mapped glyph constructors/plot signatures changed —
  this is an intentional breaking change, `feat!`). No auth/data/concurrency/migration/secret surfaces.
- **Cross-cutting concerns**: sticky `default_options` mutation via `_merge_group_params` and its error-path
  rollback consistency across glyphs (the main finding); norm-building parity after extraction.

# Findings

## Critical

None.

## High

None.

## Medium

### M1 — Failed `plot(classify=/color=/contour=)` bricks primitive glyph instances (no rollback)

- Files:
  - `src/cleopatra/glyphs/primitives/flow_glyph.py` (`plot`, merge at the top of the method)
  - `src/cleopatra/glyphs/primitives/vector_glyph.py` (`plot`)
  - `src/cleopatra/glyphs/primitives/scatter_glyph.py` (`plot`)
  - `src/cleopatra/glyphs/primitives/polygon_glyph.py` (`plot`)
  - root cause: `src/cleopatra/glyphs/base/glyph.py::Glyph._merge_group_params` (mutates the persistent
    `self._default_options` in place, lines ~586-590)
- Impact/risk: `_merge_group_params(color, contour, classify)` runs at the start of `plot()` and writes the
  group's keys into the glyph's persistent `default_options`. If the render later raises (e.g. `FlowGlyph`/
  `VectorGlyph` rejecting `scheme="categorical"`, or any glyph rejecting an unknown scheme name), the mutation
  is **not** rolled back. The invalid option stays in `default_options`, so a subsequent *plain* `plot()` on
  the same instance re-raises the same error — the instance is effectively bricked until the caller manually
  re-passes a valid group. KDE explicitly guards against exactly this (commit `33c196f`, snapshot/rollback of
  all co-passed group keys), but the four primitive glyphs got no equivalent guard.
- Reproduced:
  - `FlowGlyph(...).plot(classify=Classify(scheme="categorical"))` raises; `default_options["scheme"]` is left
    `"categorical"`; the next `g.plot()` raises `FlowGlyph does not support scheme='categorical'` although the
    caller passed nothing.
  - `ScatterGlyph(...).plot(classify=Classify(scheme="not_a_scheme"))` raises; `default_options["scheme"]` is
    left `"not_a_scheme"`; the next `g.plot()` re-raises `Unknown classification scheme 'not_a_scheme'`.
- Suggested fix: give the primitive glyphs the same snapshot/rollback KDE uses — snapshot the pre-merge value
  of every key each group's `to_options()` will touch, and restore them if the render raises (a shared helper
  on `Glyph`, e.g. a context manager wrapping the merge, would avoid duplicating the KDE logic four times).

### M2 — `ArrayGlyph` invalid-style rollback reverts only `style`, leaking co-passed `color`/`contour`/`cells`

- File: `src/cleopatra/glyphs/gridded/array_glyph.py` (`plot`, lines ~3449-3455)
- Impact/risk: On an invalid `data_style`, `ArrayGlyph.plot` rolls back with
  `self.default_options["style"] = None; raise` — it reverts only `style`. Any group options co-passed in the
  same call (`color=ColorScaling(...)`, `contour=Contour(...)`, `cells=CellValues(...)`) were already merged
  into the persistent `default_options` and are left in place. A later plain `plot()` then silently renders
  with the wrong (never-successfully-applied) colour scale / discretisation. This is the exact scenario KDE's
  `test_failed_data_style_rolls_back_co_passed_color` was added to prevent (commit `33c196f`), but `ArrayGlyph`
  was not given the same full rollback.
- Reproduced:
  `ArrayGlyph(...).plot(color=ColorScaling.power(gamma=0.7), data_style=DataStyle(style="not_a_style"))` raises;
  afterwards `default_options["style"]` is correctly `None`, but `color_scale` is left `"power"` and `gamma`
  `0.7` — the failed call silently changed the sticky colour scale.
- Suggested fix: roll back the full set of co-merged group keys (same snapshot approach as M1/KDE), not just
  `style`. Add an `ArrayGlyph` test mirroring KDE's `test_failed_data_style_rolls_back_co_passed_color`.

## Low

### L1 — `build_norm` dispatch turns an unmapped `ColorScale` member into an opaque `KeyError`

- File: `src/cleopatra/styling/scaling.py` (`build_norm`, `builder = self._NORM_BUILDERS[self.kind]`, line 380)
- Impact/risk: The pre-refactor code ended its if/elif chain with an explicit
  `else: raise ValueError("No norm branch implemented for color_scale=...")` (marked `pragma: no cover`). The
  dispatch-table rewrite replaces that with a bare `self._NORM_BUILDERS[self.kind]` lookup. All 5 current
  `ColorScale` members are mapped, so this is unreachable today — but if a future `ColorScale` member is added
  without a matching builder entry, the failure is now an unguided `KeyError: <ColorScale.NEW: ...>` instead of
  the previous actionable message. Behaviour for existing members is identical.
- Suggested fix: keep an explicit guard, e.g. `builder = self._NORM_BUILDERS.get(self.kind)` with a
  `raise ValueError(...)` when `None`, preserving the actionable error for future members.

### L2 — `publication_map` can raise a duplicate-keyword `TypeError`

- File: `src/cleopatra/templates.py` (`publication_map`, `glyph.plot(data_style=data_style, **plot_kwargs)`)
- Impact/risk: `publication_map` now forwards `style` via an explicit `data_style=` argument while also
  splatting `**plot_kwargs`. If a caller passes both `style=...` and `data_style=...` (the latter absorbed into
  `plot_kwargs`), the call raises `TypeError: plot() got multiple values for keyword argument 'data_style'`
  instead of a clear message. Narrow/edge misuse, but the two ways to specify the same thing now collide.
- Suggested fix: either document that `data_style` is not an accepted `plot_kwargs` key here, or detect the
  collision and raise a clear error (or let an explicit `data_style` in `plot_kwargs` win over `style`).

## Nit

None.

# Tests

Run with the external uv env
(`VIRTUAL_ENV=C:/python-environments/uv/cleopatra PYTHONPATH=src .../python.exe -m pytest ... -p no:cacheprovider`):

- `tests/test_scaling.py tests/test_mesh_glyph.py tests/test_kde_glyph.py tests/test_classify.py
  tests/test_categorize.py` — **320 passed**.
- `tests/test_array_glyph.py tests/test_flow_glyph.py tests/test_polygon_glyph.py tests/test_scatter_glyph.py
  tests/test_vector_glyph.py tests/test_glyph.py tests/test_colorbar_glyphs.py tests/test_review_fixes.py
  tests/test_styles.py tests/test_array_glyph_projection.py` — **903 passed**.
- Doctests: `pytest --doctest-modules src/cleopatra/styling/scaling.py src/cleopatra/styling/params.py` —
  **14 passed**.

Added/updated tests reviewed:

- `tests/test_scaling.py` — new coverage of the group objects' `to_options()` full-emit behaviour.
- The `pytest.raises` setup-hoisting refactors (`test_array_glyph.py`, `test_categorize.py`,
  `test_classify.py`, `test_kde_glyph.py`) — verified the hoisted constructor calls
  (`DataStyle(...)`, `Classify(...)`, `ColorScaling.power(...)`) do not themselves raise, so isolating the
  single raising `plot()` call preserves each test's intent.

Test gaps / specific missing tests:

- No test asserts that a **failed** `plot(classify=...)` / `plot(color=...)` on the primitive glyphs
  (`Flow`/`Vector`/`Scatter`/`Polygon`) leaves `default_options` clean for a later plain `plot()` (M1).
- No `ArrayGlyph` analogue of KDE's `test_failed_data_style_rolls_back_co_passed_color`, which would have
  caught M2.

# Questions and Assumptions

- Assumed the base of the diff is `main` and reviewed the merge-base range `main...HEAD` (workflow step 2a).
- Assumed the ndarray-`bounds` truthiness edge in `_boundary_norm` (`if self.bounds:` raises on an ndarray) is
  out of scope for this round: it is unchanged behaviour carried over from the earlier `ColorScaling`
  extraction (commit `f02899f`, covered by the round-2 review), and the `boundary()` constructor annotates
  `bounds` as `list[float]`.
- Is the primitive-glyph poisoning (M1) considered acceptable because the trigger is a prior user error, or
  should it get the same rollback KDE received? The KDE `fix` commit suggests the project treats this class of
  bug as worth guarding.

# Residual Risks

- M1/M2 are the only correctness issues; both are error-path state-leak/poisoning bugs on glyph reuse, not
  data-loss or security. They do not block a first successful render but degrade the reuse-after-error path.
- The three recent refactors (build_norm dispatch, extend unnest, test hoisting) introduced no behavioural
  change — verified by line-by-line reading plus the full impacted suite and doctests.
- Public API break is intentional (`feat!`) and consistently applied across glyphs; docs/notebooks were
  migrated in the same PR.

# Issue Tracker

| ID | Severity | State | Description | File(s) |
|----|----------|-------|-------------|---------|
| M1 | Medium | Solved | Failed `plot()` with a group object poisons primitive glyphs' `default_options` (no rollback), bricking later plain `plot()`. Fixed by a frame-free `_rollback_options_on_error` context manager on `Glyph`, wrapping each primitive's render body (commit `53e1a1a`); regression tests in `tests/test_group_rollback.py` | `src/cleopatra/glyphs/primitives/flow_glyph.py`, `vector_glyph.py`, `scatter_glyph.py`, `polygon_glyph.py`, `src/cleopatra/glyphs/base/glyph.py` |
| M2 | Medium | Solved | `ArrayGlyph` invalid-style rollback reverted only `style`, leaking co-passed `color`/`contour`/`cells`. Extended the style-validation rollback to restore every co-merged group key (commit `53e1a1a`); added `test_failed_style_rolls_back_co_passed_color` | `src/cleopatra/glyphs/gridded/array_glyph.py` |
| L1 | Low | Solved | `build_norm` dispatch yielded an opaque `KeyError` for a future unmapped `ColorScale` member; restored the actionable `ValueError` via `_NORM_BUILDERS.get()` (commit after `53e1a1a`) | `src/cleopatra/styling/scaling.py` |
| L2 | Low | Solved | `publication_map` could raise a duplicate-keyword `TypeError` when `style` and a `data_style` in `plot_kwargs` were both given; now pops `data_style` out, raises a clear `ValueError` on conflict, else lets an explicit `data_style` win | `src/cleopatra/templates.py` |
