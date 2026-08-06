# Third-party notices

cleopatra bundles a small amount of third-party **colour data** (colormap control points, and per-parameter
colour / contour-level / label associations) under `src/cleopatra/data/`. **No third-party source code is vendored** —
only colour values and the parameter/label metadata needed to render them. Each vendored asset, its upstream source, and
that source's license are listed below. The maintainer-side derivation scripts live in `tools/` (they are not shipped
and are not run at install time).

Every preset asset is also **self-describing**: each carries a machine-readable `source` and `license` (at the asset
level, overridable per preset) alongside its `presets`, and validates against `src/cleopatra/data/preset.schema.json`.
This file is the human-readable summary; the assets themselves are the authoritative per-record provenance.

## `src/cleopatra/data/weather_presets.json`

Derived from two open ECMWF projects — colour data plus GRIB shortName / parameter-label / contour-level associations
only, never any code. See `tools/build_weather_presets.py` for the exact derivation.

- **ECMWF Magics** — Apache License 2.0 — <https://github.com/ecmwf/magics>
  The operational parameter colour bands, resolved from `share/magics/styles/default/palettes.json` +
  `contours.json` + the named-colour table in `src/common/Colour.cc`.

- **ECMWF earthkit-plots** — Apache License 2.0 — <https://github.com/ecmwf/earthkit-plots>
  A curated subset of the `optimal` per-parameter styles
  (`src/earthkit/plots/data/styles/auto-styles/*.yml`): a colormap name or colour list, discrete contour `levels`,
  and an `extend` cap. Where both sources cover the same shortName, the earthkit-plots record is kept.

## `src/cleopatra/data/ocean_presets.json`

Derived from — see `tools/build_ocean_presets.py` for the exact derivation.

- **cmocean** — MIT License — Copyright (c) 2015 Kristen M. Thyng — <https://github.com/matplotlib/cmocean>
  Perceptually-uniform oceanography colormaps, sampled to hex control points and paired with ocean / hydrology / DEM
  variable labels and an opacity policy.

## `src/cleopatra/data/terrain_presets.json`

Derived from — see `tools/build_terrain_presets.py` for the exact derivation.

- **Scientific Colour Maps (Fabio Crameri)** — MIT License — Copyright (c) 2023 Fabio Crameri —
  <https://www.fabiocrameri.ch/colourmaps/> (mirror: <https://github.com/callumrollo/cmcrameri>)
  The hypsometric hinge maps `oleron`, `bukavu`, and `fes`, sampled to hex control points and paired with elevation
  labels. Each has a land/sea break at the colour-bar midpoint, so the presets carry `center=0` and are loaded with a
  hinge-faithful linear interpolation.

## `src/cleopatra/data/ncl_presets.json`

Derived from — see `tools/build_ncl_presets.py` for the exact derivation.

- **NCL / MeteoSwiss colour tables** — public-domain colour specifications (NCAR Command Language) —
  <https://www.ncl.ucar.edu/Document/Graphics/ColorTables/> The MeteoSwiss operational stepped tables
  (`precip_11lev`, `precip_diff_12lev`, `temp_19lev`, `temp_diff_18lev`, `sunshine_9lev`, `hotcold_18lev`), parsed to
  hex control points and rendered as discrete `ListedColormap` bands.

## `src/cleopatra/data/builtin_presets.json`

cleopatra's own hand-authored presets (haze, flame, `temperature`/`anomaly`/`elevation`/…, the categorical hydrology
presets, and the NEXRAD `radar_reflectivity` scale). These are original cleopatra colour work or reference matplotlib's
built-in colormaps by name — with one vendored exception:

- **ECMWF Magics** — Apache License 2.0 — <https://github.com/ecmwf/magics>
  The `cams_aod` preset's colour scale is the Magics `sh_BuYlRd_aod` CAMS aerosol-optical-depth palette (colour values
  only). This record declares `source: "magics"` in the asset.

---

License texts: Apache-2.0 <https://www.apache.org/licenses/LICENSE-2.0> · MIT <https://opensource.org/licenses/MIT>

_When a new colour source is vendored, add its source, URL, and license here in the same form._
