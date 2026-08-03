# Third-party notices

cleopatra bundles a small amount of third-party **colour data** (colormap control points, and per-parameter
colour / contour-level / label associations) under `src/cleopatra/data/`. **No third-party source code is vendored** —
only colour values and the parameter/label metadata needed to render them. Each vendored asset, its upstream source, and
that source's license are listed below. The maintainer-side derivation scripts live in `tools/` (they are not shipped
and are not run at install time).

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

---

License texts: Apache-2.0 <https://www.apache.org/licenses/LICENSE-2.0> · MIT <https://opensource.org/licenses/MIT>

_When a new colour source is vendored, add its source, URL, and license here in the same form._
