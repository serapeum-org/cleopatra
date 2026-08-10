"""Color presets.

- `weather_presets.json` -- colour data and parameter labels for GRIB
  shortName-keyed atmospheric presets, derived from ECMWF reference styles.
- `ocean_presets.json` -- colour data for ocean/hydrology/DEM presets,
  derived from cmocean.
- `builtin_presets.json` -- cleopatra's hand-authored presets.
- `ncl_presets.json` -- NCL / MeteoSwiss stepped colour tables.
- `terrain_presets.json` -- terrain (hypsometric) presets.
- `scientific_presets.json` -- perceptually-uniform scientific colour maps.
- `preset.schema.json` -- JSON schema the preset assets validate against.

Kept as a package so the assets are importable via `importlib.resources` and
shipped in the wheel.
"""
