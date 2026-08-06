"""Color presets.

- `weather_presets.json` -- colour data and parameter labels for GRIB
  shortName-keyed atmospheric presets, merged from ECMWF Magics and
  earthkit-plots.
- `ocean_presets.json` -- colour data for ocean/hydrology/DEM presets,
  derived from cmocean.
- `builtin_presets.json` -- cleopatra's hand-authored presets.
- `ncl_presets.json` -- NCL / MeteoSwiss stepped colour tables.
- `terrain_presets.json` -- Crameri terrain (hypsometric) presets.
- `preset.schema.json` -- JSON schema the preset assets validate against.

Kept as a package so the assets are importable via `importlib.resources` and
shipped in the wheel.
"""
