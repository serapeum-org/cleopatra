"""Color presets.

- ``weather_presets.json`` -- colour data and parameter labels for GRIB
  shortName-keyed atmospheric presets, merged from ECMWF Magics and
  earthkit-plots.
- ``ocean_presets.json`` -- colour data for ocean/hydrology/DEM presets,
  derived from cmocean.

Kept as a package so the assets are importable via ``importlib.resources`` and
shipped in the wheel.
"""
