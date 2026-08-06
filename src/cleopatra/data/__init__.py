"""Color presets.

- ``weather_presets.json`` -- colour data and parameter labels for GRIB
  shortName-keyed atmospheric presets, merged from ECMWF Magics and
  earthkit-plots.
- ``ocean_presets.json`` -- colour data for ocean/hydrology/DEM presets,
  derived from cmocean.

See ``THIRD_PARTY_NOTICES.md`` at the repository root for the upstream sources
and their licenses (Magics / earthkit-plots Apache-2.0, cmocean MIT).

Kept as a package so the assets are importable via ``importlib.resources`` and
shipped in the wheel.
"""
