"""Vendored data assets bundled with cleopatra.

Holds the vendored `DATA_STYLES` preset libraries and their attribution:

- ``magics_presets.json`` -- colour data and parameter labels derived from the
  ECMWF Magics style library (Apache-2.0); see ``MAGICS_NOTICE.txt`` and the
  bundled ``LICENSE-APACHE-2.0.txt``.
- ``cmocean_presets.json`` -- colour data derived from the cmocean colormap
  collection (MIT); see ``CMOCEAN_NOTICE.txt``.

Kept as a package so the assets are importable via ``importlib.resources`` and
shipped in the wheel.
"""
