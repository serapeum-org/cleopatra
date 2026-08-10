"""Build the vendored radar/satellite meteorology colour-map preset asset.

Fetches the NWS/NOAA radar and satellite colour tables (BSD-licensed colour
*data*, no code) from a raw-text mirror and writes them to
``src/cleopatra/styling/data/radar_presets.json``. Run once on a maintainer
machine with a network connection; the package itself never fetches at import or
draw time -- it reads the shipped JSON, so there is **no runtime dependency**.

Each source ``.tbl`` is a plain list of ``(r, g, b)`` float triples in 0-1 (one
colour band per line). They render as discrete equal-width bands over the data's
own range (``colormap="listed"``, ``bands = colour count``); pass explicit
``vmin``/``vmax`` (e.g. a canonical dBZ scale) to pin the range.

Usage::

    python tools/build_radar_presets.py src/cleopatra/styling/data/radar_presets.json
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

from matplotlib.colors import to_hex

#: Raw-text mirror of the colour tables (one ``(r, g, b)`` float triple per line).
TBL_BASE = "https://raw.githubusercontent.com/Unidata/MetPy/main/src/metpy/plots/colortable_files"

#: One colour band per line, in either upstream table format:
#: an ``(r, g, b)`` float triple, e.g. ``(0.0, 0.92, 0.92)``, or a quoted hex
#: string with a trailing value comment, e.g. ``"#ccffff" # -30``.
_RGB = re.compile(r"\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*\)")
_HEX = re.compile(r"#[0-9a-fA-F]{6}")

#: Source table stem -> (descriptive preset key, legend label). Names are the
#: quantity, not the upstream table id.
TABLES = [
    ("NWSReflectivity", "reflectivity", "Radar reflectivity (dBZ)"),
    ("NWSReflectivityExpanded", "reflectivity_expanded", "Radar reflectivity, expanded"),
    ("NWSStormClearReflectivity", "reflectivity_clear_air", "Radar reflectivity, clear-air mode"),
    ("NWSVelocity", "radial_velocity", "Radar radial velocity"),
    ("NWS8bitVel", "radial_velocity_hires", "Radar radial velocity, high-res"),
    ("NWSSpectrumWidth", "spectrum_width", "Radar spectrum width"),
    ("precipitation", "precipitation_accumulation", "Precipitation accumulation"),
    ("ir_rgbv", "infrared", "Infrared satellite brightness temperature"),
    ("ir_bd", "infrared_enhanced", "Infrared satellite, enhanced"),
    ("WVCIMSS", "water_vapor", "Water-vapour satellite brightness temperature"),
    ("wv_tpc", "water_vapor_enhanced", "Water-vapour satellite, enhanced"),
]


def _fetch_colours(stem: str) -> list[str]:
    """Fetch one colour table as a list of hex strings (one per band)."""
    url = f"{TBL_BASE}/{stem}.tbl"
    with urllib.request.urlopen(url) as resp:  # noqa: S310 - fixed https maintainer source
        text = resp.read().decode("utf-8")
    out = []
    for line in text.splitlines():
        rgb = _RGB.search(line)
        if rgb is not None:
            out.append(to_hex(tuple(float(x) for x in rgb.groups())))
            continue
        hex_match = _HEX.search(line)  # the first `#rrggbb` (never the value comment)
        if hex_match is not None:
            out.append(to_hex(hex_match.group(0)))
    return out


def build() -> dict:
    presets = {}
    for stem, name, label in TABLES:
        colours = _fetch_colours(stem)
        presets[name] = {
            "layers": {
                name: {
                    "label": label,
                    "colors": colours,
                    "colormap": "listed",
                    "bands": len(colours),
                }
            }
        }
    return {
        "version": 1,
        "source": "nws",
        "license": "BSD-3-Clause",
        "presets": dict(sorted(presets.items())),
    }


def _safe_out_path(out_path: str) -> Path:
    """Resolve `out_path` and confine it to the current working directory tree."""
    base = Path.cwd().resolve()
    resolved = Path(out_path)
    resolved = resolved.resolve() if resolved.is_absolute() else (base / resolved).resolve()
    if resolved != base and base not in resolved.parents:
        raise ValueError(f"refusing to write outside {base}: {out_path!r}")
    return resolved


def main(out_path: str) -> None:
    asset = build()
    with open(_safe_out_path(out_path), "w", encoding="utf-8") as fh:
        json.dump(asset, fh, indent=1, ensure_ascii=False)
    print(f"wrote {len(asset['presets'])} radar/satellite presets to {out_path}")


if __name__ == "__main__":
    main(*sys.argv[1:])
