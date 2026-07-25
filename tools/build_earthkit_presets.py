#!/usr/bin/env python3
"""Build cleopatra's ECMWF/earthkit preset asset (maintainer-side, offline).

This script is **not** part of the shipped package and is **not** run at import
or install time. It is the one-off, maintainer-machine tool that derives
``src/cleopatra/data/earthkit_presets.json`` from ECMWF's open **earthkit-plots**
style library (Apache-2.0):

    ecmwf/earthkit-plots : src/earthkit/plots/data/styles/auto-styles/<param>.yml

Each parameter YAML declares an ``optimal`` (default) ``Style`` variant giving
ECMWF's *actual* professional look for that field: a colormap (a matplotlib name
or an explicit colour list), discrete contour ``levels``, and an ``extend`` cap.
Those map directly onto cleopatra's ``DATA_STYLES`` cfg (``cmap``/``levels``/
``extend``), so ``style="2t"`` renders like an ECMWF chart (muted ``Spectral_r``
2 degC bands) instead of the vendored Magics rainbow.

Only a **curated set** of common parameters is vendored (see ``PARAMS``); extend
it to add more. Only ECMWF's colour *data* + level/label associations are
copied, never any earthkit code.

Re-run (from the repo root)::

    python tools/build_earthkit_presets.py src/cleopatra/data/earthkit_presets.json [<ref>]

``<ref>`` defaults to ``main``; the resolved ref and generation date are recorded
in the asset's ``_meta`` block.
"""
import datetime as _dt
import json
import re
import sys
import urllib.request

import yaml

RAW = (
    "https://raw.githubusercontent.com/ecmwf/earthkit-plots/{ref}"
    "/src/earthkit/plots/data/styles/auto-styles/{stem}.yml"
)

#: earthkit auto-style file stem -> (GRIB shortName cleopatra keys it by, label).
#: The `optimal` variant of each is vendored as the parameter's default style.
PARAMS = {
    "2t": ("2t", "2 m temperature"),
    "2t_dewpoint": ("2d", "2 m dewpoint temperature"),
    "composition_aod550": ("aod550", "Total aerosol optical depth at 550 nm"),
    "composition_duaod550": ("duaod550", "Dust aerosol optical depth at 550 nm"),
    "10u": ("10u", "10 m U wind component"),
    "10v": ("10v", "10 m V wind component"),
    "wind-speed-at-10m": ("10si", "10 m wind speed"),
    "total-precipitation": ("tp", "Total precipitation"),
    "cape": ("cape", "Convective available potential energy"),
}

_RANGE = re.compile(r"range\(\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")


def parse_levels(levels):
    """earthkit `levels` (a ``range(a, b, c)`` string or an explicit list) -> list | None."""
    if levels is None:
        return None
    if isinstance(levels, str):
        match = _RANGE.search(levels)
        if match is None:
            return None
        start, stop, step = (int(x) for x in match.groups())
        return list(range(start, stop, step))
    return [float(v) for v in levels]


def clean_colors(colors):
    """Keep a matplotlib cmap NAME as-is; normalise a colour LIST (#rrggbbaa -> #rrggbb)."""
    if not isinstance(colors, list):
        return colors  # a matplotlib colormap name, e.g. "Spectral_r"
    out = []
    for c in colors:
        if isinstance(c, str) and c.startswith("#") and len(c) == 9:
            out.append(c[:7])  # drop the (opaque) alpha byte
        else:
            out.append(c)
    return out


def fetch(ref, stem):
    with urllib.request.urlopen(RAW.format(ref=ref, stem=stem)) as r:
        return yaml.safe_load(r.read().decode("utf-8"))


def build(ref):
    presets, skipped = {}, []
    for stem, (short, label) in PARAMS.items():
        doc = fetch(ref, stem)
        opt = doc["styles"][doc["optimal"]]
        if opt.get("type") != "Style" or opt.get("colors") is None:
            skipped.append((short, stem))
            continue
        presets[short] = {
            "label": label,
            "colors": clean_colors(opt["colors"]),
            "levels": parse_levels(opt.get("levels")),
            "extend": opt.get("extend", "neither"),
            "units": opt.get("units"),
            "source_style": doc["optimal"],
        }
    return presets, skipped


def main(out_path, ref="main"):
    presets, skipped = build(ref)
    asset = {
        "_meta": {
            "source": "ecmwf/earthkit-plots",
            "source_ref": ref,
            "source_files": ["src/earthkit/plots/data/styles/auto-styles/<param>.yml"],
            "license": "Apache-2.0",
            "generated_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d"),
            "note": (
                "ECMWF's default parameter styles, from earthkit-plots' open style "
                "library (Apache-2.0). Each preset is the `optimal` Style variant: a "
                "colormap (matplotlib name or colour list) + discrete contour `levels` "
                "+ `extend` cap -- the professional weather-service look. `units` names "
                "the unit the levels assume (cleopatra does not convert). Only colour "
                "data and level/label associations are vendored, no earthkit code."
            ),
        },
        "presets": dict(sorted(presets.items())),
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(asset, fh, indent=1, ensure_ascii=False)
    print(f"wrote {len(presets)} earthkit presets to {out_path}; skipped {len(skipped)}")
    for short, stem in skipped:
        print(f"  skipped {short} ({stem}): optimal variant is not a shadable Style")
    # Surface curated params whose optimal variant declares no contour levels
    # (or a spec parse_levels can't read): they load as a continuous auto-ranged
    # ramp, not discrete bands -- flag it so it is a deliberate choice, not silent.
    no_levels = [short for short, rec in presets.items() if rec["levels"] is None]
    if no_levels:
        print(
            f"  NOTE: {len(no_levels)} preset(s) have no contour levels "
            f"(continuous, auto-ranged): {', '.join(sorted(no_levels))}"
        )


if __name__ == "__main__":
    main(*sys.argv[1:])
