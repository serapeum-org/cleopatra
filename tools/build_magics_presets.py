#!/usr/bin/env python3
"""Build cleopatra's ECMWF/Magics preset asset (maintainer-side, offline).

This script is **not** part of the shipped package and is **not** run at
import or install time. It is the one-off, maintainer-machine tool that
derives ``src/cleopatra/data/magics_presets.json`` from the open ECMWF Magics
style data, following the chain that is fully recoverable from that data:

    parameter (contours.json: shortName / long_name)
      -> default style name (contours.json "style")
        -> palette          (palettes.json -- palettes are TAGGED with the style name)
          -> hex colours + any alpha ramp

Magics is Apache-2.0; only its colour *data* and parameter/label associations
are vendored (see ``src/cleopatra/data/MAGICS_NOTICE.txt``), never its code.
The exact numeric contour levels are NOT in Magics' open data (they live in
ECMWF's style server), so the generated presets carry no ``vmin``/``vmax`` and
rely on cleopatra's auto-ranging at draw time.

Maintainer dependencies: only ``matplotlib`` (already a cleopatra dependency),
used to resolve Magics' named colours to hex.

Re-run (from the repo root)::

    python tools/build_magics_presets.py src/cleopatra/data/magics_presets.json [<magics_ref>]

``<magics_ref>`` defaults to ``develop``; the resolved ref and generation date
are recorded in the asset's ``_meta`` block.
"""
import datetime as _dt
import json
import re
import sys
import urllib.request

from matplotlib.colors import to_hex, to_rgba

BASE_TEMPLATE = "https://raw.githubusercontent.com/ecmwf/magics/{ref}/share/magics/styles"


def fetch(base, path):
    with urllib.request.urlopen(f"{base}/{path}") as r:
        return json.load(r)


def parse_color(c):
    """A Magics colour string -> (hex, alpha|None), or None if unparseable.

    Handles ``rgb()``/``rgba()`` (integer 0-255 or float 0-1 components,
    clamping the stray ``256`` some Magics entries carry -- an out-of-range
    data quirk) and matplotlib-named colours (e.g. ``"white"``, ``"navy"``).
    """
    c = c.strip()
    if c.lower().startswith(("rgb(", "rgba(")):
        is_float = "." in c
        nums = re.findall(r"[\d.]+", c.replace(" ", ""))
        vals = [float(x) for x in nums]
        if len(vals) < 3:
            return None
        rgb, alpha = vals[:3], (vals[3] if len(vals) > 3 else None)
        out = []
        for v in rgb:
            iv = round(v * 255) if (is_float and v <= 1.0) else round(v)
            out.append(min(255, max(0, iv)))
        return "#{:02x}{:02x}{:02x}".format(*out), (round(alpha, 4) if alpha is not None else None)
    # Named colour or bare hex -> let matplotlib resolve it.
    try:
        r, g, b, a = to_rgba(c)
        return to_hex((r, g, b)), (round(float(a), 4) if a < 1.0 else None)
    except (ValueError, TypeError):
        return None


def build(magics_ref):
    base = BASE_TEMPLATE.format(ref=magics_ref)
    palettes = fetch(base, "default/palettes.json")
    contours = fetch(base, "default/contours.json")

    # style name -> palette record, via the style names carried in palette tags.
    style_to_palette = {}
    for pval in palettes.values():
        for tag in pval.get("tags", []):
            style_to_palette.setdefault(str(tag), pval)

    presets, skipped = {}, []
    for entry in contours:
        crit = entry.get("criteria", {})
        short = crit.get("shortName")
        style = entry.get("style")
        if not short or not style:
            continue
        pal = style_to_palette.get(style)
        if pal is None:
            skipped.append((short, style))
            continue
        hexes, alphas = [], []
        for c in pal.get("values", []):
            parsed = parse_color(c)
            if parsed is None:
                continue
            hexes.append(parsed[0])
            alphas.append(parsed[1])
        if len(hexes) < 2:
            skipped.append((short, style))
            continue
        has_alpha = any(a is not None for a in alphas)
        presets[short] = {
            "label": crit.get("long_name") or short,
            "palette": hexes,
            "opacity": "overlay" if has_alpha else "opaque",
            "magics_style": style,
        }
    return presets, skipped


def main(out_path, magics_ref="develop"):
    presets, skipped = build(magics_ref)
    asset = {
        "_meta": {
            "source": "ecmwf/magics",
            "source_ref": magics_ref,
            "source_files": ["share/magics/styles/default/palettes.json",
                             "share/magics/styles/default/contours.json"],
            "license": "Apache-2.0",
            "generated_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d"),
            "note": (
                "Colour data and parameter/label associations derived from ECMWF "
                "Magics (Apache-2.0); contains no Magics code. Exact contour levels "
                "are not in the open data, so presets carry no vmin/vmax and "
                "auto-range. Opacity is opaque unless the source palette carries a "
                "built-in alpha ramp, in which case it is an overlay."
            ),
        },
        "presets": dict(sorted(presets.items())),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(asset, f, indent=1, ensure_ascii=False)
    print(f"wrote {len(presets)} presets to {out_path}; skipped {len(skipped)} (no shade palette)")


if __name__ == "__main__":
    main(*sys.argv[1:])
