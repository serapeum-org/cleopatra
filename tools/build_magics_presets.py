#!/usr/bin/env python3
"""Build cleopatra's ECMWF/Magics preset asset (maintainer-side, offline).

This script is **not** part of the shipped package and is **not** run at
import or install time. It is the one-off, maintainer-machine tool that
derives ``src/cleopatra/data/magics_presets.json`` from the open ECMWF Magics
style data, following the chain that is fully recoverable from that data:

    parameter (contours.json: shortName / long_name)
      -> default style name (contours.json "style")
        -> palette          (palettes.json -- palettes are TAGGED with the style name)
          -> colours (rgb()/HSL()/named) + any alpha ramp

Magics is Apache-2.0; only its colour *data* and parameter/label associations
are vendored (see ``src/cleopatra/data/MAGICS_NOTICE.txt``), never its code.
Palette entries may be ``rgb()``/``HSL()`` values **or** Magics named colours
(``greenish_blue``, ``orangish_red``, ...); the names are resolved from Magics'
own colour table (``src/common/Colour.cc``) so the full ramp is kept -- dropping
the unrecognised names would truncate it and mis-weight the ends. The contour
range and interval are encoded in each style name (``f<from>t<to>[i<interval>]``,
``M`` = minus), vendored verbatim as each preset's ``magics_style``; cleopatra
decodes it at load time so presets render over ECMWF's fixed scale. The colour
list is recovered in full, but the exact per-level *boundary values* are not in
the open data -- so cleopatra lays the palette down as a discrete
``ListedColormap`` banded over ``[vmin, vmax]`` in equal-width intervals (a
``BoundaryNorm`` with one band per colour), reproducing the flat Magics
shaded-contour look rather than a smooth interpolation.

Maintainer dependencies: only ``matplotlib`` (already a cleopatra dependency),
used to resolve Magics' named colours to hex.

Re-run (from the repo root)::

    python tools/build_magics_presets.py src/cleopatra/data/magics_presets.json [<magics_ref>]

``<magics_ref>`` defaults to ``develop``; the resolved ref and generation date
are recorded in the asset's ``_meta`` block.
"""
import colorsys
import datetime as _dt
import json
import re
import sys
import urllib.request

from matplotlib.colors import to_hex, to_rgba

BASE_TEMPLATE = "https://raw.githubusercontent.com/ecmwf/magics/{ref}/share/magics/styles"
REPO_TEMPLATE = "https://raw.githubusercontent.com/ecmwf/magics/{ref}"

#: Matches a Magics named-colour definition in ``src/common/Colour.cc``:
#: ``colours_["greenish_blue"] = Rgb(0.0000, 0.5000, 1.0000);`` (float 0-1
#: components; the ``undefined`` sentinel is ``Rgb(-1., -1., -1.)``).
_RGB_DEF = re.compile(
    r'colours_\["([^"]+)"\]\s*=\s*Rgb\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)'
)


def fetch(base, path):
    with urllib.request.urlopen(f"{base}/{path}") as r:
        return json.load(r)


def fetch_magics_colours(magics_ref):
    """Magics' named-colour table (name -> hex), parsed from ``src/common/Colour.cc``.

    Magics palettes reference colours by name (``greenish_blue``, ``orangish_red``,
    ...) as well as by ``rgb()``. Those names are **not** matplotlib colours, so
    without this table they are silently dropped and the palette is truncated --
    e.g. the 27-colour ``2t`` temperature ramp collapses to 15, over-weighting its
    magenta cap. The definitions are ``colours_["name"] = Rgb(r, g, b);`` lines
    (components float 0-1); the ``undefined`` = -1 sentinel is skipped.
    """
    url = REPO_TEMPLATE.format(ref=magics_ref) + "/src/common/Colour.cc"
    with urllib.request.urlopen(url) as r:
        src = r.read().decode("utf-8", "replace")
    table = {}
    for name, red, green, blue in _RGB_DEF.findall(src):
        rgb = (float(red), float(green), float(blue))
        if min(rgb) < 0:  # the 'undefined' sentinel
            continue
        table[name] = to_hex(rgb)
    return table


def parse_color(c, named_colours):
    """A Magics colour string -> (hex, alpha|None), or None if unresolvable.

    Handles ``rgb()``/``rgba()`` (integer 0-255 or float 0-1 components,
    clamping the stray ``256`` some Magics entries carry -- an out-of-range
    data quirk), Magics *named* colours (from ``named_colours``, e.g.
    ``greenish_blue``), and finally matplotlib-named colours / bare hex. The
    Magics table takes precedence over matplotlib: a few names collide with a
    different value (Magics ``purple`` is magenta, not matplotlib's ``#800080``),
    and reproducing Magics faithfully means Magics wins.
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
    if c.lower().startswith("hsl("):
        # HSL(hue 0-360, sat 0-1, light 0-1); colorsys uses HLS component order.
        nums = [float(x) for x in re.findall(r"[-\d.]+", c)]
        if len(nums) < 3:
            return None
        red, green, blue = colorsys.hls_to_rgb(nums[0] / 360.0, nums[2], nums[1])
        return to_hex((red, green, blue)), None
    if c in named_colours:
        return named_colours[c], None
    # matplotlib-named colour or bare hex.
    try:
        r, g, b, a = to_rgba(c)
        return to_hex((r, g, b)), (round(float(a), 4) if a < 1.0 else None)
    except (ValueError, TypeError):
        return None


def build(magics_ref):
    base = BASE_TEMPLATE.format(ref=magics_ref)
    palettes = fetch(base, "default/palettes.json")
    contours = fetch(base, "default/contours.json")
    named_colours = fetch_magics_colours(magics_ref)

    # style name -> palette record, via the style names carried in palette tags.
    style_to_palette = {}
    for pval in palettes.values():
        for tag in pval.get("tags", []):
            style_to_palette.setdefault(str(tag), pval)

    presets, skipped, unresolved = {}, [], set()
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
            parsed = parse_color(c, named_colours)
            if parsed is None:
                # A colour we still cannot resolve would silently truncate the
                # palette (mis-weighting the ramp), so record it for the report
                # rather than dropping it unseen.
                unresolved.add(c.strip())
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
    return presets, skipped, sorted(unresolved)


def main(out_path, magics_ref="develop"):
    presets, skipped, unresolved = build(magics_ref)
    asset = {
        "_meta": {
            "source": "ecmwf/magics",
            "source_ref": magics_ref,
            "source_files": ["share/magics/styles/default/palettes.json",
                             "share/magics/styles/default/contours.json",
                             "src/common/Colour.cc"],
            "license": "Apache-2.0",
            "generated_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d"),
            "note": (
                "Colour data and parameter/label associations derived from ECMWF "
                "Magics (Apache-2.0); contains no Magics code. Palette colours are "
                "resolved from rgb() values and Magics' named-colour table "
                "(Colour.cc), so the full ramp is kept. The contour range and "
                "interval are encoded in each preset's magics_style name "
                "(f<from>t<to>[i<interval>], M=minus) and cleopatra decodes them at "
                "load time, so presets render over ECMWF's fixed scale (a caller "
                "vmin/vmax still overrides). The exact per-level boundary values are "
                "not in the open data, so the ramp is spread linearly across the "
                "range. Opacity is opaque unless the source palette carries a "
                "built-in alpha ramp, in which case it is an overlay."
            ),
        },
        "presets": dict(sorted(presets.items())),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(asset, f, indent=1, ensure_ascii=False)
    print(f"wrote {len(presets)} presets to {out_path}; skipped {len(skipped)} (no shade palette)")
    if unresolved:
        print(f"WARNING: {len(unresolved)} colour name(s) still unresolved "
              f"(palettes truncated): {', '.join(unresolved)}")


if __name__ == "__main__":
    main(*sys.argv[1:])
