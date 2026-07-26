#!/usr/bin/env python3
"""Build cleopatra's merged weather preset asset (maintainer-side, offline).

This script is **not** part of the shipped package and is **not** run at import
or install time. It is the one-off, maintainer-machine tool that derives
``src/cleopatra/data/weather_presets.json`` from two open ECMWF sources, keyed
by GRIB shortName:

- **ecmwf/magics** (Apache-2.0): the full operational parameter library, each
  style's discrete colour bands resolved from ``share/magics/styles/default/
  palettes.json`` + ``contours.json`` + the named-colour table in
  ``src/common/Colour.cc``. Every Magics preset renders as equal-width bands
  (``bands`` = its colour count) over a fixed ``vmin``/``vmax`` decoded from its
  style name's ``f<from>t<to>[i<interval>]`` grammar, when the name carries one.
- **ecmwf/earthkit-plots** (Apache-2.0): a curated subset of parameters using
  earthkit's ``optimal`` style variant -- ECMWF's actual professional look
  (explicit, non-uniform contour ``levels`` + an ``extend`` cap, or a genuine
  continuous ramp when a parameter has no discrete levels).

Where both sources cover the same shortName, the earthkit-plots record is kept
and the Magics one dropped (see ``EARTHKIT_OVERRIDE_KEYS``) -- earthkit's is the
professional weather-service look; only one record per shortName ships.

Only each source's colour *data* and parameter/label associations are vendored,
never any Magics or earthkit-plots code.

Maintainer dependencies: ``matplotlib`` (already a cleopatra dependency, used to
resolve Magics' named colours) and ``PyYAML`` (``import yaml``, used to parse
the earthkit style files) -- neither is a cleopatra runtime dependency; install
PyYAML in the maintainer environment before re-running.

Re-run (from the repo root)::

    python tools/build_weather_presets.py src/cleopatra/data/weather_presets.json [<magics_ref>] [<earthkit_ref>]

``<magics_ref>`` defaults to ``develop``, ``<earthkit_ref>`` defaults to ``main``.
"""

import colorsys
import json
import re
import sys
import urllib.request
from pathlib import Path

import yaml
from matplotlib.colors import to_hex, to_rgba

# Opener restricted to http(s): only the HTTP(S) handlers are registered (plus
# the supporting redirect/error-processing handlers `urlopen` needs), so a
# file:///ftp:///data: URL structurally cannot be opened through it -- no
# handler claims it.
_HTTP_ONLY_OPENER = urllib.request.OpenerDirector()
for _handler in (
    urllib.request.HTTPHandler(),
    urllib.request.HTTPSHandler(),
    urllib.request.HTTPErrorProcessor(),
    urllib.request.HTTPRedirectHandler(),
    urllib.request.HTTPDefaultErrorHandler(),
    urllib.request.UnknownHandler(),
):
    _HTTP_ONLY_OPENER.add_handler(_handler)

#: GRIB shortNames earthkit-plots' curated defaults supersede -- dropped from
#: the Magics side so the merged file carries exactly one record per shortName.
EARTHKIT_OVERRIDE_KEYS = {"2d", "2t", "aod550", "duaod550", "tp"}

# --- Magics -----------------------------------------------------------------

MAGICS_BASE_TEMPLATE = (
    "https://raw.githubusercontent.com/ecmwf/magics/{ref}/share/magics/styles"
)
MAGICS_REPO_TEMPLATE = "https://raw.githubusercontent.com/ecmwf/magics/{ref}"

#: Matches a Magics named-colour definition in ``src/common/Colour.cc``:
#: ``colours_["greenish_blue"] = Rgb(0.0000, 0.5000, 1.0000);`` (float 0-1
#: components; the ``undefined`` sentinel is ``Rgb(-1., -1., -1.)``).
_RGB_DEF = re.compile(
    r'colours_\["([^"]+)"\]\s*=\s*Rgb\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)'
)

#: An ECMWF Magics style name encodes its contour range in an
#: ``f<from>t<to>[i<interval>]`` grammar (``M`` = minus). The interval is
#: optional and, once decoded, unused (see `transform_magics`) -- only the
#: range matters for the merged asset.
_MAGICS_RANGE = re.compile(r"f(M?\d+(?:_\d+)?)t(M?\d+(?:_\d+)?)(?:i(\d+(?:_\d+)?))?")


def _magics_num(token):
    """Decode one numeric token of a Magics range name.

    The grammar: ``M`` prefixes a negative; ``_`` is the decimal point
    (``1_5`` -> 1.5); a leading ``0`` on a multi-digit token marks a decimal
    (``05`` -> 0.5, ``01`` -> 0.1); ``0`` alone is zero; any other token is an
    integer (``10`` -> 10, not 1.0).
    """
    negative = token.startswith("M")
    if negative:
        token = token[1:]
    if "_" in token:
        value = float(token.replace("_", "."))
    elif len(token) > 1 and token[0] == "0":
        value = float("0." + token[1:])
    else:
        value = float(token)
    return -value if negative else value


def decode_magics_range(magics_style):
    """A Magics style name -> ``(vmin, vmax)``, or ``None`` if it carries no usable range."""
    match = _MAGICS_RANGE.search(magics_style or "")
    if match is None:
        return None
    vmin = _magics_num(match.group(1))
    vmax = _magics_num(match.group(2))
    if vmin >= vmax:
        return None
    return vmin, vmax


def fetch_magics(base, path):
    with _HTTP_ONLY_OPENER.open(f"{base}/{path}") as r:
        return json.load(r)


def fetch_magics_colours(magics_ref):
    """Magics' named-colour table (name -> hex), parsed from ``src/common/Colour.cc``."""
    url = MAGICS_REPO_TEMPLATE.format(ref=magics_ref) + "/src/common/Colour.cc"
    with _HTTP_ONLY_OPENER.open(url) as r:
        src = r.read().decode("utf-8", "replace")
    table = {}
    for name, red, green, blue in _RGB_DEF.findall(src):
        rgb = (float(red), float(green), float(blue))
        if min(rgb) < 0:  # the 'undefined' sentinel
            continue
        table[name] = to_hex(rgb)
    return table


def _parse_rgb(c):
    """A Magics ``rgb()``/``rgba()`` string -> (hex, alpha|None), or None."""
    is_float = "." in c
    vals = [float(x) for x in re.findall(r"[\d.]+", c.replace(" ", ""))]
    if len(vals) < 3:
        return None
    rgb, alpha = vals[:3], (vals[3] if len(vals) > 3 else None)
    out = [
        min(255, max(0, round(v * 255) if (is_float and v <= 1.0) else round(v)))
        for v in rgb
    ]
    return "#{:02x}{:02x}{:02x}".format(*out), (
        round(alpha, 4) if alpha is not None else None
    )


def parse_color(c, named_colours):
    """A Magics colour string -> (hex, alpha|None), or None if unresolvable."""
    c = c.strip()
    if c.lower().startswith(("rgb(", "rgba(")):
        return _parse_rgb(c)
    if c.lower().startswith("hsl("):
        nums = [float(x) for x in re.findall(r"[-\d.]+", c)]
        if len(nums) < 3:
            return None
        red, green, blue = colorsys.hls_to_rgb(nums[0] / 360.0, nums[2], nums[1])
        return to_hex((red, green, blue)), None
    if c in named_colours:
        return named_colours[c], None
    try:
        r, g, b, a = to_rgba(c)
        return to_hex((r, g, b)), (round(float(a), 4) if a < 1.0 else None)
    except (ValueError, TypeError):
        return None


def build_magics(magics_ref):
    """Fetch + resolve every Magics parameter style -> (presets, skipped, unresolved)."""
    base = MAGICS_BASE_TEMPLATE.format(ref=magics_ref)
    palettes = fetch_magics(base, "default/palettes.json")
    contours = fetch_magics(base, "default/contours.json")
    named_colours = fetch_magics_colours(magics_ref)

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


# --- earthkit-plots -----------------------------------------------------------

EARTHKIT_RAW = (
    "https://raw.githubusercontent.com/ecmwf/earthkit-plots/{ref}"
    "/src/earthkit/plots/data/styles/auto-styles/{stem}.yml"
)

#: earthkit auto-style file stem -> (GRIB shortName cleopatra keys it by, label).
#: The `optimal` variant of each is vendored as the parameter's default style.
EARTHKIT_PARAMS = {
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

_EARTHKIT_RANGE = re.compile(r"range\(\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")


def parse_levels(levels):
    """earthkit `levels` (a ``range(a, b, c)`` string or an explicit list) -> list | None."""
    if levels is None:
        return None
    if isinstance(levels, str):
        match = _EARTHKIT_RANGE.search(levels)
        if match is None:
            return None
        start, stop, step = (int(x) for x in match.groups())
        return list(range(start, stop, step))
    return [float(v) for v in levels]


def clean_colors(colors):
    """Keep a matplotlib cmap NAME as-is; normalise a colour LIST (#rrggbbaa -> #rrggbb)."""
    if not isinstance(colors, list):
        return colors
    out = []
    for c in colors:
        if isinstance(c, str) and c.startswith("#") and len(c) == 9:
            out.append(c[:7])
        else:
            out.append(c)
    return out


def fetch_earthkit(ref, stem):
    with _HTTP_ONLY_OPENER.open(EARTHKIT_RAW.format(ref=ref, stem=stem)) as r:
        return yaml.safe_load(r.read().decode("utf-8"))


def build_earthkit(ref):
    """Fetch + resolve every curated earthkit parameter style -> (presets, skipped)."""
    presets, skipped = {}, []
    for stem, (short, label) in EARTHKIT_PARAMS.items():
        doc = fetch_earthkit(ref, stem)
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
        }
    return presets, skipped


# --- merge --------------------------------------------------------------------


def transform_magics(raw_presets):
    """Magics raw records (``palette``/``opacity``/``magics_style``) -> the merged schema.

    Renames ``palette`` to ``colors``; records the discrete band count
    (``bands``, always ``len(colors)`` -- every Magics preset renders as flat
    equal-width bands); decodes ``magics_style``'s ``f<from>t<to>`` range into
    plain ``vmin``/``vmax`` when the name carries one, dropping the encoded
    token itself (its optional interval is decoded but never used downstream,
    same as before); drops the shortNames earthkit-plots supersedes.
    """
    out = {}
    for key, rec in raw_presets.items():
        if key in EARTHKIT_OVERRIDE_KEYS:
            continue
        colors = rec["palette"]
        merged = {
            "label": rec["label"],
            "colors": colors,
            "opacity": rec["opacity"],
            "bands": len(colors),
        }
        rng = decode_magics_range(rec.get("magics_style"))
        if rng is not None:
            merged["vmin"], merged["vmax"] = rng
        out[key] = merged
    return out


def transform_earthkit(raw_presets):
    """Earthkit raw records -> the merged schema (adds the explicit opacity policy).

    Every earthkit-plots preset renders as an opaque full field.
    """
    out = {}
    for key, rec in raw_presets.items():
        merged = dict(rec)
        merged["opacity"] = "opaque"
        out[key] = merged
    return out


def _safe_out_path(out_path):
    """Resolve `out_path` and confine it to the current working directory tree.

    This is a maintainer-only CLI whose destination comes from `argv`, so
    canonicalise the path and reject anything that resolves outside the invoking
    directory (the repo) before opening it -- a stray or `..`-laden argument must
    not escape the tree (path traversal, CWE-22).
    """
    base = Path.cwd().resolve()
    resolved = Path(out_path)
    resolved = (
        resolved.resolve() if resolved.is_absolute() else (base / resolved).resolve()
    )
    if resolved != base and base not in resolved.parents:
        raise ValueError(f"refusing to write outside {base}: {out_path!r}")
    return resolved


def main(out_path, magics_ref="develop", earthkit_ref="main"):
    magics_presets, magics_skipped, unresolved = build_magics(magics_ref)
    earthkit_presets, earthkit_skipped = build_earthkit(earthkit_ref)
    merged = {
        **transform_magics(magics_presets),
        **transform_earthkit(earthkit_presets),
    }
    asset = {"presets": dict(sorted(merged.items()))}
    with open(_safe_out_path(out_path), "w", encoding="utf-8") as f:
        json.dump(asset, f, indent=1, ensure_ascii=False)
    print(
        f"wrote {len(merged)} weather presets to {out_path} "
        f"({len(magics_presets) - len(EARTHKIT_OVERRIDE_KEYS)} Magics + "
        f"{len(earthkit_presets)} earthkit, {len(magics_skipped)} Magics + "
        f"{len(earthkit_skipped)} earthkit skipped)"
    )
    if unresolved:
        print(
            f"WARNING: {len(unresolved)} Magics colour name(s) still unresolved "
            f"(palettes truncated): {', '.join(unresolved)}"
        )


if __name__ == "__main__":
    main(*sys.argv[1:])
