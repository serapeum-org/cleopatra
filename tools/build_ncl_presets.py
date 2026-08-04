"""Build the vendored NCL / MeteoSwiss stepped colour-table preset asset.

Fetches a curated set of NCAR Command Language (NCL) colour tables -- the
MeteoSwiss operational weather tables re-hosted by NCL -- from
``ncl.ucar.edu`` and writes them to ``src/cleopatra/data/ncl_presets.json``.
Run once on a maintainer machine with a network connection; the package never
fetches at import or draw time -- it reads the shipped JSON.

The NCL ``.rgb`` files are plain text: ``#`` comment lines, an ``ncolors = N``
header, then ``R G B`` integer triples in 0-255 (each optionally followed by an
inline ``# label``). These are **stepped** tables -- a small number of flat
bands -- so every record carries ``interp="listed"``: the loader builds a
discrete ``ListedColormap`` (one band per colour) rather than a smooth ramp,
preserving the operational banded look. Anomaly tables additionally carry
``center=0`` so they render symmetric about zero.

Usage::

    python tools/build_ncl_presets.py src/cleopatra/data/ncl_presets.json

These tables are public-domain colour specifications (NCAR/NCL); provenance is
recorded in ``THIRD_PARTY_NOTICES.md``.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

#: Where NCL hosts the raw colour-table files.
RGB_BASE = "https://www.ncl.ucar.edu/Document/Graphics/ColorTables/Files"

#: (NCL table name, preset key, legend label, diverging center or None).
CURATED = [
    ("precip_11lev", "precipitation_steps", "Precipitation (MeteoSwiss)", None),
    ("precip_diff_12lev", "precipitation_anomaly", "Precipitation anomaly", 0.0),
    ("temp_19lev", "temperature_steps", "Temperature (MeteoSwiss)", None),
    ("temp_diff_18lev", "temperature_anomaly", "Temperature anomaly", 0.0),
    ("sunshine_9lev", "sunshine_hours", "Sunshine", None),
    ("hotcold_18lev", "hot_cold", "Hot / cold (diverging)", 0.0),
]


def _parse_rgb(text: str) -> list[str]:
    """Parse an NCL ``.rgb`` table body into ``#rrggbb`` control points.

    Ignores ``#`` comment lines and the ``ncolors = N`` header; strips any inline
    ``# label`` after a triple; accepts 0-255 integer (or float) ``R G B`` rows.

    Args:
        text: The raw ``.rgb`` file contents.

    Returns:
        list: The table's colours as ``#rrggbb`` strings, in file order.
    """
    colours: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()  # drop comments / inline labels
        if not line or line.lower().startswith("ncolors"):
            continue
        parts = line.split()
        if len(parts) != 3:
            continue
        try:
            r, g, b = (float(p) for p in parts)
        except ValueError:
            continue
        scale = 255.0 if max(r, g, b) > 1.0 else 1.0
        rgb = tuple(int(round(v / scale * 255)) for v in (r, g, b))
        colours.append("#{:02x}{:02x}{:02x}".format(*rgb))
    return colours


def _fetch_table(name: str) -> list[str]:
    """Fetch and parse one NCL colour table.

    Args:
        name: The upstream table name (e.g. ``"precip_11lev"``).

    Returns:
        list: The parsed ``#rrggbb`` control points.
    """
    url = f"{RGB_BASE}/{name}.rgb"
    with urllib.request.urlopen(url) as resp:  # noqa: S310 - fixed https maintainer source
        return _parse_rgb(resp.read().decode("utf-8"))


def _safe_out_path(out_path: str) -> Path:
    """Resolve `out_path` and confine it to the invoking directory tree.

    A maintainer-only CLI whose destination comes from ``argv``: canonicalise and
    reject anything resolving outside the repo (path traversal, CWE-22).

    Args:
        out_path: The requested output path.

    Returns:
        Path: The resolved, confined path.

    Raises:
        ValueError: If `out_path` resolves outside the current directory tree.
    """
    base = Path.cwd().resolve()
    resolved = Path(out_path)
    resolved = (
        resolved.resolve() if resolved.is_absolute() else (base / resolved).resolve()
    )
    if resolved != base and base not in resolved.parents:
        raise ValueError(f"refusing to write outside {base}: {out_path!r}")
    return resolved


def main(out_path: str) -> None:
    """Fetch the curated NCL tables and write the preset asset.

    Args:
        out_path: Destination JSON path (inside the repo tree).
    """
    presets: dict[str, dict] = {}
    for name, key, label, center in CURATED:
        palette = _fetch_table(name)
        if len(palette) < 2:
            print(f"WARNING: {name} yielded {len(palette)} colours, skipping", file=sys.stderr)
            continue
        rec: dict = {
            "label": label,
            "palette": palette,
            "opacity": "opaque",
            "interp": "listed",
        }
        if center is not None:
            rec["center"] = center
        presets[key] = rec
        print(f"{name}: {len(palette)} colours -> {key}")

    asset = dict(sorted(presets.items()))
    with open(_safe_out_path(out_path), "w", encoding="utf-8") as f:
        json.dump(asset, f, indent=1, ensure_ascii=False)
    print(f"wrote {len(presets)} NCL/MeteoSwiss presets to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python build_ncl_presets.py <out_path>", file=sys.stderr)
        raise SystemExit(2)
    main(sys.argv[1])
