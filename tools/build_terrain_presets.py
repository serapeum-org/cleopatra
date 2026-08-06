"""Build the vendored Crameri terrain (hypsometric) preset asset.

Fetches a curated set of Fabio Crameri's *Scientific Colour Maps*
(https://www.fabiocrameri.ch/colourmaps/, MIT-licensed) from the ``cmcrameri``
mirror and writes them to ``src/cleopatra/data/terrain_presets.json``. Run once
on a maintainer machine with a network connection; the package itself never
fetches at import or draw time -- it reads the shipped JSON.

These are **hinge maps**: each has a hard land/sea break at the colour-bar
midpoint (0.5), ocean tones below and land tones above. Every record therefore
carries ``center=0`` (so ``apply_data_style`` renders it symmetric about sea
level with a ``TwoSlopeNorm``) and ``colormap="linear"`` (so the loader builds it
with a plain ``from_list`` that keeps the hinge at 0.5 -- the perceptual default
reparameterises by CIELAB arc-length and would drift the sea-level break; see
``cleopatra.colors._preset_cmap``).

Usage::

    python tools/build_terrain_presets.py src/cleopatra/data/terrain_presets.json

Provenance/attribution lives in ``THIRD_PARTY_NOTICES.md``.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

from matplotlib.colors import to_hex

#: Raw text mirror of the Scientific Colour Maps (256 rows of ``R G B`` floats
#: in 0-1). The upstream ``.txt`` files carry no header.
CMAP_BASE = "https://raw.githubusercontent.com/callumrollo/cmcrameri/main/cmcrameri/cmaps"

#: Number of evenly-spaced control points sampled from each 256-row map.
N_POINTS = 64

#: (Crameri map name, preset key, legend label). Each is a hypsometric map with
#: its land/sea hinge at the midpoint.
CURATED = [
    ("oleron", "elevation_oleron", "Elevation, land & sea (Crameri oleron)"),
    ("bukavu", "elevation_bukavu", "Elevation, fine relief (Crameri bukavu)"),
    ("fes", "elevation_fes", "Elevation, arid terrain (Crameri fes)"),
]


def _fetch_rows(name: str) -> list[tuple[float, float, float]]:
    """Fetch one Scientific Colour Map as a list of RGB triples in 0-1.

    Args:
        name: The upstream colormap name (e.g. ``"oleron"``).

    Returns:
        list: 256 ``(r, g, b)`` float triples in 0-1, top (deep) to peak.
    """
    url = f"{CMAP_BASE}/{name}.txt"
    with urllib.request.urlopen(url) as resp:  # noqa: S310 - fixed https maintainer source
        text = resp.read().decode("utf-8")
    rows: list[tuple[float, float, float]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        r, g, b = (float(p) for p in parts)
        rows.append((r, g, b))
    return rows


def _sample_palette(rows: list[tuple[float, float, float]]) -> list[str]:
    """Sample `N_POINTS` evenly-spaced hex control points from `rows`.

    Even index spacing keeps the native hinge at its own fraction (0.5 for these
    symmetric maps), which the ``colormap="linear"`` loader then preserves.

    Args:
        rows: The full-resolution ``(r, g, b)`` rows (0-1).

    Returns:
        list: `N_POINTS` ``#rrggbb`` strings.
    """
    n = len(rows)
    idx = [round(i * (n - 1) / (N_POINTS - 1)) for i in range(N_POINTS)]
    return [to_hex(rows[i]) for i in idx]


def _hinge_fraction(palette: list[str]) -> float:
    """Estimate the sea-level hinge fraction of a sampled palette.

    Locates the control point of steepest luminance change, which for these
    hypsometric maps is the shallow-water -> low-land break. Used only to sanity
    check that the hinge lands near 0.5 (a maintainer-facing diagnostic).

    Args:
        palette: The sampled ``#rrggbb`` control points.

    Returns:
        float: The fraction in 0-1 of the steepest-luminance-step control point.
    """

    def lum(hex_c: str) -> float:
        h = hex_c.lstrip("#")
        r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
        return 0.299 * r + 0.587 * g + 0.114 * b

    lums = [lum(c) for c in palette]
    steps = [abs(lums[i + 1] - lums[i]) for i in range(len(lums) - 1)]
    peak = max(range(len(steps)), key=steps.__getitem__)
    return peak / (len(palette) - 1)


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
    """Fetch the curated Crameri maps and write the terrain preset asset.

    Args:
        out_path: Destination JSON path (inside the repo tree).
    """
    presets: dict[str, dict] = {}
    for name, key, label in CURATED:
        rows = _fetch_rows(name)
        palette = _sample_palette(rows)
        hinge = _hinge_fraction(palette)
        if not 0.40 <= hinge <= 0.60:
            print(
                f"WARNING: {name} hinge at {hinge:.3f} is far from 0.5 -- "
                f"center=0 would misregister sea level",
                file=sys.stderr,
            )
        presets[key] = {"layers": {key: {
            "label": label,
            "colors": palette,
            "colormap": "linear",
            "center": 0.0,
            "alpha": 1.0,
        }}}
        print(f"{name}: {len(palette)} colours, hinge ~{hinge:.3f}")

    asset = {
        "version": 1,
        "source": "crameri",
        "license": "MIT",
        "presets": dict(sorted(presets.items())),
    }
    with open(_safe_out_path(out_path), "w", encoding="utf-8") as f:
        json.dump(asset, f, indent=1, ensure_ascii=False)
    print(f"wrote {len(presets)} Crameri terrain presets to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python build_terrain_presets.py <out_path>", file=sys.stderr)
        raise SystemExit(2)
    main(sys.argv[1])
