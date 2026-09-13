# Solar Module — Day/Night Terminator & Tissot Indicatrix

The `cleopatra.basemap.solar` module is the flat-2D counterpart to the 3-D globe's directional
lighting ([`TexturedGlobeGlyph`](textured-globe-glyph.md), which shades a sphere and produces no
geometry). It has two layers:

1. **CRS-free solar geometry** — pure lon/lat maths that depends on nothing but the clock, with no
   ephemeris dependency:
      - `subsolar_point(when)` — the point where the sun is overhead (declination + the equation of
        time), via the low-precision NOAA/Meeus formulae (sub-degree over the relevant centuries).
      - `terminator(when, ...)` — the day/night terminator as a **small circle** at angular distance
        `90 - refraction` from the subsolar point (a great circle only when `refraction == 0`); pass
        `refraction=-6/-12/-18` for the civil/nautical/astronomical twilight lines.
      - `night_polygon(when, ...)` — the filled night region as lon/lat rings, closed along the dark
        pole's edge when a pole is in darkness and otherwise split at the antimeridian so a flat map
        does not get a band smeared across the whole world.
      - `tissot_circles(lons, lats, radius_m, ...)` — geodesic circles of a fixed ground radius, in
        lon/lat, for building a Tissot indicatrix.

2. **Artists that draw in data coordinates** — `add_nightshade` builds `night_polygon` and draws it
   as a `PolyCollection`, preserving the current axis limits and returning the artist; `add_tissot`
   draws a set of rings the caller supplies (already in the axes' coordinates).

**Scope boundary.** cleopatra computes lon/lat and *draws* — it never resolves a CRS or decides the
axes' projection. On a plain lon/lat axes the rings are drawn directly. On a projected axes the
consumer either pre-projects the vertices and passes them, or passes a `transform` callable
`(lon, lat) -> (x, y)`. The one convenience is an optional `crs=` shortcut (mutually exclusive with
`transform=`) that reprojects from EPSG:4326 through `pyproj` — the same optional
`cleopatra[tiles]` extra [`add_features`](reference-data.md) uses.

## Usage

```python
import matplotlib
matplotlib.use("Agg")  # any backend; Agg shown for headless rendering
from datetime import UTC, datetime

import matplotlib.pyplot as plt

from cleopatra.basemap.solar import add_nightshade, add_tissot, tissot_circles

when = datetime(2026, 6, 21, 12, tzinfo=UTC)

fig, ax = plt.subplots()
ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)

# Shade the night region on a plain lon/lat axes.
add_nightshade(ax, when, alpha=0.35, color="black", zorder=5)

# Tissot indicatrix: geodesic circles in lon/lat. On a lon/lat axes they can be
# drawn directly; on a projected axes, project each ring first (that distortion
# is exactly what the indicatrix shows).
circles = tissot_circles(lons=[-120, 0, 120], lats=[-45, 0, 45], radius_m=5e5)
add_tissot(ax, circles, facecolor="none", edgecolor="crimson")

fig.savefig("nightshade.png")
```

!!! note
    `add_nightshade` and `add_tissot` **draw in data coordinates and reproject nothing** unless you
    ask: pass a `transform` callable, or the optional `crs=` shortcut (which needs the
    `cleopatra[tiles]` extra and raises an actionable `ImportError` without it). `transform=` and
    `crs=` are mutually exclusive. The night region always spans about half the globe, so a
    non-global projection (orthographic, azimuthal, a regional CRS) — or a conformal projection at
    its poles — maps the far side outside the projection's domain, to non-finite coordinates.
    `add_nightshade` drops those vertices so the fill stays valid, but the drawn region is then only
    the part that lies inside the projection; choose a projection whose domain covers the area you
    are shading.

## Module Documentation

::: cleopatra.basemap.solar
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3
