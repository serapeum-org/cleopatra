"""Day/night terminator and Tissot-indicatrix artists for flat 2-D axes.

cleopatra can already shade a day/night terminator, but only as *lighting on a
3-D sphere* (`cleopatra.glyphs.globe.textured_globe_glyph.TexturedGlobeGlyph`):
the input is a world-space light *direction*, the output is per-face shaded
*facecolors* on an `Axes3D`, and no lon/lat *geometry* is ever produced. This
module is the flat-map counterpart: it computes the terminator as a great circle
in lon/lat and draws it (and the filled night region, and Tissot distortion
circles) on an ordinary `matplotlib.axes.Axes`.

Scope boundary -- the same split the rest of ``basemap`` keeps:

- **Solar geometry is CRS-free maths, so it lives here.** The subsolar point for
  a datetime (solar declination plus the Greenwich hour angle via the equation
  of time) and the terminator great circle depend on nothing but the clock; they
  are not CRS-dependent. Likewise the geodesic circles behind a Tissot
  indicatrix are generic spherical geometry.
- **Any CRS/projection transform is the consumer's.** These helpers compute
  lon/lat and draw in *data coordinates*. A consumer whose axes is not lon/lat
  either pre-projects the vertices and passes them, or passes a ``transform``
  callable ``(lon, lat) -> (x, y)``. This module never resolves a CRS, never
  decides the axes' projection, and never grows a projection engine -- the same
  contract `cleopatra.basemap.projection.apply_projection_frame` states.
- **The one convenience is an optional ``crs=``**, mirroring
  `cleopatra.basemap.reference.add_features`: it reprojects from EPSG:4326
  through pyproj (the ``[tiles]`` extra) for the common EPSG case, and is
  mutually exclusive with ``transform=``. It is a shortcut, not a licence to own
  projections.

Accuracy is deliberately low-precision: the NOAA/Meeus solar-position formulae
are sub-degree over the relevant centuries, which is far finer than a shaded
overlay needs. No ephemeris dependency.

Example (lon/lat axes)::

    from datetime import datetime, timezone
    from cleopatra.basemap.solar import add_nightshade

    when = datetime(2026, 6, 21, 12, tzinfo=timezone.utc)
    add_nightshade(ax, when, alpha=0.35, color="black", zorder=5)

See also `cleopatra.glyphs.globe.textured_globe_glyph` for the 3-D globe's
directional lighting, which answers a different question (shading on a sphere,
not geometry on a flat map).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Any

import numpy as np
from matplotlib.collections import PolyCollection

#: Solar altitude (degrees) that defines the terminator. The standard value of
#: ``-0.83`` accounts for atmospheric refraction plus the sun's semidiameter at
#: sunrise/sunset. Pass ``-6`` / ``-12`` / ``-18`` for the civil / nautical /
#: astronomical twilight lines instead.
DEFAULT_REFRACTION: float = -0.83

#: Number of samples along the terminator great circle / night-region rings.
DEFAULT_TERMINATOR_SAMPLES: int = 720

#: Number of vertices per Tissot circle.
DEFAULT_TISSOT_SAMPLES: int = 64

#: Mean Earth radius in metres (IUGG), used to turn a ground radius in metres
#: into an angular radius for the geodesic Tissot circles.
MEAN_EARTH_RADIUS_M: float = 6_371_008.8


# --------------------------------------------------------------------------- #
# CRS-free solar geometry (no matplotlib needed)                              #
# --------------------------------------------------------------------------- #
def subsolar_point(when: datetime) -> tuple[float, float]:
    """Return the subsolar point ``(lon, lat)`` in degrees for ``when``.

    The subsolar point is where the sun is directly overhead: its latitude is
    the solar declination, its longitude is derived from the Greenwich hour
    angle (apparent solar time via the equation of time). Pure maths -- no
    matplotlib import needed.

    Args:
        when: An aware `datetime.datetime`. A naive datetime is treated as UTC.

    Returns:
        tuple[float, float]: ``(lon, lat)`` in degrees, ``lon`` in
        ``(-180, 180]`` and ``lat`` in ``[-90, 90]``.
    """
    # TODO(#356): NOAA/Meeus low-precision solar position -> declination + eqn of
    # time -> subsolar (lon, lat). ~20 lines, no dependency. See issue #356.
    raise NotImplementedError("subsolar_point is not implemented yet (see #356).")


def terminator(
    when: datetime,
    *,
    refraction: float = DEFAULT_REFRACTION,
    n: int = DEFAULT_TERMINATOR_SAMPLES,
) -> np.ndarray:
    """Return the day/night terminator great circle as lon/lat vertices.

    The terminator is the great circle 90 degrees (offset by ``refraction``)
    from the subsolar point.

    Args:
        when: An aware `datetime.datetime` (naive is treated as UTC).
        refraction: Solar altitude in degrees defining the terminator; see
            `DEFAULT_REFRACTION`. Use ``-6`` / ``-12`` / ``-18`` for civil /
            nautical / astronomical twilight.
        n: Number of samples along the circle.

    Returns:
        numpy.ndarray: An ``(n, 2)`` array of ``(lon, lat)`` degrees, densified
        for a smooth curve.
    """
    # TODO(#356): great circle at (90 - refraction) from subsolar_point(when),
    # densified to n points, returned as (n, 2) lon/lat.
    raise NotImplementedError("terminator is not implemented yet (see #356).")


def night_polygon(
    when: datetime,
    *,
    refraction: float = DEFAULT_REFRACTION,
    n: int = DEFAULT_TERMINATOR_SAMPLES,
) -> list[np.ndarray]:
    """Return the filled night region as lon/lat rings.

    The rings are pre-split at the antimeridian so that a flat (lon/lat) map
    does not get a band smeared across the whole world when the night region
    wraps.

    Args:
        when: An aware `datetime.datetime` (naive is treated as UTC).
        refraction: Solar altitude in degrees; see `DEFAULT_REFRACTION`.
        n: Number of samples along the terminator used to build the region.

    Returns:
        list[numpy.ndarray]: One or more ``(m, 2)`` lon/lat rings covering the
        night side; more than one where the region crosses the antimeridian.
    """
    # TODO(#356): close the terminator into the night hemisphere, split any ring
    # crossing +/-180 longitude into separate rings so a flat map fills cleanly.
    raise NotImplementedError("night_polygon is not implemented yet (see #356).")


# --------------------------------------------------------------------------- #
# Artists: draw in data coordinates                                           #
# --------------------------------------------------------------------------- #
def add_nightshade(
    ax: Any,
    when: datetime,
    *,
    refraction: float = DEFAULT_REFRACTION,
    n: int = DEFAULT_TERMINATOR_SAMPLES,
    transform: Callable[[np.ndarray], np.ndarray] | None = None,
    crs: int | str | None = None,
    **style: Any,
) -> PolyCollection:
    """Shade the night region on ``ax`` and return the artist.

    Builds the `night_polygon` rings, maps them into the axes' coordinates,
    draws them, preserves the current axis limits (the way
    `cleopatra.basemap.reference.add_features` does), and returns the artist so
    the caller can restyle or remove it. Several calls on one axes are fine;
    each returns its own artist.

    Coordinate mapping is the consumer's responsibility:

    - Neither ``transform`` nor ``crs``: the rings are drawn as lon/lat, i.e.
      the axes is assumed to be in EPSG:4326 data coordinates.
    - ``transform``: a callable applied to each ``(m, 2)`` lon/lat ring,
      returning ``(m, 2)`` axes coordinates.
    - ``crs``: the optional pyproj shortcut -- reproject from EPSG:4326 to
      ``crs`` (reuses `cleopatra.basemap.reference._make_transformer`, so it
      needs the ``[tiles]`` extra).

    Args:
        ax: A matplotlib `~matplotlib.axes.Axes` with data already plotted.
        when: An aware `datetime.datetime` (naive is treated as UTC).
        refraction: Solar altitude in degrees; see `DEFAULT_REFRACTION`.
        n: Number of samples along the terminator.
        transform: Optional ``(lon, lat) -> (x, y)`` callable operating on
            ``(m, 2)`` arrays. Mutually exclusive with ``crs``.
        crs: Optional target CRS (EPSG int or CRS string) for the pyproj
            shortcut. Mutually exclusive with ``transform``.
        **style: Style overrides forwarded to the `PolyCollection`
            (``facecolor``/``color``, ``alpha``, ``zorder``, ...).

    Returns:
        matplotlib.collections.PolyCollection: The night-shade artist.

    Raises:
        ValueError: If both ``transform`` and ``crs`` are given.
        ImportError: If ``crs`` requires reprojection but ``pyproj`` (the
            ``[tiles]`` extra) is not installed.
    """
    # TODO(#356): reject transform+crs together; build night_polygon(when, ...);
    # apply transform() or reference._make_transformer(crs)/_reproject_arr; add a
    # PolyCollection on ax.transData; save/restore get_xlim()/get_ylim(); return it.
    raise NotImplementedError("add_nightshade is not implemented yet (see #356).")


def tissot_circles(
    lons: Sequence[float] | np.ndarray,
    lats: Sequence[float] | np.ndarray,
    radius_m: float,
    *,
    n: int = DEFAULT_TISSOT_SAMPLES,
) -> list[np.ndarray]:
    """Return geodesic circles of a fixed ground radius as lon/lat rings.

    Generic spherical geometry: each circle is the locus of points a fixed
    great-circle distance (``radius_m`` on a sphere of `MEAN_EARTH_RADIUS_M`)
    from its centre. No CRS is involved; the consumer projects these rings to
    show how a projection distorts them (that is the Tissot indicatrix).

    Args:
        lons: Circle-centre longitudes in degrees.
        lats: Circle-centre latitudes in degrees. Same length as ``lons``.
        radius_m: Ground radius of each circle in metres.
        n: Number of vertices per circle.

    Returns:
        list[numpy.ndarray]: One ``(n, 2)`` lon/lat ring per centre, in input
        order.
    """
    # TODO(#356): for each (lon, lat), sample n bearings and step the angular
    # radius radius_m / MEAN_EARTH_RADIUS_M along a great circle -> (n, 2) lon/lat.
    raise NotImplementedError("tissot_circles is not implemented yet (see #356).")


def add_tissot(ax: Any, ellipses: Sequence[np.ndarray], **style: Any) -> PolyCollection:
    """Draw pre-projected Tissot rings on ``ax`` and return the artist.

    **cleopatra draws what it is given and computes no distortion**: the shape a
    circle takes is a property of the consumer's projection, so the consumer
    projects `tissot_circles` output through its own transform and passes the
    result here. A thin wrapper over the existing
    `cleopatra.glyphs.primitives.polygon_glyph.PolygonGlyph`, which already draws
    a sequence of polygons with differing vertex counts.

    Args:
        ax: A matplotlib `~matplotlib.axes.Axes`.
        ellipses: A sequence of ``(m, 2)`` vertex arrays in the axes' own
            coordinates (already projected by the consumer).
        **style: Style overrides forwarded to the underlying collection
            (``facecolor``, ``edgecolor``, ``linewidth``, ...).

    Returns:
        matplotlib.collections.PolyCollection: The Tissot artist.
    """
    # TODO(#356): draw `ellipses` unchanged via PolygonGlyph (outline-only by
    # default); preserve axis limits; return the artist.
    raise NotImplementedError("add_tissot is not implemented yet (see #356).")
