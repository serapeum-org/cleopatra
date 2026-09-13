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

    from datetime import UTC, datetime
    from cleopatra.basemap.solar import add_nightshade

    when = datetime(2026, 6, 21, 12, tzinfo=UTC)
    add_nightshade(ax, when, alpha=0.35, color="black", zorder=5)

See also `cleopatra.glyphs.globe.textured_globe_glyph` for the 3-D globe's
directional lighting, which answers a different question (shading on a sphere,
not geometry on a flat map).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
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
    dt = when.replace(tzinfo=UTC) if when.tzinfo is None else when.astimezone(UTC)
    hours = dt.hour + dt.minute / 60.0 + dt.second / 3600.0 + dt.microsecond / 3.6e9

    # Julian Day (Gregorian) then Julian centuries since J2000.0.
    a = (14 - dt.month) // 12
    y = dt.year + 4800 - a
    m = dt.month + 12 * a - 3
    jdn = dt.day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    jd = jdn + (hours - 12.0) / 24.0
    t = (jd - 2451545.0) / 36525.0

    # NOAA low-precision solar position (Meeus, Astronomical Algorithms).
    mean_long = np.radians((280.46646 + t * (36000.76983 + t * 0.0003032)) % 360.0)
    mean_anom = np.radians(357.52911 + t * (35999.05029 - 0.0001537 * t))
    eccentricity = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
    center = (
        np.sin(mean_anom) * (1.914602 - t * (0.004817 + 0.000014 * t))
        + np.sin(2 * mean_anom) * (0.019993 - 0.000101 * t)
        + np.sin(3 * mean_anom) * 0.000289
    )
    true_long = np.degrees(mean_long) + center
    omega = np.radians(125.04 - 1934.136 * t)
    app_long = np.radians(true_long - 0.00569 - 0.00478 * np.sin(omega))
    obliquity = np.radians(
        23.0
        + (26.0 + (21.448 - t * (46.8150 + t * (0.00059 - t * 0.001813))) / 60.0) / 60.0
        + 0.00256 * np.cos(omega)
    )

    declination = np.degrees(np.arcsin(np.sin(obliquity) * np.sin(app_long)))

    # Equation of time (minutes) -> subsolar longitude via apparent solar time.
    var_y = np.tan(obliquity / 2.0) ** 2
    eqn_time = 4.0 * np.degrees(
        var_y * np.sin(2 * mean_long)
        - 2 * eccentricity * np.sin(mean_anom)
        + 4 * eccentricity * var_y * np.sin(mean_anom) * np.cos(2 * mean_long)
        - 0.5 * var_y**2 * np.sin(4 * mean_long)
        - 1.25 * eccentricity**2 * np.sin(2 * mean_anom)
    )
    lon = -15.0 * (hours - 12.0 + eqn_time / 60.0)
    return float(_wrap_longitude(lon)), float(declination)


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
        for a smooth curve. The ring is closed (last vertex equals the first).
    """
    lon_s, lat_s = subsolar_point(when)
    lon0, lat0 = np.radians(lon_s), np.radians(lat_s)
    # Points where the solar altitude equals ``refraction`` lie a great-circle
    # distance ``90 - refraction`` from the subsolar point.
    rho = np.radians(90.0 - refraction)
    bearing = np.linspace(0.0, 2.0 * np.pi, n)

    lat = np.arcsin(
        np.sin(lat0) * np.cos(rho) + np.cos(lat0) * np.sin(rho) * np.cos(bearing)
    )
    lon = lon0 + np.arctan2(
        np.sin(bearing) * np.sin(rho) * np.cos(lat0),
        np.cos(rho) - np.sin(lat0) * np.sin(lat),
    )
    return np.column_stack([_wrap_longitude(np.degrees(lon)), np.degrees(lat)])


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
    _, lat_s = subsolar_point(when)
    ring = terminator(when, refraction=refraction, n=n)

    # The sun's altitude at the north/south pole is +lat_s / -lat_s. A pole is
    # in night when its altitude drops below ``refraction``.
    north_dark = lat_s < refraction
    south_dark = -lat_s < refraction
    if north_dark != south_dark:
        # A geographic pole is enclosed: the night cap wraps every longitude, so
        # close the (single-valued) terminator along that pole's map edge.
        dark_lat = 90.0 if north_dark else -90.0
        return [_pole_cap_ring(ring, dark_lat)]

    # No pole enclosed: the night cap is a simple sub-hemispheric region bounded
    # by the terminator loop. Split it wherever it crosses the antimeridian.
    return _split_antimeridian(ring)


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


# --------------------------------------------------------------------------- #
# Internal helpers                                                            #
# --------------------------------------------------------------------------- #
def _wrap_longitude(lon: Any) -> np.ndarray:
    """Wrap longitudes (deg) into the half-open range ``(-180, 180]``."""
    wrapped = (np.asarray(lon, dtype=float) + 180.0) % 360.0 - 180.0
    return np.where(wrapped == -180.0, 180.0, wrapped)


def _pole_cap_ring(ring: np.ndarray, dark_lat: float) -> np.ndarray:
    """Close a pole-enclosing terminator into a fillable lon/lat ring.

    The terminator of a night cap that contains a geographic pole is single
    valued in longitude, so it is sorted by longitude, extended to the map edges
    (longitude is periodic, so the latitude at -180 equals the latitude at +180)
    and closed along the dark pole's edge (``dark_lat`` = +90 or -90).
    """
    boundary = ring[np.argsort(ring[:, 0])]
    edges = np.array([[-180.0, boundary[-1, 1]], [180.0, boundary[0, 1]]])
    boundary = np.vstack([edges[:1], boundary, edges[1:]])
    closure = np.array([[180.0, dark_lat], [-180.0, dark_lat]])
    return np.vstack([boundary, closure])


def _split_antimeridian(ring: np.ndarray) -> list[np.ndarray]:
    """Split a lon/lat ring into pieces that each stay within ``(-180, 180]``.

    The ring is unwrapped to continuous longitudes, then it and its +/-360 shifts
    are clipped to the ``[-180, 180]`` longitude strip. A ring straddling the
    antimeridian yields two pieces; one that does not yields a single piece.
    """
    lon_unwrapped = np.degrees(np.unwrap(np.radians(ring[:, 0])))
    poly = np.column_stack([lon_unwrapped, ring[:, 1]])
    rings: list[np.ndarray] = []
    for shift in (-360.0, 0.0, 360.0):
        clipped = _clip_lon_strip(poly + np.array([shift, 0.0]), -180.0, 180.0)
        if len(clipped) >= 3:
            rings.append(_wrap_seam(clipped))
    return rings if rings else [ring]


def _wrap_seam(poly: np.ndarray) -> np.ndarray:
    """Return ``poly`` with longitudes wrapped to ``(-180, 180]`` in place."""
    return np.column_stack([_wrap_longitude(poly[:, 0]), poly[:, 1]])


def _clip_lon_strip(poly: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Sutherland-Hodgman clip of ``poly`` to the longitude strip ``[lo, hi]``."""
    clipped = _clip_halfplane(poly, hi, keep_below=True)
    if len(clipped) == 0:
        return clipped
    return _clip_halfplane(clipped, lo, keep_below=False)


def _clip_halfplane(poly: np.ndarray, bound: float, *, keep_below: bool) -> np.ndarray:
    """Clip ``poly`` to the half-plane ``x <= bound`` (or ``x >= bound``)."""
    out: list[np.ndarray] = []
    for i in range(len(poly)):
        cur, prev = poly[i], poly[i - 1]
        cur_in = cur[0] <= bound if keep_below else cur[0] >= bound
        prev_in = prev[0] <= bound if keep_below else prev[0] >= bound
        if cur_in:
            if not prev_in:
                out.append(_intersect_x(prev, cur, bound))
            out.append(cur)
        elif prev_in:
            out.append(_intersect_x(prev, cur, bound))
    return np.array(out) if out else np.empty((0, 2))


def _intersect_x(p: np.ndarray, q: np.ndarray, x: float) -> np.ndarray:
    """Return the point where segment ``p->q`` crosses the vertical line ``x``."""
    t = (x - p[0]) / (q[0] - p[0])
    return np.array([x, p[1] + t * (q[1] - p[1])])
