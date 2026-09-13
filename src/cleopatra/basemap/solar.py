"""Day/night terminator and Tissot-indicatrix artists for flat 2-D axes.

cleopatra can already shade a day/night terminator, but only as *lighting on a
3-D sphere* (`cleopatra.glyphs.globe.textured_globe_glyph.TexturedGlobeGlyph`):
the input is a world-space light *direction*, the output is per-face shaded
*facecolors* on an `Axes3D`, and no lon/lat *geometry* is ever produced. This
module is the flat-map counterpart: it computes the terminator as a small circle
about the subsolar point in lon/lat and draws it (and the filled night region,
and Tissot distortion circles) on an ordinary `matplotlib.axes.Axes`.

Scope boundary -- the same split the rest of ``basemap`` keeps:

- **Solar geometry is CRS-free maths, so it lives here.** The subsolar point for
  a datetime (solar declination plus the Greenwich hour angle via the equation
  of time) and the terminator small circle depend on nothing but the clock; they
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
    add_nightshade(ax, when, alpha=0.35, color="black", zorder=5)   # shade the night

Consumers whose axes is not lon/lat pass a ``transform`` callable, or the
optional ``crs=`` pyproj shortcut, so cleopatra never resolves a projection.

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

from cleopatra.basemap.reference import (
    _is_4326,
    _make_transformer,
    _reproject_arr,
    _validate_axes,
)

#: Solar altitude (degrees) that defines the terminator. The standard value of
#: ``-0.83`` accounts for atmospheric refraction plus the sun's semidiameter at
#: sunrise/sunset. Pass ``-6`` / ``-12`` / ``-18`` for the civil / nautical /
#: astronomical twilight lines instead.
DEFAULT_REFRACTION: float = -0.83

#: Number of samples along the terminator small circle / night-region rings.
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

    Examples:
        - At the June solstice the sun is overhead near the Tropic of Cancer
          (~23.4 deg N), close to Greenwich at 12:00 UTC:
            ```python
            >>> from datetime import UTC, datetime
            >>> lon, lat = subsolar_point(datetime(2026, 6, 21, 12, 0, tzinfo=UTC))
            >>> 23.0 < lat < 24.0
            True
            >>> abs(lon) < 5.0
            True

            ```
        - The subsolar meridian tracks the sun ~15 deg westward each hour:
            ```python
            >>> from datetime import UTC, datetime, timedelta
            >>> noon = datetime(2026, 3, 20, 12, 0, tzinfo=UTC)
            >>> before, _ = subsolar_point(noon)
            >>> after, _ = subsolar_point(noon + timedelta(hours=1))
            >>> -15.5 < after - before < -14.5
            True

            ```
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
    """Return the day/night terminator as lon/lat vertices.

    The terminator is the small circle at angular distance ``90 - refraction``
    from the subsolar point -- a great circle only when ``refraction == 0``.

    Args:
        when: An aware `datetime.datetime` (naive is treated as UTC).
        refraction: Solar altitude in degrees defining the terminator; see
            `DEFAULT_REFRACTION`. Use ``-6`` / ``-12`` / ``-18`` for civil /
            nautical / astronomical twilight.
        n: Number of samples along the circle. The ring is closed, so ``n``
            includes the duplicated endpoint and the effective resolution is
            ``n - 1``.

    Returns:
        numpy.ndarray: An ``(n, 2)`` array of ``(lon, lat)`` degrees, densified
        for a smooth curve. The ring is closed (last vertex equals the first).

    Raises:
        ValueError: If ``refraction`` is outside ``(-90, 0]`` (0 is the
            geometric terminator; a positive value is not a terminator and a
            value of -90 or below is degenerate), or if ``n < 4`` (the ring's
            duplicated endpoint means 4 samples are the fewest that give 3
            distinct vertices).

    Examples:
        - The default terminator is a closed ring of 720 lon/lat vertices:
            ```python
            >>> import numpy as np
            >>> from datetime import UTC, datetime
            >>> ring = terminator(datetime(2026, 6, 21, 12, 0, tzinfo=UTC))
            >>> ring.shape
            (720, 2)
            >>> bool(np.allclose(ring[0], ring[-1]))
            True

            ```
        - A coarser ring stays within the lon/lat bounds; ``refraction`` shifts
          the ring's angular radius (``-6`` gives the civil-twilight line):
            ```python
            >>> import numpy as np
            >>> from datetime import UTC, datetime
            >>> ring = terminator(datetime(2026, 6, 21, 12, 0, tzinfo=UTC), refraction=-6.0, n=180)
            >>> ring.shape
            (180, 2)
            >>> bool(np.all(np.abs(ring[:, 0]) <= 180.0) and np.all(np.abs(ring[:, 1]) <= 90.0))
            True

            ```
    """
    if not -90.0 < refraction <= 0.0:
        raise ValueError(
            "refraction must be in (-90, 0] degrees (0 is the geometric "
            f"terminator; -6/-12/-18 are the twilight lines); got {refraction}."
        )
    if n < 4:
        # linspace(0, 2*pi, n) duplicates the endpoint, so n counts one closing
        # vertex; a non-degenerate ring needs 3 distinct vertices, i.e. n >= 4.
        raise ValueError(f"n must be at least 4 to form a ring; got {n}.")

    lon_s, lat_s = subsolar_point(when)
    # The terminator is the small circle at angular distance 90 - refraction from
    # the subsolar point (a great circle only when refraction == 0).
    return _small_circle(lon_s, lat_s, np.radians(90.0 - refraction), n)


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
        The rings are open (first vertex != last); matplotlib closes polygons
        on fill.

    Raises:
        ValueError: If ``refraction`` is outside ``(-90, 0]`` or ``n < 4`` (via
            `terminator`).

    Examples:
        - At a solstice one pole is in darkness, so the night region is a single
          ring that runs to that pole (the south pole, lat -90, in June):
            ```python
            >>> from datetime import UTC, datetime
            >>> rings = night_polygon(datetime(2026, 6, 21, 12, 0, tzinfo=UTC))
            >>> len(rings)
            1
            >>> float(rings[0][:, 1].min())
            -90.0

            ```
        - When the night region straddles the antimeridian it is split into two
          rings so a flat map does not smear a band across the world:
            ```python
            >>> from datetime import UTC, datetime
            >>> rings = night_polygon(datetime(2026, 3, 20, 12, 0, tzinfo=UTC))
            >>> len(rings)
            2

            ```
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

    Vertices that a ``transform`` or ``crs`` maps to non-finite values -- the
    night region always spans ~half the globe, so a non-global projection sends
    its far side outside the projection's domain -- are dropped so the fill stays
    valid (as `cleopatra.basemap.reference.add_features` does). If a mapping drops
    every vertex, the returned artist has no polygons (nothing is drawn).

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

    Examples:
        - Shade the night region on a lon/lat axes and keep the artist; the axis
          limits are preserved:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import matplotlib.pyplot as plt
            >>> from datetime import UTC, datetime
            >>> fig, ax = plt.subplots()
            >>> _ = ax.set_xlim(-180, 180)
            >>> _ = ax.set_ylim(-90, 90)
            >>> art = add_nightshade(ax, datetime(2026, 6, 21, 12, tzinfo=UTC), alpha=0.3)
            >>> art in ax.collections
            True
            >>> bool(ax.get_xlim() == (-180.0, 180.0))
            True

            ```
        - A consumer whose axes is not lon/lat passes a ``transform`` callable
          (here a trivial scale), so cleopatra never resolves the projection:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import matplotlib.pyplot as plt
            >>> from datetime import UTC, datetime
            >>> fig, ax = plt.subplots()
            >>> art = add_nightshade(
            ...     ax, datetime(2026, 3, 20, 12, tzinfo=UTC), transform=lambda a: a * 2.0
            ... )
            >>> len(art.get_paths()) >= 1
            True

            ```
    """
    if transform is not None and crs is not None:
        raise ValueError("Pass at most one of transform= or crs=, not both.")
    _validate_axes(ax)

    rings = night_polygon(when, refraction=refraction, n=n)
    if transform is not None:
        rings = [np.asarray(transform(ring), dtype=float) for ring in rings]
    elif crs is not None and not _is_4326(crs):
        transformer = _make_transformer(crs)
        rings = [_reproject_arr(ring, transformer) for ring in rings]

    # A transform or reprojection can map points outside the projection's domain
    # (e.g. a non-global projection's far side) to non-finite values; the night
    # region always spans ~half the globe, so drop those rows per ring -- and
    # rings left with < 3 vertices -- to keep the fill valid, mirroring
    # cleopatra.basemap.reference.add_features. If every vertex is dropped the
    # artist is simply empty.
    rings = [ring[np.isfinite(ring).all(axis=1)] for ring in rings]
    rings = [ring for ring in rings if len(ring) >= 3]

    opts: dict[str, Any] = {"facecolor": "black", "edgecolor": "none", "alpha": 0.35}
    if "color" in style:
        # `color` sets both face and edge; drop the split defaults to avoid a clash.
        opts.pop("facecolor")
        opts.pop("edgecolor")
    opts.update(style)

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    artist = PolyCollection(rings, **opts)
    artist.set_transform(ax.transData)
    ax.add_collection(artist)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return artist


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
        order. Each ring is closed (the last vertex duplicates the first).

    Raises:
        ValueError: If ``lons`` or ``lats`` is not 1-D or they differ in shape,
            if ``radius_m`` is not a positive sub-antipodal radius
            (``0 < radius_m < pi * R``), or if ``n < 4``.

    Examples:
        - One ~500 km circle around the origin, sampled coarsely:
            ```python
            >>> import numpy as np
            >>> rings = tissot_circles([0.0], [0.0], 5e5, n=8)
            >>> len(rings)
            1
            >>> rings[0].shape
            (8, 2)
            >>> bool(np.isfinite(rings[0]).all())
            True

            ```
        - The ground radius maps to a fixed angular radius (~4.5 deg at 500 km):
            ```python
            >>> rings = tissot_circles([0.0], [0.0], 5e5, n=4)
            >>> round(float(rings[0][:, 1].max()), 1)
            4.5

            ```
    """
    lon_arr = np.atleast_1d(np.asarray(lons, dtype=float))
    lat_arr = np.atleast_1d(np.asarray(lats, dtype=float))
    if lon_arr.ndim != 1 or lat_arr.ndim != 1:
        raise ValueError(
            f"lons and lats must be 1-D sequences of centres; got {lon_arr.ndim}-D "
            f"and {lat_arr.ndim}-D."
        )
    if lon_arr.shape != lat_arr.shape:
        raise ValueError(
            f"lons and lats must have the same shape; got {lon_arr.shape} and {lat_arr.shape}."
        )
    if not 0.0 < radius_m < np.pi * MEAN_EARTH_RADIUS_M:
        raise ValueError(
            "radius_m must be a positive, sub-antipodal ground radius in metres "
            f"(0, {np.pi * MEAN_EARTH_RADIUS_M:.0f}); got {radius_m}."
        )
    if n < 4:
        raise ValueError(f"n must be at least 4 to form a ring; got {n}.")

    radius_rad = radius_m / MEAN_EARTH_RADIUS_M
    return [
        _small_circle(float(lon), float(lat), radius_rad, n)
        for lon, lat in zip(lon_arr, lat_arr, strict=True)
    ]


def add_tissot(ax: Any, ellipses: Sequence[np.ndarray], **style: Any) -> PolyCollection:
    """Draw pre-projected Tissot rings on ``ax`` and return the artist.

    **cleopatra draws what it is given and computes no distortion**: the shape a
    circle takes is a property of the consumer's projection, so the consumer
    projects `tissot_circles` output through its own transform and passes the
    result here. It draws the rings as an unfilled `PolyCollection` (outline only
    by default) -- the same "sequence of polygons with differing vertex counts"
    shape `cleopatra.glyphs.primitives.polygon_glyph.PolygonGlyph` handles -- and
    preserves the current axis limits.

    Args:
        ax: A matplotlib `~matplotlib.axes.Axes`.
        ellipses: A sequence of ``(m, 2)`` vertex arrays in the axes' own
            coordinates (already projected by the consumer).
        **style: Style overrides forwarded to the underlying collection
            (``facecolor``, ``edgecolor``, ``linewidth``, ...).

    Returns:
        matplotlib.collections.PolyCollection: The Tissot artist.

    Raises:
        TypeError: If ``ax`` is not a matplotlib Axes.

    Examples:
        - Draw two supplied rings (already in axes coordinates) and keep them:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import matplotlib.pyplot as plt
            >>> import numpy as np
            >>> theta = np.linspace(0, 2 * np.pi, 16)
            >>> circle = np.column_stack([np.cos(theta), np.sin(theta)])
            >>> fig, ax = plt.subplots()
            >>> art = add_tissot(ax, [circle, circle + 3.0], edgecolor="crimson")
            >>> len(art.get_paths())
            2
            >>> art in ax.collections
            True

            ```
        - Generate geodesic circles with `tissot_circles` and draw them on a
          lon/lat axes:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import matplotlib.pyplot as plt
            >>> from cleopatra.basemap.solar import tissot_circles
            >>> circles = tissot_circles([-90.0, 0.0, 90.0], [0.0, 0.0, 0.0], 5e5)
            >>> fig, ax = plt.subplots()
            >>> art = add_tissot(ax, circles, edgecolor="navy")
            >>> len(art.get_paths())
            3

            ```
    """
    _validate_axes(ax)
    verts = [np.asarray(ring, dtype=float) for ring in ellipses]

    opts: dict[str, Any] = {"facecolor": "none", "edgecolor": "black", "linewidth": 0.8}
    if "color" in style:
        # `color` sets both face and edge; drop the split defaults to avoid a clash.
        opts.pop("facecolor")
        opts.pop("edgecolor")
    opts.update(style)

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    artist = PolyCollection(verts, **opts)
    artist.set_transform(ax.transData)
    ax.add_collection(artist)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return artist


# --------------------------------------------------------------------------- #
# Internal helpers                                                            #
# --------------------------------------------------------------------------- #
def _small_circle(
    center_lon: float, center_lat: float, radius_rad: float, n: int
) -> np.ndarray:
    """Return an ``(n, 2)`` closed lon/lat ring at a fixed angular radius.

    Samples ``n`` bearings around ``(center_lon, center_lat)`` and steps
    ``radius_rad`` radians of great-circle distance to each. Shared by
    `terminator` (radius ``90 - refraction``) and `tissot_circles` (radius
    ``radius_m / MEAN_EARTH_RADIUS_M``).

    Args:
        center_lon: Centre longitude in degrees.
        center_lat: Centre latitude in degrees.
        radius_rad: Angular radius (great-circle distance) in radians.
        n: Number of vertices; the ring is closed, so the last duplicates the
            first.

    Returns:
        numpy.ndarray: An ``(n, 2)`` array of ``(lon, lat)`` degrees, longitude
        wrapped to ``(-180, 180]`` and latitude clamped to ``[-90, 90]``.
    """
    lon0, lat0 = np.radians(center_lon), np.radians(center_lat)
    bearing = np.linspace(0.0, 2.0 * np.pi, n)
    # Clip guards the sum-of-products against a 1-ULP overshoot of +/-1 (which
    # would make arcsin return NaN) when the circle grazes a pole.
    lat = np.arcsin(
        np.clip(
            np.sin(lat0) * np.cos(radius_rad)
            + np.cos(lat0) * np.sin(radius_rad) * np.cos(bearing),
            -1.0,
            1.0,
        )
    )
    lon = lon0 + np.arctan2(
        np.sin(bearing) * np.sin(radius_rad) * np.cos(lat0),
        np.cos(radius_rad) - np.sin(lat0) * np.sin(lat),
    )
    return np.column_stack([_wrap_longitude(np.degrees(lon)), np.degrees(lat)])


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
            # The clip already bounds longitudes to [-180, 180]; do NOT re-wrap,
            # or a piece's seam vertices at -180 would flip to +180 and smear the
            # ring across the whole map (the very thing the split prevents).
            rings.append(clipped)
    return rings


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
