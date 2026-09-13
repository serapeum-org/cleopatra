"""Tests for `cleopatra.basemap.solar`.

Covers the CRS-free solar geometry (`subsolar_point`, `terminator`,
`night_polygon`), the module constants, the antimeridian/pole helpers, and the
`NotImplementedError` contracts of the still-scaffolded artist functions
(`add_nightshade`, `tissot_circles`, `add_tissot`).

Astronomical expectations are checked against tolerances that comfortably exceed
the low-precision NOAA/Meeus model's error budget: solar declination reaches the
obliquity (~23.44 deg) at the solstices and ~0 deg at the equinoxes, the
subsolar point tracks apparent solar time (~15 deg/hour westward), and the night
region always covers ~half the sphere.
"""

from datetime import UTC, datetime, timedelta, timezone

import numpy as np
import pytest
from matplotlib.path import Path as MplPath

from cleopatra.basemap import solar
from cleopatra.basemap.solar import (
    DEFAULT_REFRACTION,
    DEFAULT_TERMINATOR_SAMPLES,
    DEFAULT_TISSOT_SAMPLES,
    MEAN_EARTH_RADIUS_M,
    add_nightshade,
    add_tissot,
    night_polygon,
    subsolar_point,
    terminator,
    tissot_circles,
)

# 2026 solstices/equinoxes (UTC), accurate to a few minutes -- fine for the
# tolerances used below.
MAR_EQUINOX = datetime(2026, 3, 20, 14, 46, tzinfo=UTC)
JUN_SOLSTICE = datetime(2026, 6, 21, 8, 25, tzinfo=UTC)
SEP_EQUINOX = datetime(2026, 9, 23, 0, 5, tzinfo=UTC)
DEC_SOLSTICE = datetime(2026, 12, 21, 20, 50, tzinfo=UTC)

# Obliquity of the ecliptic (max |declination|) in degrees.
OBLIQUITY_DEG = 23.44


def _angular_distance_deg(lon0, lat0, lon, lat):
    """Return the great-circle distance (deg) from ``(lon0, lat0)`` to points.

    Args:
        lon0: Centre longitude in degrees (scalar).
        lat0: Centre latitude in degrees (scalar).
        lon: Longitude(s) in degrees (scalar or array).
        lat: Latitude(s) in degrees (scalar or array).

    Returns:
        numpy.ndarray: Angular distance(s) in degrees.
    """
    lon0r, lat0r, lonr, latr = (np.radians(v) for v in (lon0, lat0, lon, lat))
    cos_d = np.sin(lat0r) * np.sin(latr) + np.cos(lat0r) * np.cos(latr) * np.cos(
        lonr - lon0r
    )
    return np.degrees(np.arccos(np.clip(cos_d, -1.0, 1.0)))


def _sphere_fraction(rings):
    """Return the fraction of the sphere enclosed by lon/lat ``rings``.

    Uses the spherical-excess ("shoelace on the sphere") formula, wrapping each
    longitude step into ``(-pi, pi]`` so antimeridian seam edges do not blow up.

    Args:
        rings: Sequence of ``(m, 2)`` lon/lat arrays.

    Returns:
        float: Enclosed area as a fraction of the full sphere.
    """
    total = 0.0
    for ring in rings:
        lon = np.radians(ring[:, 0])
        lat = np.radians(ring[:, 1])
        if not (np.isclose(lon[0], lon[-1]) and np.isclose(lat[0], lat[-1])):
            lon = np.append(lon, lon[0])
            lat = np.append(lat, lat[0])
        dlon = np.diff(lon)
        dlon = (dlon + np.pi) % (2 * np.pi) - np.pi
        total += np.sum(dlon * (2 + np.sin(lat[:-1]) + np.sin(lat[1:]))) / 2.0
    return abs(total) / (4 * np.pi)


def _fill_night_fraction(rings):
    """Return the rasterised night fraction of the sphere by filling ``rings``.

    Unlike `_sphere_fraction` (a shoelace on lon/lat that cancels an
    antimeridian smear), this does a real point-in-polygon fill on a lon/lat
    grid, cos-lat weighted, so a ring smeared across the map inflates the result.

    Args:
        rings: Sequence of ``(m, 2)`` lon/lat arrays.

    Returns:
        float: Filled area as a cos-lat-weighted fraction of the sphere.
    """
    lon = np.linspace(-179.5, 179.5, 720)
    lat = np.linspace(-89.5, 89.5, 360)
    grid_lon, grid_lat = np.meshgrid(lon, lat)
    points = np.column_stack([grid_lon.ravel(), grid_lat.ravel()])
    inside = np.zeros(len(points), dtype=bool)
    for ring in rings:
        inside |= MplPath(ring).contains_points(points)
    weights = np.cos(np.radians(grid_lat)).ravel()
    return float((inside * weights).sum() / weights.sum())


class TestModuleConstants:
    """Tests for the module-level default constants."""

    @pytest.mark.parametrize(
        "name, value",
        [
            ("DEFAULT_REFRACTION", -0.83),
            ("DEFAULT_TERMINATOR_SAMPLES", 720),
            ("DEFAULT_TISSOT_SAMPLES", 64),
            ("MEAN_EARTH_RADIUS_M", 6_371_008.8),
        ],
    )
    def test_constant_values(self, name, value):
        """Test the public constants hold their documented values.

        Args:
            name: Attribute name on the module.
            value: Expected value.

        Test scenario:
            Each constant should keep the value the API advertises so callers
            can rely on the defaults.
        """
        actual = getattr(solar, name)
        assert actual == value, f"{name} expected {value}, got {actual}"

    def test_earth_radius_is_realistic(self):
        """Test the mean Earth radius is a plausible metre value.

        Test scenario:
            The IUGG mean radius should sit between 6.36e6 and 6.38e6 m.
        """
        assert 6.36e6 < MEAN_EARTH_RADIUS_M < 6.38e6, (
            f"MEAN_EARTH_RADIUS_M out of range: {MEAN_EARTH_RADIUS_M}"
        )


class TestSubsolarPoint:
    """Tests for subsolar_point."""

    @pytest.mark.parametrize(
        "when, expected_lat",
        [
            (JUN_SOLSTICE, OBLIQUITY_DEG),
            (DEC_SOLSTICE, -OBLIQUITY_DEG),
            (MAR_EQUINOX, 0.0),
            (SEP_EQUINOX, 0.0),
        ],
    )
    def test_declination_at_solstices_and_equinoxes(self, when, expected_lat):
        """Test the subsolar latitude equals the solar declination.

        Args:
            when: Datetime of a solstice or equinox.
            expected_lat: Expected declination in degrees.

        Test scenario:
            Declination should reach +/-23.44 deg at the solstices and ~0 deg at
            the equinoxes, within 0.1 deg of the expected value.
        """
        _, lat = subsolar_point(when)
        assert abs(lat - expected_lat) < 0.1, f"declination {lat} != {expected_lat}"

    @pytest.mark.parametrize("month", [3, 6, 9, 12])
    def test_longitude_near_zero_at_noon_utc(self, month):
        """Test the subsolar longitude is near 0 at 12:00 UTC.

        Args:
            month: Month to test at day 21, 12:00 UTC.

        Test scenario:
            At solar-clock noon UTC the subsolar meridian is near Greenwich; the
            equation of time keeps it within ~5 deg of 0.
        """
        lon, _ = subsolar_point(datetime(2026, month, 21, 12, 0, tzinfo=UTC))
        assert abs(lon) < 5.0, f"noon subsolar lon should be near 0, got {lon}"

    def test_longitude_moves_west_with_time(self):
        """Test the subsolar longitude moves ~15 deg west per hour.

        Test scenario:
            One hour of rotation moves the subsolar meridian ~15 deg westward
            (Earth turns 360 deg/24 h), within 0.1 deg over a non-wrapping step.
        """
        t0 = datetime(2026, 6, 21, 12, 0, tzinfo=UTC)
        lon0, _ = subsolar_point(t0)
        lon1, _ = subsolar_point(t0 + timedelta(hours=1))
        assert abs((lon1 - lon0) - (-15.0)) < 0.1, f"hourly step {lon1 - lon0} != -15"

    @pytest.mark.parametrize(
        "when",
        [
            datetime(2026, 1, 1, 3, 30, tzinfo=UTC),
            datetime(2026, 6, 21, 8, 25, tzinfo=UTC),
            datetime(2026, 11, 15, 23, 59, tzinfo=UTC),
        ],
    )
    def test_output_ranges_and_types(self, when):
        """Test subsolar_point returns floats in the documented ranges.

        Args:
            when: Datetime to evaluate.

        Test scenario:
            Longitude in (-180, 180], latitude in [-90, 90], both plain floats.
        """
        lon, lat = subsolar_point(when)
        assert isinstance(lon, float), f"lon not a float: {type(lon)}"
        assert isinstance(lat, float), f"lat not a float: {type(lat)}"
        assert -180.0 < lon <= 180.0, f"lon out of range: {lon}"
        assert -90.0 <= lat <= 90.0, f"lat out of range: {lat}"

    def test_naive_datetime_treated_as_utc(self):
        """Test a naive datetime is interpreted as UTC.

        Test scenario:
            A tz-naive datetime should give the same result as the identical
            UTC-aware datetime.
        """
        naive = datetime(2026, 6, 21, 12, 0)
        aware = datetime(2026, 6, 21, 12, 0, tzinfo=UTC)
        assert subsolar_point(naive) == pytest.approx(subsolar_point(aware)), (
            "naive datetime should be treated as UTC"
        )

    def test_aware_non_utc_datetime_is_converted(self):
        """Test an aware non-UTC datetime is converted to UTC first.

        Test scenario:
            14:00 at UTC+02:00 is the same instant as 12:00 UTC and must give the
            same subsolar point.
        """
        plus_two = datetime(2026, 6, 21, 14, 0, tzinfo=timezone(timedelta(hours=2)))
        utc = datetime(2026, 6, 21, 12, 0, tzinfo=UTC)
        assert subsolar_point(plus_two) == pytest.approx(subsolar_point(utc)), (
            "non-UTC datetime should convert to UTC"
        )


class TestTerminator:
    """Tests for terminator."""

    def test_default_shape_and_closed_ring(self):
        """Test the default terminator shape and that the ring is closed.

        Test scenario:
            Default n=720 gives a (720, 2) array whose last vertex equals the
            first (a closed curve).
        """
        term = terminator(JUN_SOLSTICE)
        assert term.shape == (DEFAULT_TERMINATOR_SAMPLES, 2), f"shape {term.shape}"
        assert np.allclose(term[0], term[-1]), "terminator ring is not closed"

    @pytest.mark.parametrize("n", [8, 90, 360])
    def test_custom_sample_count(self, n):
        """Test the sample count is honoured.

        Args:
            n: Requested number of vertices.

        Test scenario:
            The returned array should have exactly ``n`` rows.
        """
        term = terminator(JUN_SOLSTICE, n=n)
        assert term.shape == (n, 2), f"expected ({n}, 2), got {term.shape}"

    def test_longitudes_within_range(self):
        """Test terminator longitudes are wrapped to (-180, 180].

        Test scenario:
            No vertex longitude may fall outside the half-open lon range.
        """
        term = terminator(DEC_SOLSTICE)
        assert np.all(term[:, 0] > -180.0), "terminator longitude <= -180"
        assert np.all(term[:, 0] <= 180.0), "terminator longitude > 180"
        assert np.all(np.abs(term[:, 1]) <= 90.0), (
            "terminator latitudes out of [-90, 90]"
        )

    @pytest.mark.parametrize("refraction", [0.0, DEFAULT_REFRACTION, -6.0, -18.0])
    def test_points_lie_at_expected_distance(self, refraction):
        """Test every terminator point is 90-refraction deg from the subsolar point.

        Args:
            refraction: Solar altitude defining the terminator.

        Test scenario:
            By definition the terminator is the small circle at angular distance
            ``90 - refraction`` from the subsolar point; all vertices must match
            to numerical precision.
        """
        lon_s, lat_s = subsolar_point(JUN_SOLSTICE)
        term = terminator(JUN_SOLSTICE, refraction=refraction, n=180)
        dist = _angular_distance_deg(lon_s, lat_s, term[:, 0], term[:, 1])
        assert np.allclose(dist, 90.0 - refraction, atol=1e-6), (
            f"distances not {90.0 - refraction}: min {dist.min()}, max {dist.max()}"
        )

    def test_passes_through_poles_at_equinox(self):
        """Test the geometric terminator reaches both poles at an equinox.

        Test scenario:
            The refraction=0 terminator is the great circle whose pole is the
            subsolar point, so its extreme latitudes are +/-(90 - |lat_s|). At an
            equinox (declination ~0) that is ~+/-90, i.e. it passes through the
            poles. ``n=721`` is odd so bearings 0 and pi (the extremes) are both
            sampled exactly.
        """
        _, lat_s = subsolar_point(MAR_EQUINOX)
        reach = 90.0 - abs(lat_s)
        term = terminator(MAR_EQUINOX, refraction=0.0, n=721)
        assert reach > 89.9, f"equinox subsolar lat too large: {lat_s}"
        assert term[:, 1].max() == pytest.approx(reach, abs=0.05), (
            f"north extreme {term[:, 1].max()} != {reach}"
        )
        assert term[:, 1].min() == pytest.approx(-reach, abs=0.05), (
            f"south extreme {term[:, 1].min()} != {-reach}"
        )

    @pytest.mark.parametrize("refraction", [0.1, 5.0, -90.0, -100.0])
    def test_invalid_refraction_raises(self, refraction):
        """Test out-of-domain refraction values are rejected.

        Args:
            refraction: An invalid solar altitude in degrees.

        Test scenario:
            Positive refraction is not a terminator and -90 or below is
            degenerate; both must raise ValueError rather than mis-branch.
        """
        with pytest.raises(ValueError, match="refraction"):
            terminator(JUN_SOLSTICE, refraction=refraction)

    @pytest.mark.parametrize("n", [0, 1, 2, 3])
    def test_too_few_samples_raises(self, n):
        """Test a sample count below 4 is rejected.

        Args:
            n: An invalid vertex count.

        Test scenario:
            The ring's duplicated endpoint means n=3 gives only 2 distinct
            vertices (a line), so the fewest that form a real ring is n=4;
            anything below must raise ValueError.
        """
        with pytest.raises(ValueError, match="at least 4"):
            terminator(JUN_SOLSTICE, n=n)

    def test_minimum_valid_sample_count(self):
        """Test n=4 is accepted and yields 3 distinct vertices plus the closure.

        Test scenario:
            n=4 is the smallest non-degenerate ring; its first three vertices are
            distinct and the fourth duplicates the first.
        """
        ring = terminator(JUN_SOLSTICE, n=4)
        assert ring.shape == (4, 2), f"expected (4, 2), got {ring.shape}"
        assert np.allclose(ring[0], ring[-1]), (
            "n=4 ring should close (endpoint duplicates start)"
        )
        assert not np.allclose(ring[0], ring[1]), "vertices 0 and 1 should differ"
        assert not np.allclose(ring[1], ring[2]), "vertices 1 and 2 should differ"
        assert not np.allclose(ring[0], ring[2]), "vertices 0 and 2 should differ"

    def test_finite_at_terminator_through_pole(self):
        """Test the terminator stays finite when it grazes a pole.

        Test scenario:
            Setting refraction to the subsolar latitude places the terminator
            exactly through a pole (lat_s == refraction), where the arcsin
            argument reaches +/-1 and a missing clip would emit NaN vertices.
        """
        lat_s = subsolar_point(DEC_SOLSTICE)[1]
        ring = terminator(DEC_SOLSTICE, refraction=lat_s, n=721)
        assert np.isfinite(ring).all(), (
            "terminator produced non-finite vertices at the pole"
        )
        assert np.all(np.abs(ring[:, 1]) <= 90.0 + 1e-9), (
            "latitude out of range at the pole"
        )


class TestNightPolygon:
    """Tests for night_polygon."""

    def test_returns_list_of_lonlat_rings(self):
        """Test the return type is a list of (m, 2) arrays.

        Test scenario:
            Every element is a 2-column numpy array of vertices.
        """
        rings = night_polygon(JUN_SOLSTICE)
        assert isinstance(rings, list), f"not a list: {rings!r}"
        assert len(rings) >= 1, f"no rings returned: {rings!r}"
        for ring in rings:
            assert ring.ndim == 2, f"ring not 2-D: {ring.shape}"
            assert ring.shape[1] == 2, f"ring not (m, 2): {ring.shape}"

    def test_single_ring_and_dark_pole_when_pole_enclosed(self):
        """Test a solstice yields one ring closed along the dark pole edge.

        Test scenario:
            At the June solstice the south pole is dark, so the night region
            wraps every longitude as a single ring that reaches lat = -90.
        """
        rings = night_polygon(JUN_SOLSTICE)
        assert len(rings) == 1, f"expected 1 ring, got {len(rings)}"
        assert rings[0][:, 1].min() == pytest.approx(-90.0), "south pole edge missing"

    def test_north_pole_dark_at_december_solstice(self):
        """Test the December solstice darkens the north pole.

        Test scenario:
            At the December solstice the north pole is dark, so the single ring
            reaches lat = +90.
        """
        rings = night_polygon(DEC_SOLSTICE)
        assert len(rings) == 1, f"expected 1 ring, got {len(rings)}"
        assert rings[0][:, 1].max() == pytest.approx(90.0), "north pole edge missing"

    def test_two_rings_when_straddling_antimeridian(self):
        """Test an equinox with the antisolar point near +/-180 splits into two rings.

        Test scenario:
            At the March equinox at 12:00 UTC the subsolar point is near lon 0,
            so the night region centres on the antimeridian and splits into two.
        """
        rings = night_polygon(datetime(2026, 3, 20, 12, 0, tzinfo=UTC))
        assert len(rings) == 2, (
            f"expected 2 rings across the antimeridian, got {len(rings)}"
        )

    def test_single_ring_when_not_straddling(self):
        """Test an equinox with the antisolar point near lon 0 stays one ring.

        Test scenario:
            At the March equinox at 00:00 UTC the subsolar point is near +/-180,
            so the night region centres on lon 0 and does not split.
        """
        rings = night_polygon(datetime(2026, 3, 20, 0, 0, tzinfo=UTC))
        assert len(rings) == 1, f"expected 1 ring, got {len(rings)}"

    @pytest.mark.parametrize(
        "when",
        [
            JUN_SOLSTICE,
            DEC_SOLSTICE,
            datetime(2026, 3, 20, 12, 0, tzinfo=UTC),
            datetime(2026, 3, 20, 0, 0, tzinfo=UTC),
            datetime(2026, 9, 23, 6, 0, tzinfo=UTC),
        ],
    )
    def test_night_area_is_about_half_the_sphere(self, when):
        """Test the night region covers ~half the sphere for varied dates.

        Args:
            when: Datetime to evaluate.

        Test scenario:
            The night cap radius is ~89 deg, so its area is ~0.49 of the sphere
            regardless of the pole-enclosed / straddling / single-ring regime.
        """
        frac = _sphere_fraction(night_polygon(when))
        assert 0.47 < frac < 0.51, f"night fraction {frac} not ~0.49 for {when}"

    def test_rings_stay_within_longitude_range(self):
        """Test all ring longitudes stay within [-180, 180].

        Test scenario:
            After the antimeridian split no vertex may exceed the map bounds.
        """
        for when in (JUN_SOLSTICE, datetime(2026, 3, 20, 12, 0, tzinfo=UTC)):
            for ring in night_polygon(when):
                assert np.all(ring[:, 0] >= -180.0 - 1e-9), f"lon < -180 for {when}"
                assert np.all(ring[:, 0] <= 180.0 + 1e-9), f"lon > 180 for {when}"

    def test_custom_sample_count_still_valid(self):
        """Test a small sample count still produces a valid ~half-sphere region.

        Test scenario:
            n=180 should still yield a night area close to half the sphere.
        """
        frac = _sphere_fraction(night_polygon(JUN_SOLSTICE, n=180))
        assert 0.47 < frac < 0.51, f"night fraction {frac} not ~0.49 at n=180"

    @pytest.mark.parametrize(
        "when",
        [
            datetime(2026, 3, 20, 0, 0, tzinfo=UTC),
            datetime(2026, 3, 20, 12, 0, tzinfo=UTC),
            datetime(2026, 3, 20, 18, 0, tzinfo=UTC),
            JUN_SOLSTICE,
            DEC_SOLSTICE,
        ],
    )
    def test_rings_fill_about_half_the_sphere(self, when):
        """Test a real point-in-polygon fill covers ~half the sphere.

        Args:
            when: Datetime to evaluate.

        Test scenario:
            Rasterising the returned rings (not the shoelace area, which cancels
            an antimeridian smear) must give ~0.49, catching the class of bug
            where a straddling ring is smeared across the whole map.
        """
        frac = _fill_night_fraction(night_polygon(when))
        assert 0.46 < frac < 0.52, f"filled night fraction {frac} not ~0.49 for {when}"

    def test_straddling_pieces_do_not_span_the_map(self):
        """Test each straddling piece stays a narrow seam strip, not a whole band.

        Test scenario:
            When the night region splits at the antimeridian, neither piece may
            span close to 360 deg of longitude (that would be the smear bug); the
            two seam-hugging halves are each well under 200 deg wide.
        """
        rings = night_polygon(datetime(2026, 3, 20, 12, 0, tzinfo=UTC))
        assert len(rings) == 2, f"expected 2 straddling rings, got {len(rings)}"
        for ring in rings:
            span = float(ring[:, 0].max() - ring[:, 0].min())
            assert span < 200.0, f"ring spans {span} deg -- smeared across the map"

    @pytest.mark.parametrize("day", [20, 21, 22, 23, 25])
    def test_transition_regime_near_equinox_fills_half(self, day):
        """Test dates straddling the pole-enclosure boundary still fill ~half.

        Args:
            day: Day in March 2026, sweeping the subsolar latitude from ~0 deg
                (equinox, no pole enclosed) past the ``|lat_s| > |refraction|``
                boundary into the pole-enclosed regime a few days later.

        Test scenario:
            The switch between the antimeridian-split and pole-cap branches
            happens near ``|lat_s| == |refraction|`` (~0.83 deg). A real fill must
            stay ~0.49 across that transition, not just at the far-from-boundary
            solstices the other area tests cover.
        """
        frac = _fill_night_fraction(
            night_polygon(datetime(2026, 3, day, 12, 0, tzinfo=UTC))
        )
        assert 0.46 < frac < 0.52, (
            f"filled night fraction {frac} not ~0.49 on 2026-03-{day}"
        )

    def test_invalid_refraction_propagates(self):
        """Test night_polygon rejects out-of-domain refraction via terminator.

        Test scenario:
            refraction > 0 would mis-branch (both poles dark); it must raise
            rather than return a wrong region.
        """
        with pytest.raises(ValueError, match="refraction"):
            night_polygon(JUN_SOLSTICE, refraction=30.0)

    @pytest.mark.parametrize("n", [2, 3])
    def test_too_few_samples_propagates(self, n):
        """Test night_polygon rejects n < 4 via terminator.

        Args:
            n: An invalid vertex count.

        Test scenario:
            A non-degenerate ring needs n >= 4; fewer must raise ValueError.
        """
        with pytest.raises(ValueError, match="at least 4"):
            night_polygon(JUN_SOLSTICE, n=n)


class TestUnimplementedArtists:
    """Tests for the still-scaffolded artist functions."""

    def test_add_nightshade_raises_not_implemented(self):
        """Test add_nightshade raises NotImplementedError.

        Test scenario:
            The stub must raise before touching the axes, naming itself.
        """
        with pytest.raises(NotImplementedError, match="add_nightshade") as exc:
            add_nightshade(None, JUN_SOLSTICE)
        assert "356" in str(exc.value), (
            f"message should reference the issue: {exc.value}"
        )

    def test_tissot_circles_raises_not_implemented(self):
        """Test tissot_circles raises NotImplementedError.

        Test scenario:
            The stub must raise, naming itself.
        """
        with pytest.raises(NotImplementedError, match="tissot_circles"):
            tissot_circles([0.0], [0.0], 5e5)

    def test_add_tissot_raises_not_implemented(self):
        """Test add_tissot raises NotImplementedError.

        Test scenario:
            The stub must raise before touching the axes, naming itself.
        """
        with pytest.raises(NotImplementedError, match="add_tissot"):
            add_tissot(None, [])


class TestWrapLongitude:
    """Tests for the private _wrap_longitude helper."""

    @pytest.mark.parametrize(
        "value, expected",
        [
            (0.0, 0.0),
            (190.0, -170.0),
            (-190.0, 170.0),
            (180.0, 180.0),
            (-180.0, 180.0),
            (360.0, 0.0),
            (540.0, 180.0),
        ],
    )
    def test_scalar_wrapping(self, value, expected):
        """Test scalar longitudes wrap into (-180, 180].

        Args:
            value: Input longitude in degrees.
            expected: Expected wrapped longitude.

        Test scenario:
            The half-open convention keeps +180 and maps -180 to +180.
        """
        assert float(solar._wrap_longitude(value)) == pytest.approx(expected), (
            f"_wrap_longitude({value}) != {expected}"
        )

    def test_array_wrapping(self):
        """Test array inputs wrap element-wise.

        Test scenario:
            A vector of out-of-range longitudes maps into (-180, 180] with -180
            promoted to +180.
        """
        out = solar._wrap_longitude(
            np.array([-181.0, 181.0, 360.0, -360.0, 180.0, -180.0])
        )
        expected = np.array([179.0, -179.0, 0.0, 0.0, 180.0, 180.0])
        assert np.allclose(out, expected), f"array wrap wrong: {out}"


class TestSplitAntimeridian:
    """Tests for the private _split_antimeridian helper."""

    def test_straddling_ring_splits_in_two(self):
        """Test a ring around the antimeridian splits into two pieces.

        Test scenario:
            A box centred on lon 180 (given wrapped) unwraps to a continuous
            polygon and clips into two rings, one on each side of +/-180.
        """
        ring = np.array(
            [
                [178.0, 10.0],
                [-178.0, 10.0],
                [-178.0, -10.0],
                [178.0, -10.0],
                [178.0, 10.0],
            ]
        )
        rings = solar._split_antimeridian(ring)
        assert len(rings) == 2, f"expected 2 rings, got {len(rings)}"
        for piece in rings:
            assert np.all(piece[:, 0] >= -180.0 - 1e-9), (
                f"piece lon < -180: {piece[:, 0]}"
            )
            assert np.all(piece[:, 0] <= 180.0 + 1e-9), (
                f"piece lon > 180: {piece[:, 0]}"
            )

    def test_non_straddling_ring_stays_single(self):
        """Test a ring away from the antimeridian is returned as one piece.

        Test scenario:
            A box centred on lon 0 clips to a single ring.
        """
        ring = np.array(
            [[-10.0, 10.0], [10.0, 10.0], [10.0, -10.0], [-10.0, -10.0], [-10.0, 10.0]]
        )
        rings = solar._split_antimeridian(ring)
        assert len(rings) == 1, f"expected 1 ring, got {len(rings)}"


class TestClipAndCapHelpers:
    """Direct unit tests for the private clip and pole-cap helpers."""

    @pytest.mark.parametrize(
        "p, q, x, expected",
        [
            ([0.0, 0.0], [10.0, 10.0], 5.0, [5.0, 5.0]),
            ([0.0, 0.0], [4.0, 8.0], 2.0, [2.0, 4.0]),
            ([-2.0, 3.0], [2.0, -1.0], 0.0, [0.0, 1.0]),
        ],
    )
    def test_intersect_x(self, p, q, x, expected):
        """Test the segment/vertical-line intersection point.

        Args:
            p: Segment start ``[x, y]``.
            q: Segment end ``[x, y]``.
            x: Vertical line to cross.
            expected: Expected ``[x, y]`` crossing point.

        Test scenario:
            The crossing latitude is linearly interpolated at the given x.
        """
        out = solar._intersect_x(np.array(p), np.array(q), x)
        assert np.allclose(out, expected), f"_intersect_x -> {out}, expected {expected}"

    def test_clip_halfplane_keeps_inside_part(self):
        """Test half-plane clipping keeps the inside and adds the boundary crossing.

        Test scenario:
            Clipping a unit square spanning x in [0, 10] to x <= 5 leaves a
            polygon bounded by x = 5.
        """
        square = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 4.0], [0.0, 4.0]])
        clipped = solar._clip_halfplane(square, 5.0, keep_below=True)
        assert len(clipped) >= 3, f"expected a polygon, got {clipped}"
        assert clipped[:, 0].max() == pytest.approx(5.0), "not clipped at x=5"
        assert clipped[:, 0].min() == pytest.approx(0.0), "left edge lost"

    def test_clip_halfplane_empty_when_all_outside(self):
        """Test half-plane clipping returns empty when nothing is inside.

        Test scenario:
            A square entirely at x > 5 clipped to x <= 5 yields no vertices.
        """
        square = np.array([[6.0, 0.0], [10.0, 0.0], [10.0, 4.0], [6.0, 4.0]])
        clipped = solar._clip_halfplane(square, 5.0, keep_below=True)
        assert len(clipped) == 0, f"expected empty, got {clipped}"

    def test_clip_lon_strip_bounds_longitudes(self):
        """Test strip clipping bounds a wide polygon to [-180, 180].

        Test scenario:
            A polygon spanning lon -250..250 is clipped to the map strip.
        """
        poly = np.array(
            [[-250.0, 10.0], [250.0, 10.0], [250.0, -10.0], [-250.0, -10.0]]
        )
        clipped = solar._clip_lon_strip(poly, -180.0, 180.0)
        assert clipped[:, 0].min() == pytest.approx(-180.0), "left not clipped"
        assert clipped[:, 0].max() == pytest.approx(180.0), "right not clipped"

    @pytest.mark.parametrize("dark_lat", [90.0, -90.0])
    def test_pole_cap_ring_closes_on_pole_edge(self, dark_lat):
        """Test the pole-cap closure runs the ring to the dark pole across the map.

        Args:
            dark_lat: The dark pole's latitude (+90 or -90).

        Test scenario:
            Closing a terminator ring adds the dark pole's map edge, so the ring
            reaches ``dark_lat`` and spans the full [-180, 180] longitude range.
        """
        ring = terminator(JUN_SOLSTICE, n=180)
        cap = solar._pole_cap_ring(ring, dark_lat)
        edge = cap[:, 1].max() if dark_lat > 0 else cap[:, 1].min()
        assert edge == pytest.approx(dark_lat), f"cap did not reach {dark_lat}"
        assert cap[:, 0].min() == pytest.approx(-180.0), "cap missing left edge"
        assert cap[:, 0].max() == pytest.approx(180.0), "cap missing right edge"
