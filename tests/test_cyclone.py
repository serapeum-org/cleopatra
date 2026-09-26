"""Tests for the tropical-cyclone animation overlay (issue #372, part 2).

Cover the Saffir-Simpson classifier, the rapid-intensification detector, the
`CycloneOverlay` `FrameOverlay` (track / glow / ripple / wind-radii / tags,
driven through `ArrayGlyph.animate`'s blit funcs), and the intensity-key helper.
"""

import numpy as np
import pytest
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle
from matplotlib.pyplot import close, subplots

from cleopatra.glyphs.gridded.array_glyph import Animation, ArrayGlyph
from cleopatra.glyphs.gridded.cyclone import (
    SAFFIR_SIMPSON,
    CycloneOverlay,
    add_intensity_key,
    category_of,
    rapid_intensification_mask,
)


def _track(**overrides):
    """Return a small 4-fix storm track (dict of arrays), with column overrides."""
    track = {
        "lon": np.array([-120.0, -121.0, -122.0, -123.0]),
        "lat": np.array([15.0, 16.0, 17.0, 18.0]),
        "vmax_kt": np.array([30.0, 55.0, 90.0, 130.0]),
        "time": np.array([0.0, 24.0, 48.0, 72.0]),
    }
    track.update(overrides)
    return track


def _driven(overlay, n_frames=4, sub_frames=1):
    """Build an ArrayGlyph animation over `overlay`; return its FuncAnimation."""
    stack = np.zeros((n_frames, 8, 8, 3), dtype=float)
    glyph = ArrayGlyph(stack)
    return glyph.animate(
        [f"t{i}" for i in range(n_frames)],
        playback=Animation(overlays=[overlay], sub_frames=sub_frames),
    )


class TestCategoryOf:
    """Tests for the Saffir-Simpson classifier `category_of`."""

    @pytest.mark.parametrize(
        "vmax, label",
        [(20, "TD"), (40, "TS"), (70, "Cat 1"), (100, "Cat 3"), (140, "Cat 5")],
    )
    def test_category_thresholds(self, vmax, label):
        """Each wind maps to the strongest category whose threshold it meets.

        Args:
            vmax: Sustained wind in knots.
            label: The expected Saffir-Simpson label.
        """
        assert category_of(float(vmax))[0] == label, f"{vmax} kt -> {label}"

    def test_nan_maps_to_weakest(self):
        """A NaN wind falls to the weakest category rather than raising."""
        assert category_of(float("nan"))[0] == "TD"

    def test_inf_maps_to_strongest(self):
        """A wind above every threshold (incl. +inf) maps to the strongest category."""
        assert category_of(float("inf"))[0] == "Cat 5"
        assert category_of(float("-inf"))[0] == "TD"

    def test_returns_palette_colour(self):
        """The returned colour is the palette entry for that category."""
        assert category_of(140.0)[1] == SAFFIR_SIMPSON[-1][2]


class TestRapidIntensificationMask:
    """Tests for the rapid-intensification detector."""

    def test_flags_full_window_gain(self):
        """A >=30 kt gain over a full 24 h window flags the later fixes."""
        t = np.array([0.0, 24.0, 48.0])
        v = np.array([30.0, 65.0, 100.0])
        assert rapid_intensification_mask(t, v).tolist() == [False, True, True]

    def test_no_baseline_is_false(self):
        """A fix with no fix a full window before it is never flagged."""
        t = np.array([0.0, 6.0])
        v = np.array([30.0, 100.0])
        assert rapid_intensification_mask(t, v).tolist() == [False, False]

    def test_gain_below_threshold_not_flagged(self):
        """A gain below the threshold over the window is not flagged."""
        t = np.array([0.0, 24.0])
        v = np.array([30.0, 50.0])
        assert rapid_intensification_mask(t, v).tolist() == [False, False]


class TestCycloneOverlay:
    """Tests for `CycloneOverlay` normalisation and per-frame drawing."""

    def test_normalise_single_track(self):
        """A bare table (carrying a `lon` column) is one unnamed storm."""
        overlay = CycloneOverlay(_track())
        assert list(overlay.storms) == [""]

    def test_normalise_named_tracks(self):
        """A mapping of name to table keeps the storm names."""
        overlay = CycloneOverlay({"Karina": _track(), "Marie": _track()})
        assert sorted(overlay.storms) == ["Karina", "Marie"]

    def test_missing_lonlat_raises(self):
        """A table without lon/lat is rejected with a clear error."""
        bad = {"X": {"vmax_kt": np.array([30.0])}}
        with pytest.raises(ValueError, match="needs 'lon' and 'lat'"):
            CycloneOverlay(bad)

    def test_empty_track_raises_clear_error(self):
        """A present-but-empty lon/lat raises a clear error, not a cryptic IndexError."""
        empty = {"lon": np.array([]), "lat": np.array([]), "vmax_kt": np.array([])}
        with pytest.raises(ValueError, match="non-empty 1-D"):
            CycloneOverlay(empty)

    def test_scalar_lon_raises_clear_error(self):
        """A scalar (0-d) lon is rejected instead of failing opaquely later."""
        with pytest.raises(ValueError, match="non-empty 1-D"):
            CycloneOverlay({"lon": 5.0, "lat": 3.0})

    def test_column_length_mismatch_raises(self):
        """A column shorter than lon is rejected at construction, not mid-render."""
        bad = {
            "lon": np.array([0.0, 1.0, 2.0]),
            "lat": np.array([0.0, 1.0, 2.0]),
            "vmax_kt": np.array([30.0, 40.0]),
        }
        with pytest.raises(ValueError, match="expected 3 to match"):
            CycloneOverlay(bad)

    def test_init_creates_and_returns_artists(self):
        """`init` adds the storm's artists to the axes and returns them."""
        overlay = CycloneOverlay({"K": _track(r34_ne=np.array([1.0, 1.0, 1.0, 1.0]))})
        out = _driven(overlay)._init_func()
        art = overlay._artists["K"]
        assert isinstance(art.track, LineCollection), "track is a LineCollection"
        assert isinstance(art.glow, Circle), "glow is a Circle"
        assert len(art.wedges) == 1, "one r34_ne wind-radius wedge"
        assert art.track in out and art.glow in out, "artists returned for blitting"

    def test_track_segments_and_category_colour(self):
        """The track shows one segment per shown gap, coloured by category."""
        overlay = CycloneOverlay({"K": _track()})
        anim = _driven(overlay)
        anim._init_func()
        anim._func(3)
        art = overlay._artists["K"]
        assert len(art.track.get_segments()) == 3, "three segments for four fixes"
        last = tuple(art.track.get_colors()[-1])
        assert last == to_rgba(category_of(130.0)[1]), (
            "last segment is the Cat-4 colour"
        )

    def test_ripple_grows_with_phase(self):
        """Across the sub-frame phases of a held frame the ripple radius changes."""
        overlay = CycloneOverlay({"K": _track()})
        anim = _driven(overlay, sub_frames=4)
        anim._init_func()
        anim._func(4)
        r0 = overlay._artists["K"].ripple.get_radius()
        anim._func(5)
        r1 = overlay._artists["K"].ripple.get_radius()
        assert r0 != r1, f"ripple should expand with phase, got {r0} then {r1}"

    def test_ri_ripple_animates_at_sub_frames_two(self):
        """A rapidly-intensifying storm's ripple still moves at sub_frames=2."""
        overlay = CycloneOverlay({"K": _track()})  # fix 2 is rapidly intensifying
        assert overlay._ri["K"][2], "fixture fix 2 must be rapidly intensifying"
        anim = _driven(overlay, sub_frames=2)
        anim._init_func()
        anim._func(4)  # data_index 2, phase 0.0
        ripple = overlay._artists["K"].ripple
        r0, a0 = ripple.get_radius(), ripple.get_alpha()
        anim._func(5)  # data_index 2, phase 0.5
        r1, a1 = ripple.get_radius(), ripple.get_alpha()
        assert r0 != r1, f"RI ripple must expand at sub_frames=2, got {r0} then {r1}"
        assert a0 > 0 and a1 > 0, f"RI ripple must stay visible, alphas {a0}, {a1}"

    def test_dissipation_hides_glow_and_dims_track(self):
        """Past the storm's last fix the glow is hidden and the track dims."""
        overlay = CycloneOverlay({"K": _track()})
        anim = _driven(overlay, n_frames=6)
        anim._init_func()
        anim._func(5)
        art = overlay._artists["K"]
        assert tuple(art.glow.get_edgecolor())[3] == 0.0, (
            "glow hidden after dissipation"
        )
        assert art.track.get_alpha() == 0.5, "track dimmed to half after dissipation"

    def test_tag_wording(self):
        """The name tag reads name / category / mph."""
        overlay = CycloneOverlay({"Karina": _track()})
        anim = _driven(overlay)
        anim._init_func()
        anim._func(3)
        assert overlay._artists["Karina"].tag.get_text() == "Karina\nCat 4  150 mph"

    def test_effect_toggles_suppress_artists(self):
        """Switching effects off drops their artists / hides them."""
        overlay = CycloneOverlay(
            {"K": _track(r34_ne=np.array([1.0, 1.0, 1.0, 1.0]))},
            show_wind_radii=False,
            show_tags=False,
        )
        anim = _driven(overlay)
        anim._init_func()
        anim._func(2)
        art = overlay._artists["K"]
        assert art.wedges == [], "no wind-radii wedges when show_wind_radii=False"
        assert not art.tag.get_visible(), "tag hidden when show_tags=False"

    def test_show_track_false_leaves_track_empty(self):
        """With `show_track=False` the track/halo carry no segments."""
        overlay = CycloneOverlay({"K": _track()}, show_track=False)
        anim = _driven(overlay)
        anim._init_func()
        anim._func(3)
        assert overlay._artists["K"].track.get_segments() == [], "track suppressed"

    def test_datetime_time_column_coerced_to_hours(self):
        """A datetime64 `time` column is normalised to hours from the first fix."""
        track = _track(
            time=np.array(["2026-09-02", "2026-09-03"], dtype="datetime64[D]"),
            lon=np.array([-120.0, -121.0]),
            lat=np.array([15.0, 16.0]),
            vmax_kt=np.array([35.0, 60.0]),
        )
        overlay = CycloneOverlay({"K": track})
        assert overlay.storms["K"]["hours"].tolist() == [0.0, 24.0]

    def test_wind_radii_wedge_updates_to_fix_radius(self):
        """A present wind-radius wedge becomes visible with the fix's radius."""
        overlay = CycloneOverlay({"K": _track(r34_ne=np.array([1.0, 2.0, 3.0, 4.0]))})
        anim = _driven(overlay)
        anim._init_func()
        anim._func(2)
        wedge, key = overlay._artists["K"].wedges[0]
        assert key == "r34_ne"
        assert wedge.get_visible(), "wedge visible at a positive radius"
        assert wedge.r == 3.0, f"wedge radius tracks the fix, got {wedge.r}"

    def test_wind_radii_wedge_hidden_at_zero_radius(self):
        """A wind-radius of 0 at a fix hides that quadrant's wedge."""
        overlay = CycloneOverlay({"K": _track(r34_ne=np.array([0.0, 0.0, 0.0, 0.0]))})
        anim = _driven(overlay)
        anim._init_func()
        anim._func(2)
        wedge, _ = overlay._artists["K"].wedges[0]
        assert not wedge.get_visible(), "a zero wind-radius wedge stays hidden"

    def test_non_datetime_object_time_falls_back_to_float_hours(self):
        """An object-dtype numeric `time` that is not datetime is read as hours."""
        track = _track(
            time=np.array([0.0, 24.0], dtype=object),
            lon=np.array([-120.0, -121.0]),
            lat=np.array([15.0, 16.0]),
            vmax_kt=np.array([35.0, 60.0]),
        )
        overlay = CycloneOverlay({"K": track})
        assert overlay.storms["K"]["hours"].tolist() == [0.0, 24.0]

    def test_object_int_time_read_as_hours_not_seconds(self):
        """An object-dtype integer `time` is read as hours, not seconds-since-epoch."""
        track = _track(
            time=np.array([0, 24, 48], dtype=object),
            lon=np.array([-120.0, -121.0, -122.0]),
            lat=np.array([15.0, 16.0, 17.0]),
            vmax_kt=np.array([35.0, 60.0, 90.0]),
        )
        overlay = CycloneOverlay({"K": track})
        assert overlay.storms["K"]["hours"].tolist() == [0.0, 24.0, 48.0]

    def test_object_datetime_strings_time_parsed_to_hours(self):
        """An object-dtype array of ISO date strings is parsed to hours."""
        track = _track(
            time=np.array(["2026-09-02", "2026-09-03"], dtype=object),
            lon=np.array([-120.0, -121.0]),
            lat=np.array([15.0, 16.0]),
            vmax_kt=np.array([35.0, 60.0]),
        )
        overlay = CycloneOverlay({"K": track})
        assert overlay.storms["K"]["hours"].tolist() == [0.0, 24.0]


class TestAddIntensityKey:
    """Tests for the line-swatch intensity-key helper."""

    def test_row_per_category_plus_title(self):
        """The key draws a title plus a line and label for each category."""
        fig, ax = subplots()
        artists = add_intensity_key(ax)
        close(fig)
        assert len(artists) == 1 + 2 * len(SAFFIR_SIMPSON), f"got {len(artists)}"
