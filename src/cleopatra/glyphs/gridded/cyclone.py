"""A tropical-cyclone animation overlay for `ArrayGlyph.animate` (issue #372).

`CycloneOverlay` is a `FrameOverlay` (the per-frame hook `animate` grew in part 1)
that draws the furniture of a hurricane animation on top of each satellite frame,
per frame, from a per-storm track table:

- a **category-coloured track** behind each storm, coloured segment-by-segment by
  Saffir-Simpson category, sitting on a dark halo so it reads over bright cloud,
  and persisting (at half strength) after the storm's last fix;
- a hollow **eye glow** in the category colour whose radius scales with wind speed
  and pulses with the sub-frame `phase`;
- **wind-radii** outlines (34/50/64 kt) drawn per quadrant when the track carries
  them;
- a **ripple ring** expanding from the eye each data frame, cycling twice as fast
  for a rapidly-intensifying storm;
- a **name tag** (name, category, mph) with a thin leader line.

Everything is drawn in the axes' **data coordinates**, so the `ArrayGlyph` must be
built with a geographic `extent` whose units match the track's `lon`/`lat` (and the
wind-radii, which are read in those same units). Fetching the track data is out of
scope for cleopatra -- pass an in-memory table (a `pandas.DataFrame` or a plain
`dict` of arrays); pandas is not required.

Example:
    >>> import numpy as np
    >>> from cleopatra.glyphs.gridded.cyclone import CycloneOverlay
    >>> track = {
    ...     "lon": np.array([-120.0, -121.0, -122.0]),
    ...     "lat": np.array([15.0, 16.0, 17.0]),
    ...     "vmax_kt": np.array([35.0, 70.0, 100.0]),
    ...     "time": np.array([0.0, 6.0, 12.0]),  # hours
    ... }
    >>> overlay = CycloneOverlay({"Karina": track})
    >>> sorted(overlay.storms)
    ['Karina']
    >>> overlay.storms["Karina"]["vmax_kt"][-1]
    np.float64(100.0)
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Wedge
from matplotlib.text import Text

#: Saffir-Simpson categories as `(label, minimum sustained wind in kt, colour)`,
#: weakest first. A fix's category is the last entry whose `vmax_kt` threshold it
#: meets: tropical depression (grey), tropical storm (blue), then Cat 1-5 running
#: yellow, amber, orange, red, pink.
SAFFIR_SIMPSON: tuple[tuple[str, float, str], ...] = (
    ("TD", 0.0, "#9e9e9e"),
    ("TS", 34.0, "#3d8fe0"),
    ("Cat 1", 64.0, "#f6e60c"),
    ("Cat 2", 83.0, "#ffb302"),
    ("Cat 3", 96.0, "#ff7a00"),
    ("Cat 4", 113.0, "#e00000"),
    ("Cat 5", 137.0, "#ff6fd6"),
)

#: Default rapid-intensification threshold: a >= 30 kt gain in 24 h (the NHC
#: definition).
RAPID_INTENSIFICATION_KT: float = 30.0

_QUADRANTS: tuple[tuple[str, float, float], ...] = (
    ("ne", 0.0, 90.0),
    ("se", 270.0, 360.0),
    ("sw", 180.0, 270.0),
    ("nw", 90.0, 180.0),
)
_WIND_RADII_KT: tuple[int, ...] = (34, 50, 64)


def category_of(
    vmax_kt: float, palette: tuple[tuple[str, float, str], ...] = SAFFIR_SIMPSON
) -> tuple[str, str]:
    """Return the `(label, colour)` for a sustained wind in knots.

    Args:
        vmax_kt: Maximum sustained wind, in knots.
        palette: Ordered `(label, min_kt, colour)` categories, weakest first.

    Returns:
        The `(label, colour)` of the strongest category whose threshold
        `vmax_kt` meets (the weakest category for a wind below every threshold,
        including a NaN wind).

    Examples:
        - A hurricane-force wind maps to its Saffir-Simpson category:
            ```python
            >>> from cleopatra.glyphs.gridded.cyclone import category_of
            >>> category_of(100.0)[0]
            'Cat 3'

            ```
        - A gale maps to a tropical storm; calm/NaN to the weakest category:
            ```python
            >>> from cleopatra.glyphs.gridded.cyclone import category_of
            >>> category_of(40.0)[0]
            'TS'
            >>> category_of(float("nan"))[0]
            'TD'

            ```
    """
    label, colour = palette[0][0], palette[0][2]
    if np.isfinite(vmax_kt):
        for name, threshold, col in palette:
            if vmax_kt >= threshold:
                label, colour = name, col
    return label, colour


def rapid_intensification_mask(
    time_hours: np.ndarray,
    vmax_kt: np.ndarray,
    threshold_kt: float = RAPID_INTENSIFICATION_KT,
    window_h: float = 24.0,
) -> np.ndarray:
    """Flag each fix that is rapidly intensifying over the trailing window.

    A fix `i` is rapidly intensifying when `vmax` rose by at least `threshold_kt`
    from the latest earlier fix at or before `window_h` hours before it.

    Args:
        time_hours: Fix times as hours from an arbitrary origin, ascending.
        vmax_kt: Maximum sustained wind per fix, in knots.
        threshold_kt: Minimum wind gain over the window. Defaults to 30 kt.
        window_h: Trailing window in hours. Defaults to 24.

    Returns:
        A boolean array, one flag per fix; `False` for fixes younger than the
        window (no baseline yet).

    Examples:
        - The fix a full 24 h after a 45 kt gain is flagged; the 12 h fix has no
          24 h-old baseline yet, so it is not:
            ```python
            >>> import numpy as np
            >>> from cleopatra.glyphs.gridded.cyclone import rapid_intensification_mask
            >>> t = np.array([0.0, 12.0, 24.0])
            >>> v = np.array([35.0, 60.0, 80.0])
            >>> rapid_intensification_mask(t, v).tolist()
            [False, False, True]

            ```
        - A gain under the threshold over the window is not flagged:
            ```python
            >>> import numpy as np
            >>> from cleopatra.glyphs.gridded.cyclone import rapid_intensification_mask
            >>> t = np.array([0.0, 24.0])
            >>> v = np.array([40.0, 60.0])  # +20 kt < 30 kt
            >>> rapid_intensification_mask(t, v).tolist()
            [False, False]

            ```
    """
    time_hours = np.asarray(time_hours, dtype=float)
    vmax_kt = np.asarray(vmax_kt, dtype=float)
    mask = np.zeros(vmax_kt.shape, dtype=bool)
    for i in range(len(vmax_kt)):
        earlier = np.nonzero(time_hours <= time_hours[i] - window_h)[0]
        if earlier.size:
            baseline = vmax_kt[earlier[-1]]
            mask[i] = np.isfinite(baseline) and (vmax_kt[i] - baseline) >= threshold_kt
    return mask


def _as_hours(time: Any) -> np.ndarray:
    """Coerce a `time` column to hours from its first entry.

    A `datetime64` column is parsed to hours since its first fix. Numeric values
    (including an object-dtype array of Python ints/floats) are already hours and
    taken as-is; only when a float cast fails is an object column parsed as
    datetime objects / ISO strings.
    """
    arr = np.asarray(time)
    if np.issubdtype(arr.dtype, np.datetime64):
        dt = arr.astype("datetime64[s]")
        return (dt - dt[0]) / np.timedelta64(1, "h")
    if arr.dtype == object:
        try:
            return np.asarray(arr, dtype=float)  # numeric objects are hours
        except (ValueError, TypeError):
            dt = arr.astype("datetime64[s]")  # datetime objects / ISO strings
            return (dt - dt[0]) / np.timedelta64(1, "h")
    return np.asarray(arr, dtype=float)


def _column(table: Mapping[str, Any], key: str) -> np.ndarray | None:
    """Read `table[key]` as a float array, or `None` when the key is absent."""
    if key not in table:
        return None
    return np.asarray(table[key], dtype=float)


def _normalise_tracks(
    tracks: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Normalise the `tracks` argument to `{name: {column: array}}`.

    Accepts either a single storm's table (a mapping with a `lon` column) or a
    mapping of storm name to table. Each table is read column-by-column into
    numpy arrays via duck typing (a `pandas.DataFrame` or a `dict` of arrays both
    work); the `time` column is coerced to hours.
    """
    is_single = "lon" in tracks and not isinstance(tracks.get("lon"), Mapping)
    raw = {"": tracks} if is_single else tracks
    storms: dict[str, dict[str, Any]] = {}
    for name, table in raw.items():
        lon = _column(table, "lon")
        lat = _column(table, "lat")
        if lon is None or lat is None:
            raise ValueError(
                f"track {name!r} needs 'lon' and 'lat' columns (a pandas "
                f"DataFrame or a dict of arrays, not a dict of dicts)."
            )
        if lon.ndim != 1 or lon.size == 0:
            raise ValueError(
                f"track {name!r} needs a non-empty 1-D 'lon'/'lat' (got shape "
                f"{lon.shape})."
            )
        n = len(lon)
        vmax = _column(table, "vmax_kt")
        vmax = np.full(lon.shape, np.nan) if vmax is None else vmax
        time = np.asarray(table["time"]) if "time" in table else np.arange(n)
        columns: dict[str, Any] = {"lat": lat, "vmax_kt": vmax, "time": time}
        for kt in _WIND_RADII_KT:
            for quad, _, _ in _QUADRANTS:
                radius = _column(table, f"r{kt}_{quad}")
                if radius is not None:
                    columns[f"r{kt}_{quad}"] = radius
        for col_name, arr in columns.items():
            if len(arr) != n:
                raise ValueError(
                    f"track {name!r} column {col_name!r} has length {len(arr)}, "
                    f"expected {n} to match 'lon'."
                )
        storm: dict[str, Any] = {
            "lon": lon,
            "lat": lat,
            "vmax_kt": vmax,
            "hours": _as_hours(time),
        }
        for key, arr in columns.items():
            if key.startswith("r"):
                storm[key] = arr
        storms[name] = storm
    return storms


@dataclass
class _StormArtists:
    """The mutable matplotlib artists one storm owns across frames."""

    halo: LineCollection
    track: LineCollection
    glow: Circle
    ripple: Circle
    tag: Text
    leader: Line2D
    wedges: list[tuple[Wedge, str]]


class CycloneOverlay:
    """A `FrameOverlay` drawing a tropical-cyclone track and effects per frame.

    Satisfies the `FrameOverlay` protocol (`init(ax)` / `update(frame_index,
    phase)`), so it is passed to
    `animate(playback=Animation(overlays=[CycloneOverlay(...)]))`. The animation's
    data-frame index selects the fix to draw up to; `sub_frames` gives the glow
    and ripple a `phase` to animate against while the imagery is held.

    Args:
        tracks: One storm's track table, or a mapping of storm name to table.
            A table is anything with `lon`/`lat`/`vmax_kt`/`time` columns
            (`pandas.DataFrame` or a `dict` of arrays), optionally the wind-radii
            columns `r34_ne`/`r34_se`/... for 34/50/64 kt. `lon`/`lat`/radii are
            in the axes' data coordinates.
        palette: Saffir-Simpson `(label, min_kt, colour)` categories.
        rapid_intensification_kt: Wind gain over 24 h that marks rapid
            intensification (a bolder, twice-as-fast ripple). Defaults to 30 kt.
        show_track / show_glow / show_wind_radii / show_ripple / show_tags:
            Per-effect switches, all on by default.
        tag_offset: `(dx, dy)` in data units from the eye to the name tag.

    Attributes:
        storms: The normalised `{name: {column: array}}` tracks.

    Examples:
        - Build an overlay and read back a storm's category at its peak:
            ```python
            >>> import numpy as np
            >>> from cleopatra.glyphs.gridded.cyclone import CycloneOverlay, category_of
            >>> track = {
            ...     "lon": np.array([-120.0, -121.0]),
            ...     "lat": np.array([15.0, 16.0]),
            ...     "vmax_kt": np.array([35.0, 115.0]),
            ...     "time": np.array([0.0, 24.0]),
            ... }
            >>> overlay = CycloneOverlay({"Marie": track}, show_wind_radii=False)
            >>> category_of(overlay.storms["Marie"]["vmax_kt"][-1])[0]
            'Cat 4'

            ```
    """

    def __init__(
        self,
        tracks: Mapping[str, Any],
        *,
        palette: tuple[tuple[str, float, str], ...] = SAFFIR_SIMPSON,
        rapid_intensification_kt: float = RAPID_INTENSIFICATION_KT,
        show_track: bool = True,
        show_glow: bool = True,
        show_wind_radii: bool = True,
        show_ripple: bool = True,
        show_tags: bool = True,
        tag_offset: tuple[float, float] = (0.8, 0.8),
    ) -> None:
        """Initialise the overlay from one or more track tables (see class doc)."""
        self.storms = _normalise_tracks(tracks)
        self.palette = palette
        self.rapid_intensification_kt = float(rapid_intensification_kt)
        self.show_track = show_track
        self.show_glow = show_glow
        self.show_wind_radii = show_wind_radii
        self.show_ripple = show_ripple
        self.show_tags = show_tags
        self.tag_offset = tag_offset
        self._ri = {
            name: rapid_intensification_mask(
                s["hours"], s["vmax_kt"], self.rapid_intensification_kt
            )
            for name, s in self.storms.items()
        }
        self._artists: dict[str, _StormArtists] = {}

    def init(self, ax: Axes) -> list:
        """Create every storm's artists on `ax` once and return them all.

        Args:
            ax: The animation axes (in geographic data coordinates).

        Returns:
            The flat list of created artists, seeding the blit background.
        """
        self._artists.clear()
        out: list = []
        for name, storm in self.storms.items():
            halo = LineCollection([], colors="black", zorder=4.0)
            track = LineCollection([], zorder=4.1)
            glow = Circle(
                (storm["lon"][0], storm["lat"][0]),
                0.0,
                facecolor="none",
                edgecolor="none",
                zorder=4.3,
            )
            ripple = Circle(
                (storm["lon"][0], storm["lat"][0]),
                0.0,
                facecolor="none",
                edgecolor="none",
                zorder=4.2,
            )
            leader = Line2D([], [], color="white", linewidth=0.8, zorder=4.4)
            tag = ax.text(
                storm["lon"][0],
                storm["lat"][0],
                "",
                color="white",
                fontsize=8,
                zorder=4.5,
                visible=False,
            )
            wedges = self._make_wedges(storm) if self.show_wind_radii else []
            wedge_artists = [w for w, _ in wedges]
            for artist in (halo, track, glow, ripple, leader, *wedge_artists):
                ax.add_artist(artist)
            self._artists[name] = _StormArtists(
                halo, track, glow, ripple, tag, leader, wedges
            )
            out += [halo, track, glow, ripple, leader, tag, *wedge_artists]
        return out

    def _make_wedges(self, storm: dict[str, Any]) -> list[tuple[Wedge, str]]:
        """Create one hollow `Wedge` (paired with its radius column) per quadrant."""
        wedges: list[tuple[Wedge, str]] = []
        for kt in _WIND_RADII_KT:
            for quad, start, end in _QUADRANTS:
                key = f"r{kt}_{quad}"
                if key in storm:
                    wedge = Wedge(
                        (storm["lon"][0], storm["lat"][0]),
                        1e-6,  # nonzero so the initial (r-width)/r path is finite
                        start,
                        end,
                        width=0.0,
                        facecolor="none",
                        edgecolor="white",
                        linewidth=0.6,
                        alpha=0.7,
                        zorder=4.15,
                        visible=False,
                    )
                    wedges.append((wedge, key))
        return wedges

    def update(self, frame_index: int, phase: float) -> list:
        """Refresh every storm's artists for the given data frame and phase.

        Args:
            frame_index: The current data-frame index; the storm is drawn up to
                fix `frame_index` (clamped to its last fix, after which the track
                persists at half strength).
            phase: Sub-frame phase in `[0, 1)`, driving the glow pulse and the
                ripple's expansion.

        Returns:
            Every artist to keep visible this frame (each storm's track, halo,
            glow, ripple, tag, leader and wind-radii wedges).
        """
        out: list = []
        for name, storm in self.storms.items():
            art = self._artists[name]
            last = len(storm["lon"]) - 1
            i = min(frame_index, last)
            dissipated = frame_index > last
            self._update_track(storm, art, i, dissipated)
            self._update_glow(storm, art, i, phase, dissipated)
            self._update_ripple(storm, art, name, i, phase, dissipated)
            self._update_wedges(storm, art, i, dissipated)
            self._update_tag(name, storm, art, i, dissipated)
            out += [
                art.halo,
                art.track,
                art.glow,
                art.ripple,
                art.leader,
                art.tag,
                *[w for w, _ in art.wedges],
            ]
        return out

    def _segments(self, storm: dict[str, Any], i: int):
        """Return the track segments and per-segment colours up to fix `i`."""
        lon, lat = storm["lon"][: i + 1], storm["lat"][: i + 1]
        points = np.column_stack([lon, lat])
        segments = np.stack([points[:-1], points[1:]], axis=1)
        colours = [
            category_of(storm["vmax_kt"][j + 1], self.palette)[1]
            for j in range(len(segments))
        ]
        widths = np.linspace(0.6, 3.0, len(segments)) if len(segments) else []
        return segments, colours, widths

    def _update_track(self, storm, art, i, dissipated) -> None:
        """Draw the category-coloured track (on its dark halo) up to fix `i`."""
        if not self.show_track or i < 1:
            art.halo.set_segments([])
            art.track.set_segments([])
            return
        segments, colours, widths = self._segments(storm, i)
        alpha = 0.5 if dissipated else 1.0
        art.halo.set_segments(segments)
        art.halo.set_linewidth([w + 2.0 for w in widths])
        art.halo.set_alpha(alpha * 0.6)
        art.track.set_segments(segments)
        art.track.set_color(colours)
        art.track.set_linewidth(list(widths))
        art.track.set_alpha(alpha)

    def _update_glow(self, storm, art, i, phase, dissipated) -> None:
        """Position and size the hollow eye glow, pulsing with `phase`."""
        vmax = storm["vmax_kt"][i]
        _, colour = category_of(vmax, self.palette)
        centre = (storm["lon"][i], storm["lat"][i])
        art.glow.set_center(centre)
        if not self.show_glow or dissipated or not np.isfinite(vmax):
            art.glow.set_edgecolor("none")
            return
        base = 0.2 + 0.012 * max(vmax, 0.0)
        art.glow.set_radius(base * (1.0 + 0.15 * np.sin(np.pi * phase)))
        art.glow.set_edgecolor(colour)
        art.glow.set_linewidth(2.0 + 0.03 * max(vmax, 0.0))
        art.glow.set_alpha(0.85)

    def _update_ripple(self, storm, art, name, i, phase, dissipated) -> None:
        """Expand the ripple ring from the eye once per data frame."""
        centre = (storm["lon"][i], storm["lat"][i])
        art.ripple.set_center(centre)
        vmax = storm["vmax_kt"][i]
        if not self.show_ripple or dissipated or not np.isfinite(vmax):
            art.ripple.set_edgecolor("none")
            return
        _, colour = category_of(vmax, self.palette)
        # A rapidly-intensifying storm's ripple reaches full radius twice as fast
        # (clamped, not wrapped with `% 1`, which aliased to a frozen ring at
        # sub_frames == 2). The alpha fades on the *raw* phase, not the clamped
        # growth, so the fast-expanded ring stays visible through the back half of
        # the hold instead of blinking to nothing at phase 0.5.
        rapid = self._ri[name][i]
        grow = min(1.0, phase * 2.0) if rapid else phase
        art.ripple.set_radius((0.3 + 0.014 * max(vmax, 0.0)) * (0.4 + grow))
        art.ripple.set_edgecolor(colour)
        art.ripple.set_linewidth(1.6 if rapid else 1.0)
        art.ripple.set_alpha(max(0.0, 0.7 * (1.0 - phase)))

    def _update_wedges(self, storm, art, i, dissipated) -> None:
        """Point each wind-radius wedge at the eye with the fix's radius."""
        centre = (storm["lon"][i], storm["lat"][i])
        for wedge, key in art.wedges:
            radius = float(storm[key][i])
            visible = self.show_wind_radii and not dissipated and np.isfinite(radius)
            wedge.set_visible(bool(visible) and radius > 0.0)
            if visible and radius > 0.0:
                wedge.set_center(centre)
                wedge.set_radius(radius)
                wedge.set_width(radius)

    def _update_tag(self, name, storm, art, i, dissipated) -> None:
        """Word and place the name tag with its leader line back to the eye."""
        if not self.show_tags:
            art.tag.set_visible(False)
            art.leader.set_data([], [])
            return
        eye = (storm["lon"][i], storm["lat"][i])
        vmax = storm["vmax_kt"][i]
        label, colour = category_of(vmax, self.palette)
        mph = "" if not np.isfinite(vmax) else f"  {round(vmax * 1.15078)} mph"
        art.tag.set_text(f"{name}\n{label}{mph}")
        tx, ty = eye[0] + self.tag_offset[0], eye[1] + self.tag_offset[1]
        art.tag.set_position((tx, ty))
        art.tag.set_color(colour)
        art.tag.set_alpha(0.6 if dissipated else 1.0)
        art.tag.set_visible(True)
        art.leader.set_data([eye[0], tx], [eye[1], ty])
        art.leader.set_alpha(0.4 if dissipated else 0.8)


def add_intensity_key(
    ax: Axes,
    *,
    palette: tuple[tuple[str, float, str], ...] = SAFFIR_SIMPSON,
    title: str = "Storm intensity",
    location: tuple[float, float] = (0.02, 0.02),
) -> list:
    """Draw a line-swatch colour key -- one coloured line per category.

    A companion to `CycloneOverlay`: a small legend of the Saffir-Simpson palette
    as a coloured horizontal line per category with its label, anchored in axes
    fraction coordinates. Returns the created artists (a title `Text`, and a
    `Line2D`+`Text` per category) so a caller can reposition or remove them.

    Args:
        ax: The axes to draw the key on.
        palette: The `(label, min_kt, colour)` categories to show.
        title: The key's heading.
        location: `(x, y)` axes-fraction anchor of the key's bottom-left.

    Returns:
        The list of artists drawn (title text, then a line and label per row).

    Examples:
        - The key has one row per category plus a title:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import matplotlib.pyplot as plt
            >>> from cleopatra.glyphs.gridded.cyclone import add_intensity_key
            >>> fig, ax = plt.subplots()
            >>> artists = add_intensity_key(ax)
            >>> len(artists)
            15
            >>> plt.close(fig)

            ```
        - The title text is the heading given:
            ```python
            >>> import matplotlib
            >>> matplotlib.use("Agg")
            >>> import matplotlib.pyplot as plt
            >>> from cleopatra.glyphs.gridded.cyclone import add_intensity_key
            >>> fig, ax = plt.subplots()
            >>> artists = add_intensity_key(ax, title="Category")
            >>> artists[0].get_text()
            'Category'
            >>> plt.close(fig)

            ```
    """
    x0, y0 = location
    row = 0.035
    artists: list = [
        ax.text(
            x0,
            y0 + row * (len(palette) + 0.5),
            title,
            transform=ax.transAxes,
            color="white",
            fontsize=9,
            fontweight="bold",
            zorder=5.0,
        )
    ]
    for idx, (label, _, colour) in enumerate(reversed(palette)):
        y = y0 + row * (len(palette) - 1 - idx)
        artists.append(
            Line2D(
                [x0, x0 + 0.03],
                [y + row * 0.4, y + row * 0.4],
                transform=ax.transAxes,
                color=colour,
                linewidth=3.0,
                zorder=5.0,
            )
        )
        ax.add_line(artists[-1])
        artists.append(
            ax.text(
                x0 + 0.04,
                y + row * 0.15,
                label,
                transform=ax.transAxes,
                color="white",
                fontsize=8,
                zorder=5.0,
            )
        )
    return artists
