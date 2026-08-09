"""Grouped rendering-parameter objects.

Each class here bundles a family of related options that glyphs used to
accept as loose keyword arguments into one discoverable object, exposing a
`to_options()` that flattens the *explicitly set* fields back into the flat
`default_options` keys the rendering engine reads. Only fields the caller
set are emitted, so passing a group never clobbers a glyph's own defaults,
and a glyph applies only the emitted keys it actually supports (see
`cleopatra.glyphs.base.glyph.Glyph._merge_group_params`).

The colour-scale group lives separately in
`cleopatra.styling.scaling.ColorScaling` (it also owns the norm-building
logic); this module holds the remaining groups: `Contour`, `CellValues`,
`DataStyle`, and `Classify`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Contour:
    """Contour discretisation and inline-label options.

    Groups the `levels` / `labels` / `label_kw` options. `levels` applies
    to every colour-mapped glyph that discretises a scale (array, vector,
    flow, polygon, scatter, kde); `labels` / `label_kw` draw inline numeric
    labels on isolines and are honoured only by the glyphs that render
    contour lines (`ArrayGlyph` with `kind="contour"`, `MeshGlyph` node
    contours).

    Attributes:
        levels: Discrete colour levels -- an int count or an explicit
            sequence of edges. `None` leaves the scale continuous.
        labels: Draw inline numeric labels on isolines. `None` leaves the
            glyph default (`False`).
        label_kw: Extra keyword arguments forwarded to `ax.clabel` when
            `labels` is true.

    Examples:
        - Only the set fields are emitted:
            ```python
            >>> from cleopatra.styling.params import Contour
            >>> Contour(levels=5).to_options()
            {'levels': 5}
            >>> Contour(labels=True, label_kw={"fontsize": 8}).to_options()
            {'labels': True, 'label_kw': {'fontsize': 8}}

            ```
    """

    levels: int | Sequence[float] | None = None
    labels: bool | None = None
    label_kw: dict[str, Any] | None = None

    def to_options(self) -> dict[str, Any]:
        """Flatten the explicitly-set fields into `default_options` keys.

        Returns:
            dict: `levels` / `labels` / `label_kw` for the fields that were
                set (non-`None`); an empty dict when nothing was set.
        """
        options: dict[str, Any] = {}
        if self.levels is not None:
            options["levels"] = self.levels
        if self.labels is not None:
            options["labels"] = self.labels
        if self.label_kw is not None:
            options["label_kw"] = self.label_kw
        return options


@dataclass(frozen=True)
class CellValues:
    """Per-cell value-text display options (`ArrayGlyph` only).

    Groups the `display_cell_value` / `num_size` /
    `background_color_threshold` options that overlay each cell's numeric
    value on an `imshow` / `pcolormesh` render.

    Attributes:
        show: Draw each cell's value as text. `None` leaves the glyph
            default (`False`).
        size: Font size of the cell-value text. `None` leaves the default.
        background_threshold: Value above which the text switches to the
            light colour (for contrast against a dark cell). `None` leaves
            the default (`max(array) / 2`).

    Examples:
        - Enable the overlay with a custom font size:
            ```python
            >>> from cleopatra.styling.params import CellValues
            >>> CellValues(show=True, size=10).to_options()
            {'display_cell_value': True, 'num_size': 10}

            ```
    """

    show: bool | None = None
    size: int | None = None
    background_threshold: float | None = None

    def to_options(self) -> dict[str, Any]:
        """Flatten the explicitly-set fields into `default_options` keys.

        Returns:
            dict: `display_cell_value` / `num_size` /
                `background_color_threshold` for the fields that were set.
        """
        options: dict[str, Any] = {}
        if self.show is not None:
            options["display_cell_value"] = self.show
        if self.size is not None:
            options["num_size"] = self.size
        if self.background_threshold is not None:
            options["background_color_threshold"] = self.background_threshold
        return options


class _Unset:
    """Sentinel type marking a `DataStyle` field the caller did not set."""

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return "<unset>"


#: Single sentinel instance for `DataStyle`'s unset fields. Distinguishes
#: "not passed" (keep the glyph's current sticky value) from an explicit
#: `None` (clear the preset / disable hillshade), which a plain `None`
#: default could not.
_UNSET = _Unset()


@dataclass(frozen=True)
class DataStyle:
    """Named data-style preset and relief-shading options.

    Groups the `style` / `hillshade` options honoured by `ArrayGlyph`,
    `MeshGlyph`, and `KDEGlyph`. Each field has three states: left unset
    (keep the glyph's current value -- these options are sticky), set to a
    value (apply it), or set explicitly to `None` (clear the preset /
    disable hillshade). `to_options()` emits a key only for a field that
    was given (set or explicit `None`), never for an unset one.

    Attributes:
        style: Name of a `cleopatra.styling.colors.DATA_STYLES` preset, or
            `None` to clear a sticky preset back to plain colouring.
        hillshade: Relief-shade a regular-grid DEM -- `True` for defaults,
            a dict tuning `vert_exag` / `azimuth` / `altitude` /
            `blend_mode` / `multidirectional`, or `None`/`False` to
            disable.

    Examples:
        - Select a preset and turn on relief shading:
            ```python
            >>> from cleopatra.styling.params import DataStyle
            >>> DataStyle(style="dem", hillshade=True).to_options()
            {'style': 'dem', 'hillshade': True}

            ```
        - An unset field is omitted (keeping the sticky value); an
            explicit `None` is emitted (clearing it):
            ```python
            >>> from cleopatra.styling.params import DataStyle
            >>> DataStyle().to_options()
            {}
            >>> DataStyle(style=None).to_options()
            {'style': None}

            ```
    """

    style: str | None | _Unset = _UNSET
    hillshade: bool | dict[str, Any] | None | _Unset = _UNSET

    def to_options(self) -> dict[str, Any]:
        """Flatten the explicitly-given fields into `default_options` keys.

        Returns:
            dict: `style` / `hillshade` for the fields the caller gave (a
                value or an explicit `None`); unset fields are omitted.
        """
        options: dict[str, Any] = {}
        if not isinstance(self.style, _Unset):
            options["style"] = self.style
        if not isinstance(self.hillshade, _Unset):
            options["hillshade"] = self.hillshade
        return options


@dataclass(frozen=True)
class Classify:
    """Value-classification (choropleth) options.

    Groups the `scheme` / `k` / `category_legend_kwargs` options honoured
    by the glyphs whose colour mapping routes through
    `Glyph._prepare_scalar_mapping` -- `VectorGlyph`, `FlowGlyph`,
    `PolygonGlyph`, `ScatterGlyph`.

    Attributes:
        scheme: A `cleopatra.styling.styles.classify` scheme name (e.g.
            `"quantiles"`, `"equal_interval"`), an explicit sequence of bin
            edges, or the literal `"categorical"` for a distinct-value
            mapping. `None` leaves the default (no classification).
        k: The class count for count/width schemes. `None` leaves the
            default (`5`).
        category_legend_kwargs: Extra keyword arguments forwarded to the
            legend a `"categorical"` scheme draws (e.g. `loc`, `ncol`).

    Examples:
        - A quantile scheme with four classes:
            ```python
            >>> from cleopatra.styling.params import Classify
            >>> Classify(scheme="quantiles", k=4).to_options()
            {'scheme': 'quantiles', 'k': 4}

            ```
    """

    scheme: str | Sequence[float] | None = None
    k: int | None = None
    category_legend_kwargs: dict[str, Any] | None = None

    def to_options(self) -> dict[str, Any]:
        """Flatten the explicitly-set fields into `default_options` keys.

        Returns:
            dict: `scheme` / `k` / `category_legend_kwargs` for the fields
                that were set.
        """
        options: dict[str, Any] = {}
        if self.scheme is not None:
            options["scheme"] = self.scheme
        if self.k is not None:
            options["k"] = self.k
        if self.category_legend_kwargs is not None:
            options["category_legend_kwargs"] = self.category_legend_kwargs
        return options
