"""Shared helpers for the categorical glyphs (`BarGlyph`, `PieGlyph`).

Both glyphs take a `(categories, values)` pair and colour one artist per
category from the shared palette system, so the validation and the
category -> colour resolution live here rather than being duplicated in each
glyph module.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from cleopatra.styling.colors import resolve_colormap, validate_color_sequence
from cleopatra.styling.palettes import get_palette


def resolve_categories(
    categories: Sequence, values: Any
) -> tuple[list[str], np.ndarray]:
    """Validate a `(categories, values)` pair and normalise it.

    Args:
        categories: Sequence of category labels; each is coerced to `str`.
        values: 1-D sequence/array of numeric values, one per category.

    Returns:
        tuple[list[str], np.ndarray]: The string labels and a 1-D float array
            of values, in the given order.

    Raises:
        ValueError: If `values` is not 1-D, if the two lengths differ, if
            either is empty, or if any value is non-finite (NaN or inf).
        TypeError: If `values` cannot be coerced to a float array (a
            non-numeric / object input), surfaced by `numpy`.

    Examples:
        - A simple pair is normalised to labels and a float array:
            ```python
            >>> from cleopatra.glyphs.stats._categorical import resolve_categories
            >>> labels, values = resolve_categories(["a", "b"], [1, 2])
            >>> labels
            ['a', 'b']
            >>> values.tolist()
            [1.0, 2.0]

            ```
    """
    labels = [str(c) for c in categories]
    vals = np.asarray(values, dtype=float)
    if vals.ndim != 1:
        raise ValueError(f"values must be 1-D, got {vals.ndim}-D.")
    if len(labels) != vals.size:
        raise ValueError(
            f"categories ({len(labels)}) and values ({vals.size}) must have the "
            "same length."
        )
    if vals.size == 0:
        raise ValueError("categories and values must be non-empty.")
    # Reject NaN / inf up front: a non-finite bar is drawn silently and a
    # non-finite pie wedge only fails later inside matplotlib with a cryptic
    # message, so catch it here with a clear one.
    if not np.isfinite(vals).all():
        raise ValueError("values must be finite (no NaN or inf).")
    return labels, vals


def resolve_series(categories: Sequence, values: Any) -> tuple[list[str], np.ndarray]:
    """Validate a multi-series `(categories, values)` pair and normalise it.

    The 2-D counterpart to `resolve_categories`, for a stacked / grouped bar
    chart: one row per category, one column per series.

    Args:
        categories: Sequence of category labels; each is coerced to `str`.
        values: 2-D sequence/array of shape `(n_categories, n_series)` -- one
            row per category, one column per series.

    Returns:
        tuple[list[str], np.ndarray]: The string labels and a 2-D float array
            of shape `(n_categories, n_series)`.

    Raises:
        ValueError: If `values` is not 2-D, if its row count differs from the
            number of categories, if it is empty, or if any value is non-finite
            (NaN or inf).
        TypeError: If `values` cannot be coerced to a float array, surfaced by
            `numpy`.

    Examples:
        - A `(2 categories, 2 series)` block is normalised to labels and a 2-D
          float array:
            ```python
            >>> from cleopatra.glyphs.stats._categorical import resolve_series
            >>> labels, values = resolve_series(["a", "b"], [[1, 2], [3, 4]])
            >>> labels
            ['a', 'b']
            >>> values.shape
            (2, 2)

            ```
    """
    labels = [str(c) for c in categories]
    vals = np.asarray(values, dtype=float)
    if vals.ndim != 2:
        raise ValueError(
            f"multi-series values must be 2-D (n_categories, n_series), got "
            f"{vals.ndim}-D."
        )
    if vals.shape[0] != len(labels):
        raise ValueError(
            f"categories ({len(labels)}) must match the number of value rows "
            f"({vals.shape[0]})."
        )
    if vals.size == 0:
        raise ValueError("categories and values must be non-empty.")
    if not np.isfinite(vals).all():
        raise ValueError("values must be finite (no NaN or inf).")
    return labels, vals


def resolve_series_colors(n_series: int, colors: Sequence | None, cmap: Any) -> list:
    """Return one colour per series for a multi-series bar chart.

    Unlike `resolve_category_colors` (one colour per *category*, from a
    qualitative palette), this returns one colour per *series*, sampled from a
    continuous colormap when no explicit colours are given -- matching the
    `colormap=`-style spread a `pandas` stacked/grouped bar chart uses.

    Args:
        n_series: Number of series (and colours) needed.
        colors: Explicit per-series colours, or `None` to sample from `cmap`.
            When given, its length must equal `n_series`.
        cmap: A colormap name or `Colormap` (resolved via
            `cleopatra.styling.colors.resolve_colormap`) sampled at evenly
            spaced positions, one per series.

    Returns:
        list: Exactly `n_series` colours, in series order.

    Raises:
        ValueError: If `colors` is given and its length is not `n_series`.

    Examples:
        - Explicit colours are returned verbatim:
            ```python
            >>> from cleopatra.glyphs.stats._categorical import resolve_series_colors
            >>> resolve_series_colors(2, ["#ff0000", "#00ff00"], "viridis")
            ['#ff0000', '#00ff00']

            ```
        - Without explicit colours, the colormap is sampled once per series:
            ```python
            >>> from cleopatra.glyphs.stats._categorical import resolve_series_colors
            >>> cols = resolve_series_colors(3, None, "viridis")
            >>> len(cols)
            3

            ```
    """
    if colors is not None:
        colors = list(colors)
        if len(colors) != n_series:
            raise ValueError(
                f"colors ({len(colors)}) must match the number of series ({n_series})."
            )
        return colors
    cmap_obj = resolve_colormap(cmap)
    positions = [0.5] if n_series == 1 else np.linspace(0.0, 1.0, n_series)
    return [cmap_obj(float(p)) for p in positions]


def resolve_category_colors(n: int, colors: Sequence | None, palette: str) -> list:
    """Return one colour per category.

    Args:
        n: Number of categories (and colours) needed.
        colors: Explicit per-category colours, or `None` to draw from the
            palette. When given, its length must equal `n`.
        palette: Name of a registered qualitative palette (see
            `cleopatra.styling.palettes`). Its swatches are cycled when there
            are more categories than swatches.

    Returns:
        list: Exactly `n` colours, in category order.

    Raises:
        TypeError: If `colors` is given as a bare `str`/`bytes` rather than a
            sequence of colours.
        ValueError: If `colors` is given and its length is not `n`, or if
            `palette` is not a registered palette name.

    Examples:
        - Palette swatches are cycled to the requested length:
            ```python
            >>> from cleopatra.glyphs.stats._categorical import resolve_category_colors
            >>> cols = resolve_category_colors(2, None, "category12")
            >>> len(cols)
            2

            ```
        - Explicit colours are returned verbatim:
            ```python
            >>> from cleopatra.glyphs.stats._categorical import resolve_category_colors
            >>> resolve_category_colors(2, ["#ff0000", "#00ff00"], "category12")
            ['#ff0000', '#00ff00']

            ```
    """
    if colors is not None:
        return validate_color_sequence(colors, n, param="colors", noun="categories")
    try:
        swatches = list(get_palette(palette).colors)
    except KeyError as exc:
        # get_palette raises KeyError for an unknown name; re-raise as ValueError
        # so callers see the same exception type as every other validation here.
        raise ValueError(f"unknown palette {palette!r}.") from exc
    return [swatches[i % len(swatches)] for i in range(n)]
