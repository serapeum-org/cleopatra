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

from cleopatra.styling.palettes import get_palette


def resolve_categories(categories: Sequence, values: Any) -> tuple[list[str], np.ndarray]:
    """Validate a `(categories, values)` pair and normalise it.

    Args:
        categories: Sequence of category labels; each is coerced to `str`.
        values: 1-D sequence/array of numeric values, one per category.

    Returns:
        tuple[list[str], np.ndarray]: The string labels and a 1-D float array
            of values, in the given order.

    Raises:
        ValueError: If `values` is not 1-D, if the two lengths differ, or if
            either is empty.

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
    return labels, vals


def resolve_category_colors(
    n: int, colors: Sequence | None, palette: str
) -> list:
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
        ValueError: If `colors` is given and its length is not `n`.

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
        colors = list(colors)
        if len(colors) != n:
            raise ValueError(
                f"colors ({len(colors)}) must match the number of categories ({n})."
            )
        return colors
    swatches = list(get_palette(palette).colors)
    return [swatches[i % len(swatches)] for i in range(n)]
