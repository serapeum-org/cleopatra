"""Tests for the one-call ``publication_map`` composer."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from cleopatra.styling.params import DataStyle
from cleopatra.templates import publication_map

pytestmark = pytest.mark.plot


@pytest.fixture(scope="function")
def field():
    """Provide a small temperature-like field.

    Returns:
        numpy.ndarray: A ``(10, 12)`` array of values in a plausible degC range.
    """
    return np.random.default_rng(0).random((10, 12)) * 30.0


class TestPublicationMap:
    """Tests for ``cleopatra.templates.publication_map``."""

    def test_styled_field_sets_title_and_draws(self, field):
        """A styled call sets the title and draws the field.

        Test scenario:
            ``style`` + ``title`` produce a titled axes carrying a mappable.
        """
        fig, ax = publication_map(field, style="temperature_2m", title="2 m temperature")
        assert ax.get_title() == "2 m temperature", f"unexpected title: {ax.get_title()!r}"
        assert ax.images or ax.collections, "the field should be drawn"
        plt.close(fig)

    def test_style_and_data_style_conflict_raises(self, field):
        """Passing both `style=` and `data_style=` raises, not an opaque TypeError.

        Test scenario:
            Both set the same grouped render option, so `publication_map`
            rejects the collision up front instead of crashing with
            `plot() got multiple values for keyword argument 'data_style'`.
        """
        with pytest.raises(ValueError, match="not both"):
            publication_map(
                field,
                style="temperature_2m",
                data_style=DataStyle(style="temperature_2m"),
            )

    def test_cmap_and_flat_projection_compose(self, field):
        """A `cmap` with a flat projection composes without a style.

        Test scenario:
            Passing 1-D coords + ``projection='flat'`` + a plain ``cmap`` renders.
        """
        lon = np.linspace(-10.0, 10.0, 12)
        lat = np.linspace(30.0, 50.0, 10)
        fig, ax = publication_map(field, coords=(lon, lat), cmap="viridis", projection="flat")
        assert ax.collections, "a projected render should add a QuadMesh collection"
        plt.close(fig)
