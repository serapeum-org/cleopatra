from collections import OrderedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.legend import Legend
from matplotlib.patches import Patch

from cleopatra.styles import ColorScale, Styles, disjoint_legend


def test_create_instance():
    assert isinstance(Styles.marker_style_list, list)
    assert isinstance(Styles.line_styles, OrderedDict)


class TestColorScale:
    """Tests for the `cleopatra.styles.ColorScale` `StrEnum`."""

    def test_members_are_strings(self):
        """Each member equals (and stringifies to) its lowercase value."""
        assert ColorScale.LINEAR == "linear"
        assert ColorScale.BOUNDARY_NORM == "boundary-norm"
        assert str(ColorScale.POWER) == "power"
        assert isinstance(ColorScale.MIDPOINT, str)

    def test_member_names_and_values(self):
        """The enum covers exactly the five supported scales."""
        assert {m.value for m in ColorScale} == {
            "linear", "power", "sym-lognorm", "boundary-norm", "midpoint"
        }

    @pytest.mark.parametrize(
        "given, expected",
        [
            ("linear", ColorScale.LINEAR),
            ("Linear", ColorScale.LINEAR),
            ("POWER", ColorScale.POWER),
            ("Sym-LogNorm", ColorScale.SYM_LOGNORM),
            ("BOUNDARY-norm", ColorScale.BOUNDARY_NORM),
            (ColorScale.MIDPOINT, ColorScale.MIDPOINT),
        ],
    )
    def test_construction_is_case_insensitive(self, given, expected):
        """`ColorScale(...)` accepts any case and existing members.

        Args:
            given: Input passed to `ColorScale(...)`.
            expected: The member it should resolve to.
        """
        assert ColorScale(given) is expected

    @pytest.mark.parametrize("bad", ["rainbow", "", "lin ear", 1, 2.0, None])
    def test_invalid_inputs_raise_valueerror(self, bad):
        """Anything that isn't a recognised value raises `ValueError`.

        Args:
            bad: An input that should not resolve to a member.
        """
        with pytest.raises(ValueError):
            ColorScale(bad)


class TestDisjointLegend:
    """Tests for cleopatra.styles.disjoint_legend (T0.4)."""

    @pytest.fixture()
    def ax(self):
        """A fresh axes, closed after the test to bound figure count."""
        fig, ax = plt.subplots()
        yield ax
        plt.close(fig)

    def test_returns_legend_added_to_axes(self, ax):
        """The helper returns a Legend that is attached to the axes.

        Test scenario:
            A two-class legend is built and the returned object is the
            same one registered on the axes.
        """
        legend = disjoint_legend(ax, ["red", "blue"], ["hot", "cold"])
        assert isinstance(legend, Legend), f"Expected a Legend, got {type(legend)}"
        assert ax.get_legend() is legend, "Legend should be attached to the axes"

    def test_one_swatch_patch_per_category(self, ax):
        """One Patch handle is created per category, in order.

        Test scenario:
            Three categories -> three Patch handles whose face colors
            match the requested colors.
        """
        colors = ["#1b9e77", "#d95f02", "#7570b3"]
        legend = disjoint_legend(ax, colors, ["water", "urban", "forest"])
        handles = legend.legend_handles
        assert len(handles) == 3, f"Expected 3 handles, got {len(handles)}"
        assert all(isinstance(h, Patch) for h in handles), "Handles must be Patches"
        expected = [matplotlib.colors.to_rgba(c) for c in colors]
        actual = [h.get_facecolor() for h in handles]
        assert actual == expected, f"Face colors mismatch: {actual} != {expected}"

    def test_labels_preserved_in_order(self, ax):
        """Legend texts match the labels in the given order.

        Test scenario:
            The drawn legend texts equal the input labels.
        """
        labels = ["a", "b", "c"]
        legend = disjoint_legend(ax, ["r", "g", "b"], labels)
        assert [t.get_text() for t in legend.get_texts()] == labels, (
            "Legend texts should equal the input labels in order"
        )

    def test_default_edgecolor_is_none(self, ax):
        """Swatches have no border by default (edgecolor='none').

        Test scenario:
            The first handle's edge color is fully transparent.
        """
        legend = disjoint_legend(ax, ["red"], ["x"])
        edge = legend.legend_handles[0].get_edgecolor()
        assert matplotlib.colors.to_rgba(edge)[3] == 0.0, (
            f"Default edge should be transparent, got {edge}"
        )

    def test_custom_edgecolor_applied(self, ax):
        """An explicit edgecolor is applied to every swatch.

        Test scenario:
            edgecolor='black' -> handle edge resolves to black.
        """
        legend = disjoint_legend(ax, ["red"], ["x"], edgecolor="black")
        edge = legend.legend_handles[0].get_edgecolor()
        assert matplotlib.colors.to_rgba(edge) == matplotlib.colors.to_rgba("black"), (
            f"Edge color should be black, got {edge}"
        )

    def test_legend_kwargs_forwarded(self, ax):
        """Extra kwargs are forwarded to Axes.legend (e.g. title).

        Test scenario:
            title='Class' surfaces on the legend's title text.
        """
        legend = disjoint_legend(ax, ["red", "blue"], ["hot", "cold"], title="Class")
        assert legend.get_title().get_text() == "Class", (
            "title kwarg should be forwarded to Axes.legend"
        )

    @pytest.mark.parametrize(
        "colors, labels",
        [
            (["red", "blue"], ["only-one"]),
            (["red"], ["a", "b"]),
            ([], ["a"]),
        ],
    )
    def test_length_mismatch_raises(self, ax, colors, labels):
        """Mismatched colors/labels lengths raise ValueError.

        Args:
            colors: Color sequence of one length.
            labels: Label sequence of a different length.

        Test scenario:
            Any length mismatch is rejected with a descriptive message.
        """
        with pytest.raises(ValueError, match="same length") as exc:
            disjoint_legend(ax, colors, labels)
        assert "same length" in str(exc.value), f"Unexpected message: {exc.value}"


class TestDiscreteContourfAcceptance:
    """T0.4 acceptance: explicit `levels` yields a discrete contourf + cbar."""

    def test_contourf_with_explicit_levels_is_discrete(self):
        """ArrayGlyph contourf with `levels` builds a discrete colorbar.

        Test scenario:
            Plotting with kind='contourf' and explicit edges produces a
            colorbar whose ticks are exactly those edges, confirming the
            already-implemented discrete-levels path still holds.
        """
        from cleopatra.array_glyph import ArrayGlyph

        data = np.linspace(0, 10, 36).reshape(6, 6)
        edges = [0, 2, 4, 6, 8, 10]
        glyph = ArrayGlyph(data, levels=edges)
        glyph.plot(kind="contourf", cmap="viridis")
        ticks = list(glyph.cbar.get_ticks())
        assert ticks == [float(e) for e in edges], (
            f"Colorbar ticks should equal the level edges, got {ticks}"
        )
        plt.close("all")
