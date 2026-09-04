from collections import OrderedDict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.container import BarContainer
from matplotlib.legend import Legend
from matplotlib.patches import Patch

from cleopatra.styling.params import Contour
from cleopatra.styling.styles import (
    ColorScale,
    Scale,
    Styles,
    _categorical_is_null,
    apply_blank_canvas,
    colorbar_legend,
    disjoint_legend,
    histogram_legend,
    swatch_legend,
)


def test_create_instance():
    assert isinstance(Styles.marker_style_list, list)
    assert isinstance(Styles.line_styles, OrderedDict)


class TestColorScale:
    """Tests for the `cleopatra.styling.styles.ColorScale` `StrEnum`."""

    def test_members_are_strings(self):
        """Each member equals (and stringifies to) its lowercase value."""
        assert ColorScale.LINEAR == "linear"
        assert ColorScale.BOUNDARY_NORM == "boundary-norm"
        assert str(ColorScale.POWER) == "power"
        assert isinstance(ColorScale.MIDPOINT, str)

    def test_member_names_and_values(self):
        """The enum covers exactly the five supported scales."""
        assert {m.value for m in ColorScale} == {
            "linear",
            "power",
            "sym-lognorm",
            "boundary-norm",
            "midpoint",
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
    """Tests for cleopatra.styling.styles.disjoint_legend (T0.4)."""

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
        from cleopatra.glyphs.gridded.array_glyph import ArrayGlyph

        data = np.linspace(0, 10, 36).reshape(6, 6)
        edges = [0, 2, 4, 6, 8, 10]
        glyph = ArrayGlyph(data)
        glyph.plot(kind="contourf", cmap="viridis", contour=Contour(levels=edges))
        ticks = list(glyph.cbar.get_ticks())
        assert ticks == [float(e) for e in edges], (
            f"Colorbar ticks should equal the level edges, got {ticks}"
        )
        plt.close("all")


class TestColorbarLegend:
    """Tests for cleopatra.styling.styles.colorbar_legend (T5.3c)."""

    @pytest.fixture()
    def scatter(self):
        """A coloured scatter mappable on its own axes.

        Returns:
            tuple: (axes, PathCollection) with a value array attached.
        """
        fig, ax = plt.subplots()
        sc = ax.scatter([0, 1, 2], [0, 1, 0], c=[10.0, 20.0, 30.0])
        yield ax, sc
        plt.close(fig)

    def test_returns_colorbar(self, scatter):
        """A Colorbar is created for the mappable.

        Test scenario:
            The helper returns a matplotlib Colorbar instance.
        """
        ax, sc = scatter
        cbar = colorbar_legend(sc, ax)
        assert isinstance(cbar, Colorbar), f"Expected Colorbar, got {type(cbar)}"

    def test_label_forwarded(self, scatter):
        """The label kwarg is forwarded to Figure.colorbar.

        Test scenario:
            label='depth' surfaces on the colorbar axis.
        """
        ax, sc = scatter
        cbar = colorbar_legend(sc, ax, label="depth")
        assert cbar.ax.get_ylabel() == "depth", (
            f"Unexpected colorbar label: {cbar.ax.get_ylabel()}"
        )

    def test_infers_axes_from_mappable(self, scatter):
        """With no ax, the mappable's own axes is used.

        Test scenario:
            Omitting ax still produces a colorbar (axes inferred).
        """
        _, sc = scatter
        cbar = colorbar_legend(sc)
        assert isinstance(cbar, Colorbar), "Should infer axes from the mappable"

    def test_no_axes_raises(self):
        """A detached mappable with no ax raises ValueError.

        Test scenario:
            A ScalarMappable not attached to any axes and ax=None fails.
        """
        from matplotlib.cm import ScalarMappable

        sm = ScalarMappable()
        sm.set_array([0.0, 1.0])
        with pytest.raises(ValueError, match="determine an axes"):
            colorbar_legend(sm)


class TestHistogramLegend:
    """Tests for cleopatra.styling.styles.histogram_legend (T5.3c)."""

    @pytest.fixture()
    def ax(self):
        """A fresh axes closed after the test."""
        fig, ax = plt.subplots()
        yield ax
        plt.close(fig)

    def test_returns_one_bar_per_bin(self, ax):
        """A BarContainer with one bar per bin is returned.

        Test scenario:
            bins=3 -> three bars from the explicit values.
        """
        bars = histogram_legend(
            ax, [0.0, 1.0, 1.0, 2.0, 2.0, 2.0], cmap="viridis", bins=3
        )
        assert isinstance(bars, BarContainer), (
            f"Expected BarContainer, got {type(bars)}"
        )
        assert len(bars) == 3, f"Expected 3 bars, got {len(bars)}"

    def test_bars_coloured_by_cmap(self, ax):
        """Bars take distinct colours across the colormap.

        Test scenario:
            The first and last bar differ in colour (cmap applied).
        """
        bars = histogram_legend(ax, np.linspace(0, 10, 100), cmap="viridis", bins=5)
        first = bars.patches[0].get_facecolor()
        last = bars.patches[-1].get_facecolor()
        assert first != last, "First and last bars should differ in colour"

    def test_inherits_cmap_norm_and_array_from_mappable(self, ax):
        """cmap/norm/data are taken from a mappable when values is None.

        Test scenario:
            A scatter mappable's array drives the histogram (4 points,
            4 bins -> 4 bars) with no explicit values.
        """
        fig2, main_ax = plt.subplots()
        sc = main_ax.scatter(
            [0, 1, 2, 3], [0, 1, 0, 1], c=[1.0, 2.0, 3.0, 4.0], cmap="plasma"
        )
        bars = histogram_legend(ax, mappable=sc, bins=4)
        assert len(bars) == 4, (
            f"Expected 4 bars from the mappable array, got {len(bars)}"
        )
        plt.close(fig2)

    def test_horizontal_orientation(self, ax):
        """Horizontal orientation draws barh bars.

        Test scenario:
            orientation='horizontal' still yields one bar per bin.
        """
        bars = histogram_legend(
            ax, [0.0, 1.0, 2.0, 3.0], cmap="viridis", bins=2, orientation="horizontal"
        )
        assert len(bars) == 2, f"Expected 2 horizontal bars, got {len(bars)}"

    def test_non_finite_values_dropped(self, ax):
        """NaN/inf entries are filtered before histogramming.

        Test scenario:
            A values array with NaNs still histograms the finite part.
        """
        bars = histogram_legend(
            ax, [0.0, np.nan, 1.0, np.inf, 2.0], cmap="viridis", bins=2
        )
        assert len(bars) == 2, "Non-finite values should be dropped, not crash"

    def test_no_values_and_no_mappable_raises(self, ax):
        """Calling with neither values nor a mappable raises ValueError.

        Test scenario:
            Missing data source is rejected.
        """
        with pytest.raises(ValueError, match="Provide `values`"):
            histogram_legend(ax)

    def test_all_non_finite_raises(self, ax):
        """All-non-finite values raise ValueError.

        Test scenario:
            No finite data to histogram is rejected.
        """
        with pytest.raises(ValueError, match="No finite values"):
            histogram_legend(ax, [np.nan, np.inf], cmap="viridis")

    def test_invalid_orientation_raises(self, ax):
        """An invalid orientation raises ValueError.

        Test scenario:
            orientation must be vertical or horizontal.
        """
        with pytest.raises(ValueError, match="orientation must be"):
            histogram_legend(ax, [0.0, 1.0], cmap="viridis", orientation="diagonal")

    def test_does_not_mutate_unscaled_mappable_norm(self, ax):
        """Building from a mappable does not autoscale its norm (L3 fix).

        Test scenario:
            A ScalarMappable whose norm has vmin/vmax unset supplies the
            cmap/norm while values are passed explicitly. histogram_legend
            must copy the norm so mapping the bin centres does not
            autoscale the caller's norm in place (vmin/vmax stay None).
        """
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        sm = ScalarMappable(norm=Normalize(), cmap="viridis")
        assert sm.norm.vmin is None and sm.norm.vmax is None, "precondition: unscaled"

        histogram_legend(ax, values=[0.0, 1.0, 2.0, 3.0], mappable=sm, bins=3)

        assert sm.norm.vmin is None and sm.norm.vmax is None, (
            f"mappable norm should remain unscaled, got "
            f"vmin={sm.norm.vmin}, vmax={sm.norm.vmax}"
        )

    def test_preserves_boundary_norm_subtype_from_mappable(self, ax):
        """A BoundaryNorm on the mappable is preserved (copy keeps subtype).

        Test scenario:
            A scatter coloured with a BoundaryNorm drives the legend; the
            histogram still renders one bar per bin without raising.
        """
        from matplotlib.colors import BoundaryNorm

        fig2, main_ax = plt.subplots()
        norm = BoundaryNorm([0, 1, 2, 3], ncolors=256)
        sc = main_ax.scatter(
            [0, 1, 2, 3],
            [0, 1, 0, 1],
            c=[0.5, 1.5, 2.5, 0.5],
            cmap="viridis",
            norm=norm,
        )
        bars = histogram_legend(ax, mappable=sc, bins=3)
        assert len(bars) == 3, f"Expected 3 bars, got {len(bars)}"
        assert norm.vmin == 0 and norm.vmax == 3, "Original BoundaryNorm untouched"
        plt.close(fig2)


class TestSwatchLegend:
    """Tests for `swatch_legend`."""

    @pytest.fixture
    def ax(self):
        """A fresh Axes on the Agg backend, closed after the test."""
        fig, ax = plt.subplots()
        yield ax
        plt.close(fig)

    def test_returns_inset_axes_child_of_ax(self, ax):
        """The swatch is an `Axes` registered as a child of the given axes."""
        swatch = swatch_legend(ax, "viridis", "Dust")
        assert isinstance(swatch, Axes), f"expected Axes, got {type(swatch)}"
        assert swatch in ax.child_axes, "swatch should be a child of ax"

    def test_label_and_default_endpoints(self, ax):
        """The label and default (0/≥1) endpoint text are drawn on the swatch."""
        swatch = swatch_legend(ax, "viridis", "Organic Matter")
        texts = [t.get_text() for t in swatch.texts]
        assert texts == ["Organic Matter", "0", "≥1"], f"unexpected texts: {texts}"

    def test_custom_vmin_vmax_formatting(self, ax):
        """Custom `vmin`/`vmax` are formatted without a trailing '.0'."""
        swatch = swatch_legend(ax, "plasma", "Elevation", vmin=0, vmax=5000)
        texts = [t.get_text() for t in swatch.texts]
        assert texts[1:] == ["0", "≥5000"], f"unexpected endpoint labels: {texts[1:]}"

    def test_vmax_prefix_can_be_disabled(self, ax):
        """An empty `vmax_prefix` yields a plain endpoint value, no '≥'."""
        swatch = swatch_legend(ax, "viridis", "Depth", vmax=100, vmax_prefix="")
        texts = [t.get_text() for t in swatch.texts]
        assert texts[-1] == "100", f"expected plain '100', got {texts[-1]!r}"

    def test_vmin_prefix_marks_a_capped_low_endpoint(self, ax):
        """A `vmin_prefix` (e.g. '≤') is drawn before the low endpoint value."""
        swatch = swatch_legend(ax, "viridis", "T", vmin=-40, vmax=40, vmin_prefix="≤")
        texts = [t.get_text() for t in swatch.texts]
        assert texts[1] == "≤-40", f"expected '≤-40' low endpoint, got {texts[1]!r}"

    def test_accepts_colormap_object(self, ax):
        """A `Colormap` instance (not just a name string) is accepted directly."""
        from matplotlib.colors import LinearSegmentedColormap

        cmap = LinearSegmentedColormap.from_list("test", ["white", "black"])
        swatch = swatch_legend(ax, cmap, "Custom")
        assert swatch.images, "gradient image should be drawn on the swatch"
        assert swatch.images[0].get_cmap() is cmap, "custom cmap should be used as-is"

    def test_label_and_endpoints_get_a_legibility_outline(self, ax):
        """Every swatch text carries a `path_effects` halo so a white label stays
        readable over the pale middle of a colormap (not just white-on-white)."""
        swatch = swatch_legend(ax, "Spectral_r", "Temperature")
        for text in swatch.texts:
            assert text.get_path_effects(), f"{text.get_text()!r} should have a legibility outline"

    def test_dark_label_gets_a_light_outline(self, ax):
        """A dark `text_color` is haloed in light (contrast flips with luminance)."""
        swatch = swatch_legend(ax, "viridis", "T", text_color="black")
        effects = swatch.texts[0].get_path_effects()
        assert effects, "a dark label should still get an outline"

    def test_boundary_norm_swatch_matches_the_norm_not_a_raw_ramp(self, ax):
        """A BoundaryNorm swatch paints exactly cmap(norm(midpoints)), honouring extend (not a 0..1 ramp)."""
        from matplotlib.colors import BoundaryNorm

        cmap = plt.get_cmap("viridis")
        norm = BoundaryNorm(
            [0, 1, 2, 3, 4], ncolors=cmap.N, extend="both"
        )  # 4 in-range bands
        swatch = swatch_legend(ax, cmap, "T", vmin=0, vmax=4, norm=norm)
        bands = np.asarray(swatch.images[0].get_array())[
            0
        ]  # (4, 4) RGBA, one row per band
        mids = np.array([0.5, 1.5, 2.5, 3.5])
        assert bands.shape[0] == 4, "swatch should show 4 discrete bands"
        assert np.allclose(bands, np.asarray(cmap(norm(mids)))), (
            "bands must match the map's colours"
        )
        # `extend` reserves the colormap ends for under/over, so band 0 is not cmap(0.0).
        assert not np.allclose(bands[0], cmap(0.0)), (
            "extend must shift in-range bands off the ends"
        )

    def test_no_axes_ticks_or_spines(self, ax):
        """The swatch is chrome-free: no ticks and no visible spines."""
        swatch = swatch_legend(ax, "viridis", "Dust")
        assert swatch.get_xticks().size == 0, "swatch should have no x ticks"
        assert swatch.get_yticks().size == 0, "swatch should have no y ticks"
        assert not any(s.get_visible() for s in swatch.spines.values()), (
            "swatch spines should all be hidden"
        )

    def test_independent_of_projection_or_other_styling(self, ax):
        """`swatch_legend` composes with unrelated data already drawn on `ax`.

        Test scenario:
            A swatch attached to an axes that already holds a plain imshow
            plot (representing "some other style") does not disturb it.
        """
        img = ax.imshow([[0, 1], [1, 0]], cmap="gray")
        swatch = swatch_legend(ax, "viridis", "Dust")
        assert img in ax.images, "pre-existing plot should be untouched"
        assert swatch in ax.child_axes, "swatch should still attach cleanly"


class TestApplyBlankCanvas:
    """Tests for `apply_blank_canvas`."""

    @pytest.fixture
    def ax(self):
        """A fresh Axes on the Agg backend, closed after the test."""
        fig, ax = plt.subplots()
        yield ax
        plt.close(fig)

    def test_removes_ticks(self, ax):
        """Both x and y ticks are removed."""
        apply_blank_canvas(ax)
        assert ax.get_xticks().size == 0, "x ticks should be removed"
        assert ax.get_yticks().size == 0, "y ticks should be removed"

    def test_hides_all_spines(self, ax):
        """Every spine (frame edge) is hidden."""
        apply_blank_canvas(ax)
        assert not any(s.get_visible() for s in ax.spines.values()), (
            "all spines should be hidden"
        )

    def test_default_facecolor_is_black(self, ax):
        """The default `facecolor` ('black') is applied to axes and figure."""
        apply_blank_canvas(ax)
        assert ax.get_facecolor() == (
            0.0,
            0.0,
            0.0,
            1.0,
        ), f"unexpected axes facecolor: {ax.get_facecolor()}"
        assert ax.figure.get_facecolor() == (
            0.0,
            0.0,
            0.0,
            1.0,
        ), f"unexpected figure facecolor: {ax.figure.get_facecolor()}"

    def test_custom_facecolor_applied_to_axes_and_figure(self, ax):
        """A custom `facecolor` is applied consistently to both axes and figure."""
        apply_blank_canvas(ax, facecolor="white")
        assert ax.get_facecolor() == (1.0, 1.0, 1.0, 1.0)
        assert ax.figure.get_facecolor() == (1.0, 1.0, 1.0, 1.0)

    def test_returns_the_same_axes(self, ax):
        """The function returns `ax` itself, enabling call chaining."""
        result = apply_blank_canvas(ax)
        assert result is ax, "should return the same Axes instance"

    def test_composes_with_existing_plot(self, ax):
        """Stripping chrome does not remove or disturb already-drawn artists."""
        img = ax.imshow([[0, 1], [1, 0]], cmap="gray")
        apply_blank_canvas(ax)
        assert img in ax.images, "pre-existing plot should be untouched"


class TestGetLineStyleUnknownName:
    """`Styles.get_line_style` with a name that is not registered."""

    def test_unknown_name_returns_none_and_lists_the_options(self, capsys):
        """An unknown style name prints the available names and returns `None`.

        Test scenario:
            The lookup is a convenience helper, so a typo reports the valid
            choices on stdout rather than raising -- the caller gets `None`
            and matplotlib's own default.
        """
        assert Styles.get_line_style("not-a-line-style") is None

        printed = capsys.readouterr().out
        assert "not-a-line-style" in printed, printed
        assert "solid" in printed, f"the available styles should be listed: {printed}"

    def test_known_name_still_resolves(self):
        """A registered name resolves to its matplotlib dash spec."""
        assert Styles.get_line_style("solid") == Styles.line_styles["solid"]


class TestScaleConstruction:
    """`Scale` is usable as an object even though its API is static."""

    def test_instance_dispatches_the_static_helpers(self):
        """An instance reaches the same scaling as the class does.

        Test scenario:
            `Scale()` exists only so the documented "not typically necessary"
            construction keeps working. `isinstance(Scale(), Scale)` and
            `callable(...)` are guaranteed by the language and could not fail,
            so the assertion is on the scaling an instance actually computes.
        """
        assert Scale().log_scale(100.0) == pytest.approx(2.0)
        assert Scale().power_scale(-10.0)(100.0) == pytest.approx(
            Scale.power_scale(-10.0)(100.0)
        ), "an instance must build the same scaling function as the class"


class TestSwatchLegendBackingPanel:
    """`swatch_legend(box=...)` draws an opaque panel behind the swatch."""

    @staticmethod
    def _added_panel(box):
        """Draw a swatch legend with `box` and return the one patch it added.

        Args:
            box: The `box=` argument under test (`True`, a colour, or a dict).

        Returns:
            tuple: `(figure, panel)` -- the figure to close and the single
                patch the `box` argument put on the axes.
        """
        fig, ax = plt.subplots()
        before = set(map(id, ax.patches))
        swatch_legend(ax, "viridis", "label", vmin=0.0, vmax=1.0, box=box)
        added = [p for p in ax.patches if id(p) not in before]
        assert len(added) == 1, f"box= should add exactly one panel; got {len(added)}"
        return fig, added[0]

    def test_string_box_sets_the_panel_colour(self):
        """A colour string is used as the panel's face colour.

        Test scenario:
            The panel is identified as the patch the call *added*, not by
            searching the axes for one of the right colour -- otherwise any
            unrelated black artist would satisfy the assertion.
        """
        fig, panel = self._added_panel("black")

        assert panel.get_facecolor()[:3] == (0.0, 0.0, 0.0)
        plt.close(fig)

    def test_dict_box_overrides_the_panel_defaults(self):
        """A dict is merged over the default panel style.

        Test scenario:
            `facecolor` replaces the default white and `linewidth` replaces
            the default 0.6, proving the dict is merged rather than used
            wholesale or ignored.
        """
        fig, panel = self._added_panel({"facecolor": "red", "linewidth": 2.5})

        assert panel.get_linewidth() == 2.5, "the dict's linewidth should be applied"
        assert panel.get_facecolor()[:3] == (1.0, 0.0, 0.0), (
            "the dict's facecolor should override the default white"
        )
        assert panel.get_edgecolor()[:3] == pytest.approx((0.6, 0.6, 0.6)), (
            "an unspecified key should keep its default, proving a merge"
        )
        plt.close(fig)


class TestCategoricalIsNullGuards:
    """`_categorical_is_null` never lets a null-check raise through.

    Both fallbacks are guarded because a container-like element makes the
    null test return an array, whose truth value is ambiguous.
    """

    def test_pandas_check_failure_reads_as_not_null(self):
        """A `pd.isna` that raises leaves the value classified as real.

        Test scenario:
            `pd.isna` returns an array for an array-like element, so
            `bool(...)` raises `ValueError`; such a value is simply not
            null and is carried through.
        """

        def _raising_isna(_value):
            raise ValueError("truth value of an array is ambiguous")

        assert _categorical_is_null(object(), _raising_isna) is False

    def test_numpy_fallback_failure_reads_as_not_null(self):
        """Without pandas, an unrecognised scalar is a real, colourable value.

        Test scenario:
            `pd_isna=None` (pandas absent) sends the value to `np.isnat`,
            which rejects a non-datetime object; the guard turns that into
            "not null" rather than an error.
        """
        assert _categorical_is_null(object(), None) is False

    @pytest.mark.parametrize(
        "value", [np.datetime64("NaT"), np.timedelta64("NaT"), float("nan"), None]
    )
    def test_real_nulls_are_still_recognised_without_pandas(self, value):
        """Every null flavour is dropped even when pandas is unavailable.

        Test scenario:
            The two tests above cover only the `except -> False` arms, which a
            function answering `False` for everything would satisfy just as
            well. These are the positive cases that rule that out.

            Which numpy branch answers is deliberately not asserted: on numpy
            2.4 `np.isnan` accepts datetime64/timedelta64 and returns `True`
            for `NaT`, so the `np.isnat` fallback below it is unreachable
            here -- it exists for builds whose `np.isnan` rejects those
            dtypes. The contract (a `NaT` never becomes a category) is what
            callers depend on and what holds across both.
        """
        assert _categorical_is_null(value, None) is True
