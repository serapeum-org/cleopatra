from collections import OrderedDict

import pytest

from cleopatra.styles import ColorScale, Styles


def test_create_instance():
    assert isinstance(Styles.marker_style_list, list)
    assert isinstance(Styles.line_styles, OrderedDict)


class TestColorScale:
    """Tests for the :class:`cleopatra.styles.ColorScale` ``StrEnum``."""

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
        """``ColorScale(...)`` accepts any case and existing members.

        Args:
            given: Input passed to ``ColorScale(...)``.
            expected: The member it should resolve to.
        """
        assert ColorScale(given) is expected

    @pytest.mark.parametrize("bad", ["rainbow", "", "lin ear", 1, 2.0, None])
    def test_invalid_inputs_raise_valueerror(self, bad):
        """Anything that isn't a recognised value raises ``ValueError``.

        Args:
            bad: An input that should not resolve to a member.
        """
        with pytest.raises(ValueError):
            ColorScale(bad)
