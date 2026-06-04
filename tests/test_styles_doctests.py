"""Doctest runner for `cleopatra.styles`.

Pytest is not configured with ``--doctest-modules``, so the docstring examples
in ``src/cleopatra/styles.py`` (ColorScale, Scale, MidpointNormalize, classify,
resolve_sizes, the legend helpers, ...) would otherwise never run. This module
executes them in-band so example drift fails CI — mirroring the existing
``test_projection`` / ``test_statistical_glyph`` doctest runners.
"""

import doctest

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import pytest

import cleopatra.styles as styles_module


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test to bound memory."""
    yield
    plt.close("all")


def test_module_doctests_execute():
    """Run every `cleopatra.styles` docstring example in-band.

    Test scenario:
        All collected doctest examples in the module execute with zero
        failures, and at least one example is collected (so the coverage is
        not silently dropped if examples are moved or removed).

    Note:
        ``ELLIPSIS`` is enabled to match pytest's ``--doctest-modules``
        behavior, since some examples use ``...`` in expected tracebacks.
    """
    try:
        results = doctest.testmod(
            styles_module, verbose=False, optionflags=doctest.ELLIPSIS
        )
    finally:
        plt.close("all")
    assert results.failed == 0, f"{results.failed} doctest example(s) failed in styles"
    assert results.attempted > 0, (
        "no doctest examples were collected from styles; the module's docstring "
        "examples may have been moved or removed, silently dropping this coverage"
    )
