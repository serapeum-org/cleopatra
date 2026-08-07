"""Doctest runner for `cleopatra.config`.

Pytest is not configured with ``--doctest-modules``, so the docstring examples
in ``src/cleopatra/config.py`` (notably ``Config.get_cache_dir``) would
otherwise never run. This module executes them in-band so example drift fails
CI — mirroring the existing ``test_styles_doctests`` / ``test_projection``
doctest runners.
"""

import doctest

import cleopatra.config as config_module


def test_module_doctests_execute():
    """Run every `cleopatra.config` docstring example in-band.

    Test scenario:
        All collected doctest examples in the module execute with zero
        failures, and at least one example is collected (so the coverage is
        not silently dropped if examples are moved or removed).

    Note:
        ``ELLIPSIS`` is enabled to match pytest's ``--doctest-modules``
        behavior for any examples that use ``...``.
    """
    results = doctest.testmod(
        config_module, verbose=False, optionflags=doctest.ELLIPSIS
    )
    assert results.failed == 0, f"{results.failed} doctest example(s) failed in config"
    assert results.attempted > 0, (
        "no doctest examples were collected from config; the module's docstring "
        "examples may have been moved or removed, silently dropping this coverage"
    )
