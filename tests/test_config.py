import inspect
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytest

from cleopatra.config import Config, is_notebook


def test_create_config():
    assert Config()


class TestGetCacheDir:
    """`Config.get_cache_dir` resolves the basemap-asset cache directory."""

    def test_default_when_unset(self, monkeypatch):
        """With no override, it is `~/.cleopatra/naturalearth`."""
        monkeypatch.delenv("CLEOPATRA_CACHE_DIR", raising=False)
        assert (
            Config.get_cache_dir() == Path.home() / ".cleopatra" / "naturalearth"
        ), "default should be ~/.cleopatra/naturalearth"

    def test_env_var_override(self, monkeypatch, tmp_path):
        """`CLEOPATRA_CACHE_DIR` overrides the default."""
        monkeypatch.setenv("CLEOPATRA_CACHE_DIR", str(tmp_path / "cache"))
        assert Config.get_cache_dir() == tmp_path / "cache"

    def test_explicit_arg_overrides_env(self, monkeypatch, tmp_path):
        """An explicit `path` argument wins over the environment variable."""
        monkeypatch.setenv("CLEOPATRA_CACHE_DIR", str(tmp_path / "from_env"))
        assert (
            Config.get_cache_dir(tmp_path / "explicit") == tmp_path / "explicit"
        ), "explicit path should take precedence over the env var"

    def test_tilde_is_expanded(self, monkeypatch):
        """A leading `~` in the env var is expanded to the home directory."""
        monkeypatch.setenv("CLEOPATRA_CACHE_DIR", "~/foo/bar")
        assert Config.get_cache_dir() == Path.home() / "foo" / "bar"

    def test_empty_env_falls_back_to_default(self, monkeypatch):
        """An empty `CLEOPATRA_CACHE_DIR` is treated as unset."""
        monkeypatch.setenv("CLEOPATRA_CACHE_DIR", "")
        assert Config.get_cache_dir() == Path.home() / ".cleopatra" / "naturalearth"

    def test_does_not_create_the_directory(self, monkeypatch, tmp_path):
        """Resolving the path must not create it (the getter is side-effect free).

        Test scenario:
            A non-existent target is resolved but never created on disk, so
            the getter is safe to call merely to discover the location.
        """
        target = tmp_path / "should_not_be_created"
        monkeypatch.setenv("CLEOPATRA_CACHE_DIR", str(target))
        result = Config.get_cache_dir()
        assert result == target
        assert not target.exists(), "get_cache_dir must not create the directory"

    def test_accepts_path_object_not_just_str(self, monkeypatch, tmp_path):
        """The `path` argument accepts any `os.PathLike`, not only `str`.

        Test scenario:
            A `pathlib.Path` passed as `path` is honoured, confirming the
            `str | os.PathLike` contract.
        """
        monkeypatch.setenv("CLEOPATRA_CACHE_DIR", str(tmp_path / "from_env"))
        explicit = tmp_path / "as_path_object"
        assert Config.get_cache_dir(explicit) == explicit

    def test_always_returns_a_path(self, monkeypatch, tmp_path):
        """The return value is always a `pathlib.Path`, for every branch.

        Test scenario:
            Default, env-var, and explicit-arg resolutions each return a
            `Path` instance (so callers can chain path operations).
        """
        monkeypatch.delenv("CLEOPATRA_CACHE_DIR", raising=False)
        assert isinstance(Config.get_cache_dir(), Path), "default branch must return Path"
        monkeypatch.setenv("CLEOPATRA_CACHE_DIR", str(tmp_path))
        assert isinstance(Config.get_cache_dir(), Path), "env branch must return Path"
        assert isinstance(
            Config.get_cache_dir(str(tmp_path)), Path
        ), "explicit branch must return Path"

    def test_is_a_staticmethod(self):
        """`get_cache_dir` works without an instance (parity with the other helper).

        Test scenario:
            Calling it on the class (not an instance) resolves normally, and
            it does not appear as a bound method requiring `self`.
        """
        assert isinstance(
            inspect.getattr_static(Config, "get_cache_dir"), staticmethod
        ), "get_cache_dir should be a staticmethod"


class TestSetMatplotlibBackend:
    def test_set_set_matplotlib_backend(self):
        Config.set_matplotlib_backend()
        backend = plt.get_backend()
        assert backend == "TkAgg" or backend == "Agg"
        # reset the backend to the agg for the tests for run without UI
        matplotlib.use("agg")


def test_is_notebook():
    assert not is_notebook()


def test_is_notebook_true_in_zmq_shell(monkeypatch):
    """A ZMQInteractiveShell (Jupyter / qtconsole) is detected as a notebook."""
    ipython = pytest.importorskip("IPython")

    class ZMQInteractiveShell:
        pass

    monkeypatch.setattr(ipython, "get_ipython", lambda: ZMQInteractiveShell())
    assert is_notebook()


def test_is_notebook_false_in_terminal_shell(monkeypatch):
    """A TerminalInteractiveShell (plain IPython REPL) is not a notebook."""
    ipython = pytest.importorskip("IPython")

    class TerminalInteractiveShell:
        pass

    monkeypatch.setattr(ipython, "get_ipython", lambda: TerminalInteractiveShell())
    assert not is_notebook()
