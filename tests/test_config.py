import importlib
import inspect
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytest

import cleopatra.config as config_mod
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

    def test_explicit_arg_tilde_is_expanded(self, monkeypatch):
        """A leading `~` in an explicit `path` argument is expanded too.

        Test scenario:
            The explicit-argument branch also calls `expanduser`, so
            `get_cache_dir("~/foo")` resolves under the home directory
            (guarding against the env branch being special-cased alone).
        """
        monkeypatch.delenv("CLEOPATRA_CACHE_DIR", raising=False)
        assert Config.get_cache_dir("~/foo/bar") == Path.home() / "foo" / "bar"

    def test_whitespace_only_arg_falls_through(self, monkeypatch, tmp_path):
        """A whitespace-only `path` is treated as not provided (honours the env).

        Test scenario:
            `get_cache_dir("   ")` must not create a literally-named
            whitespace directory; it strips to empty and falls through.
        """
        monkeypatch.setenv("CLEOPATRA_CACHE_DIR", str(tmp_path / "from_env"))
        assert Config.get_cache_dir("   ") == tmp_path / "from_env"

    def test_whitespace_only_env_falls_back_to_default(self, monkeypatch):
        """A whitespace-only `CLEOPATRA_CACHE_DIR` falls back to the default.

        Test scenario:
            An all-spaces env value strips to empty and is treated as unset.
        """
        monkeypatch.setenv("CLEOPATRA_CACHE_DIR", "   ")
        assert Config.get_cache_dir() == Path.home() / ".cleopatra" / "naturalearth"

    def test_empty_path_object_is_current_directory(self, monkeypatch):
        """`Path("")` is `Path(".")` under pathlib and resolves to the CWD.

        Test scenario:
            An empty `Path` is indistinguishable from the current directory,
            so it is used as-is (documented contract) rather than falling
            through — callers pass an empty string, not `Path("")`, to fall
            through.
        """
        monkeypatch.setenv("CLEOPATRA_CACHE_DIR", "/ignored")
        assert Config.get_cache_dir(Path("")) == Path(".")

    def test_empty_env_falls_back_to_default(self, monkeypatch):
        """An empty `CLEOPATRA_CACHE_DIR` is treated as unset."""
        monkeypatch.setenv("CLEOPATRA_CACHE_DIR", "")
        assert Config.get_cache_dir() == Path.home() / ".cleopatra" / "naturalearth"

    def test_empty_string_arg_falls_through_to_env(self, monkeypatch, tmp_path):
        """A falsy explicit `path` is treated as not provided (honours the env var).

        Test scenario:
            `get_cache_dir("")` must not short-circuit to the default; an
            empty argument falls through to `CLEOPATRA_CACHE_DIR`.
        """
        monkeypatch.setenv("CLEOPATRA_CACHE_DIR", str(tmp_path / "from_env"))
        assert Config.get_cache_dir("") == tmp_path / "from_env"

    def test_empty_string_arg_falls_back_to_default(self, monkeypatch):
        """A falsy explicit `path` with no env var yields the default.

        Test scenario:
            With `CLEOPATRA_CACHE_DIR` unset, `get_cache_dir("")` behaves
            exactly like `get_cache_dir()`.
        """
        monkeypatch.delenv("CLEOPATRA_CACHE_DIR", raising=False)
        assert Config.get_cache_dir("") == Path.home() / ".cleopatra" / "naturalearth"

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


class TestSetMatplotlibBackendInNotebook:
    """`Config.set_matplotlib_backend(None)` inside a Jupyter kernel.

    With no explicit `backend`, the notebook path runs `%matplotlib` rather
    than `plt.switch_backend`, so the inline/notebook figure plumbing IPython
    installs is preserved. The shell is faked -- these tests never need a
    live kernel.
    """

    @staticmethod
    def _fake_notebook(monkeypatch):
        """Pretend we are in a notebook and record the magics that are run.

        Returns:
            list: The `(magic, argument)` pairs `set_matplotlib_backend` ran.
        """
        calls: list[tuple[str, str]] = []

        class _Shell:
            def run_line_magic(self, magic, argument):
                calls.append((magic, argument))

        ipython = pytest.importorskip("IPython")
        monkeypatch.setattr(config_mod, "is_notebook", lambda: True)
        monkeypatch.setattr(ipython, "get_ipython", lambda: _Shell())
        return calls

    def test_defaults_to_inline_magic(self, monkeypatch):
        """No backend + notebook -> `%matplotlib inline`.

        Test scenario:
            Inside a kernel the magic is used rather than
            `plt.switch_backend`, because it also installs IPython's inline
            figure plumbing that a bare backend switch would skip.
        """
        calls = self._fake_notebook(monkeypatch)

        Config.set_matplotlib_backend()

        assert calls == [("matplotlib", "inline")], f"unexpected magics: {calls}"

    def test_interactive_selects_the_notebook_magic(self, monkeypatch):
        """`interactive=True` + notebook -> `%matplotlib notebook`.

        Test scenario:
            The same notebook branch, selecting the interactive widget
            backend instead of the static inline one.
        """
        calls = self._fake_notebook(monkeypatch)

        Config.set_matplotlib_backend(interactive=True)

        assert calls == [("matplotlib", "notebook")], f"unexpected magics: {calls}"

    def test_explicit_backend_beats_the_notebook_default(self, monkeypatch):
        """An explicit `backend` switches directly instead of running a magic.

        Test scenario:
            Even inside a notebook, an explicit `backend` must take the
            `plt.switch_backend` path -- the magic is only the *default*. The
            switch is spied on rather than performed: really switching would
            leave a global unrestored, and the method's own docstring warns
            that it closes every open figure. A backend name that is not the
            active one is used, so the assertion pins that the caller's choice
            is forwarded rather than coinciding with the suite's Agg default.
        """
        switched: list[str] = []
        monkeypatch.setattr(plt, "switch_backend", switched.append)
        magics = self._fake_notebook(monkeypatch)
        before = matplotlib.get_backend()

        Config.set_matplotlib_backend(backend="TkAgg")

        assert switched == ["TkAgg"], f"expected a direct switch; got {switched}"
        assert magics == [], f"an explicit backend must run no magic; got {magics}"
        assert matplotlib.get_backend() == before, "the active backend must be untouched"


def test_is_notebook_false_without_ipython(monkeypatch):
    """`is_notebook` returns False when IPython is not installed.

    Test scenario:
        A plain pytest run answers `False` anyway (`get_ipython()` returns
        `None` outside a kernel), so a bare "is it False?" assertion cannot
        tell the missing-IPython branch from the ordinary one -- and would
        keep passing if the import blocker silently stopped working. The
        ambient shell is therefore first made to report a notebook, and that
        is asserted: from there, only the import failing can turn the answer
        back to `False`. The blocker is checked directly too.
    """
    ipython = pytest.importorskip("IPython")

    class ZMQInteractiveShell:
        pass

    monkeypatch.setattr(ipython, "get_ipython", lambda: ZMQInteractiveShell())
    assert is_notebook() is True, (
        "precondition: with IPython importable and reporting a kernel shell, "
        "is_notebook must be True -- otherwise the assertion below proves nothing"
    )

    class _BlockIPython:
        """A finder that makes `import IPython` fail, as if it were absent."""

        def find_spec(self, name, path=None, target=None):
            if name == "IPython" or name.startswith("IPython."):
                raise ModuleNotFoundError(f"No module named {name!r}", name=name)
            return None

    monkeypatch.delitem(sys.modules, "IPython", raising=False)
    monkeypatch.setattr(sys, "meta_path", [_BlockIPython(), *sys.meta_path])
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("IPython")

    assert is_notebook() is False
