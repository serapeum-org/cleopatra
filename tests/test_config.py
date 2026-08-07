import matplotlib
import matplotlib.pyplot as plt
import pytest

from cleopatra.config import Config, is_notebook


def test_create_config():
    assert Config()


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
