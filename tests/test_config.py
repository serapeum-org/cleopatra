import matplotlib
import matplotlib.pyplot as plt

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
