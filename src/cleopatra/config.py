"""Matplotlib backend helpers for cleopatra.

Importing cleopatra does **not** change the active matplotlib backend —
picking a backend is the application's responsibility, not a library's.
Use `Config.set_matplotlib_backend` if you want a one-liner that selects
a sensible backend (`%matplotlib inline` in a Jupyter notebook, `Agg` in
a plain script).
"""

import logging

import matplotlib

logger = logging.getLogger(__name__)


class Config:
    """Configuration helpers for the cleopatra package."""

    def __init__(self):
        pass

    @staticmethod
    def set_matplotlib_backend(
        backend: str | None = None, interactive: bool = False
    ) -> str:
        """Switch the active matplotlib backend (opt-in helper).

        cleopatra does not call this automatically. It is provided for
        users who want a one-liner to pick a backend. Switching the
        backend **closes every currently-open figure** — that is
        matplotlib's behaviour, not cleopatra's — so call this before you
        start plotting.

        Args:
            backend: Backend name to switch to (e.g. `"Agg"`, `"TkAgg"`,
                `"Qt5Agg"`). If `None`, an environment-appropriate default
                is chosen: `%matplotlib inline` inside a Jupyter notebook
                (or `%matplotlib notebook` when `interactive` is `True`),
                otherwise `"Agg"`.
            interactive: When `backend` is `None` and running inside a
                Jupyter notebook, use the interactive notebook backend
                instead of inline. Ignored otherwise. Default `False`.

        Returns:
            str: The name of the backend that is now active.
        """
        import matplotlib.pyplot as plt

        if backend:
            plt.switch_backend(backend)
            logger.info("Matplotlib backend set to %s", backend)
        elif is_notebook():
            magic = "notebook" if interactive else "inline"
            get_ipython().run_line_magic("matplotlib", magic)  # noqa: F821
            logger.info("Matplotlib set to %%matplotlib %s for Jupyter", magic)
        else:
            plt.switch_backend("Agg")
            logger.info("Matplotlib backend set to Agg (non-interactive)")
        return matplotlib.get_backend()


def is_notebook() -> bool:
    """Return True if the code is running in a Jupyter notebook / qtconsole."""
    try:
        shell = get_ipython().__class__.__name__  # noqa: F821
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (probably not an IPython environment)
    except NameError:
        return False  # Probably standard Python interpreter
