"""Configuration helpers for cleopatra.

`Config` gathers the package's cross-cutting, user-facing settings in one
discoverable place:

* `Config.set_matplotlib_backend` — opt-in matplotlib backend selection.
  Importing cleopatra does **not** change the active backend; picking one
  is the application's responsibility, not a library's.
* `Config.get_cache_dir` — where cleopatra caches downloaded basemap
  assets. Resolves an explicit argument, then the `CLEOPATRA_CACHE_DIR`
  environment variable, then the default `~/.cleopatra/naturalearth`; it
  only resolves the path and does not create the directory.
"""

import logging
import os
from pathlib import Path

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
            from IPython import get_ipython

            magic = "notebook" if interactive else "inline"
            get_ipython().run_line_magic("matplotlib", magic)
            logger.info("Matplotlib set to %%matplotlib %s for Jupyter", magic)
        else:
            plt.switch_backend("Agg")
            logger.info("Matplotlib backend set to Agg (non-interactive)")
        return matplotlib.get_backend()

    @staticmethod
    def get_cache_dir(path: str | os.PathLike | None = None) -> Path:
        """Resolve the directory cleopatra caches downloaded basemap assets in.

        This is the single, discoverable home for cleopatra's on-disk
        cache setting — the Natural Earth vectors and hypsometric relief
        downloaded by `cleopatra.basemap.reference`. Resolution order:

        1. a non-empty explicit `path` argument;
        2. the `CLEOPATRA_CACHE_DIR` environment variable, if set;
        3. the default `~/.cleopatra/naturalearth`.

        A falsy `path` (`None` or an empty string) is treated as "not
        provided" and falls through to the environment variable and the
        default, so `get_cache_dir("")` behaves like `get_cache_dir()`. A
        leading `~` is expanded. This function only **resolves** the path;
        it does not create the directory (the download helpers create it
        on first use), so it is safe to call just to discover where the
        cache lives.

        Args:
            path: An explicit cache directory to use, overriding the
                environment variable and the default. A falsy value
                (`None` or `""`) is treated as not provided. A relative
                path (from `path` or the environment variable) is kept
                relative and resolved against the current working
                directory when the directory is created. Default `None`.

        Returns:
            pathlib.Path: The resolved (not necessarily existing) cache
            directory.

        Examples:
            - An explicit `path` is resolved as given (and overrides
                everything else):
                ```python
                >>> from cleopatra.config import Config
                >>> Config.get_cache_dir("/data/cleopatra").as_posix()
                '/data/cleopatra'

                ```
            - With no argument, the `CLEOPATRA_CACHE_DIR` environment
                variable is honoured:
                ```python
                >>> import os
                >>> from cleopatra.config import Config
                >>> os.environ["CLEOPATRA_CACHE_DIR"] = "/var/cache/cleopatra"
                >>> Config.get_cache_dir().as_posix()
                '/var/cache/cleopatra'
                >>> del os.environ["CLEOPATRA_CACHE_DIR"]

                ```
            - An explicit argument wins over the environment variable, and
                the returned path composes into an asset path:
                ```python
                >>> import os
                >>> from cleopatra.config import Config
                >>> os.environ["CLEOPATRA_CACHE_DIR"] = "/ignored"
                >>> asset = Config.get_cache_dir("/data") / "ne_110m_coastline.geojson.gz"
                >>> asset.as_posix()
                '/data/ne_110m_coastline.geojson.gz'
                >>> del os.environ["CLEOPATRA_CACHE_DIR"]

                ```

        See Also:
            cleopatra.basemap.reference: Downloads basemap assets into this
                directory (its private `_cache_dir` resolves the location
                through this method and creates the directory on first use).
        """
        if not path:
            path = os.environ.get("CLEOPATRA_CACHE_DIR")
        if path:
            return Path(path).expanduser()
        return Path.home() / ".cleopatra" / "naturalearth"


def is_notebook() -> bool:
    """Return True if the code is running in a Jupyter notebook / qtconsole."""
    try:
        from IPython import get_ipython
    except ModuleNotFoundError:
        return False  # IPython is not installed.

    # Only Jupyter / qtconsole report "ZMQInteractiveShell"; a terminal IPython
    # ("TerminalInteractiveShell") or any other environment is not a notebook.
    shell = get_ipython().__class__.__name__
    return shell == "ZMQInteractiveShell"
