import matplotlib
import matplotlib.pyplot as plt


class Config:
    """Configuration class for the cleopatra package."""

    def __init__(self):
        pass

    @staticmethod
    def set_matplotlib_backend(backend: str = None, interactive: bool = False):
        """Set Matplotlib backend.

        Set the matplotlib backend based on the user's environment or explicitly provided backend.

        Args:
            backend: The name of the matplotlib backend to use. If None, the backend is chosen based on the environment.
            interactive: If True, use an interactive backend in Jupyter notebooks. Default is False (static backend).
        """
        # close all open figures
        plt.close("all")

        if backend:
            plt.switch_backend(backend)
            print(f"Matplotlib backend set to {backend}")
        else:
            if is_notebook():
                if interactive:
                    get_ipython().run_line_magic("matplotlib", "notebook")  # noqa: F821
                    # plt.switch_backend('nbAgg')
                    print(
                        "Matplotlib backend set to interactive backend for Jupyter notebook"
                    )
                else:
                    # Running in a Jupyter notebook
                    get_ipython().run_line_magic("matplotlib", "inline")  # noqa: F821
                    print(
                        "Matplotlib backend set to inline for static plots in Jupyter notebook"
                    )
            else:
                try:
                    # Running in an IDE or script
                    # plt.switch_backend("TkAgg")
                    matplotlib.use("TkAgg")
                    print("Matplotlib backend set to TkAgg for script or IDE")
                except ImportError:
                    plt.switch_backend("Agg")
                    print("Matplotlib backend set to Agg (non-interactive)")


def is_notebook():
    """
    Returns True if the code is running in a Jupyter notebook.
    """
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (probably not an IPython environment)
    except NameError:
        return False  # Probably standard Python interpreter
