# Config Module

The `cleopatra.config` module provides a small, **opt-in** helper for selecting the
matplotlib backend. Importing `cleopatra` does *not* change the backend on its own —
picking a backend is the application's job, not a library's.

Call `Config.set_matplotlib_backend()` yourself if you want cleopatra to choose a sensible
one for you: `%matplotlib inline` inside a Jupyter notebook (or `%matplotlib notebook`
when `interactive=True`), otherwise `Agg` in a plain script. You can also pass an explicit
backend name. `set_matplotlib_backend` is a `staticmethod`, so `Config.set_matplotlib_backend(...)`
works without an instance. Note that switching the backend closes any open figures — that
is matplotlib's behaviour — so call it before you start plotting.

```python
from cleopatra.config import Config

Config.set_matplotlib_backend("Agg")          # explicit
Config.set_matplotlib_backend()               # auto: inline in notebooks, Agg otherwise
```

## Module Documentation

::: cleopatra.config
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3
