# Config Module

The `cleopatra.config` module gathers cleopatra's cross-cutting, user-facing settings in
one discoverable place: an **opt-in** matplotlib-backend helper and the on-disk **cache
directory** used for downloaded basemap assets. Importing `cleopatra` does *not* change
the backend on its own — picking a backend is the application's job, not a library's.

## Matplotlib backend

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

## Cache directory

`Config.get_cache_dir()` resolves where cleopatra caches the basemap assets it downloads —
the Natural Earth vectors and hypsometric relief used by
[`cleopatra.basemap.reference`](reference-data.md). It is the discoverable home for that
setting, resolved in this order: an explicit `path` argument, then the
`CLEOPATRA_CACHE_DIR` environment variable, then the default `~/.cleopatra/naturalearth`.
A leading `~` is expanded. The getter only **resolves** the path — it does not create the
directory (the download helpers create it on first use), so it is safe to call just to
discover where the cache lives.

```python
from cleopatra.config import Config

Config.get_cache_dir()                         # ~/.cleopatra/naturalearth (default)
# or set CLEOPATRA_CACHE_DIR=/data/cleopatra to override, or:
Config.get_cache_dir("/data/cleopatra")        # explicit override
```

## Module Documentation

::: cleopatra.config
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3
