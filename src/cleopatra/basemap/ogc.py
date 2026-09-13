"""OGC WMS and WMTS services addressed as tile providers.

`cleopatra.basemap.tiles` talks to a provider through exactly one method --
`build_url(x=, y=, z=)` returning an http(s) URL (`tiles.fetch_single_tile`).
Everything downstream of that call is service-agnostic: the fetch returns opaque
bytes, the mosaic is stitched from `dict[Tile, bytes]`, and the composite is
drawn with `ax.imshow`. So a service that is *not* an XYZ template needs no new
pipeline -- only a different way to turn a tile into a URL.

That is all this module is. `WMTSProvider` and `WMSProvider` are frozen
dataclasses satisfying that one-method contract, so they work through the
unchanged public entry point:

```python
from cleopatra.basemap.ogc import WMSProvider
from cleopatra.basemap.tiles import add_tiles

add_tiles(ax, WMSProvider(url="https://example.org/wms", layers="ortho"))
```

The two service kinds reach the same place differently:

- **WMTS is a tiling scheme.** A `GetTile` request is a
  `(TileMatrix, TileRow, TileCol)` triple, which *is* `(z, y, x)`. On the
  `GoogleMapsCompatible` matrix set -- EPSG:3857, `2**z` columns, top-left
  origin, square tiles -- that is exactly the grid `tiles._lonlat_to_tile_xy`
  already computes, so the mapping is the identity.
- **WMS is a single-image request.** A `GetMap` takes a `BBOX` plus
  `WIDTH`/`HEIGHT`, not a tile index. It fits by asking for *one `GetMap` per
  tile*: `tiles._tile_xy_bounds` already returns precisely the EPSG:3857 bounds
  a `BBOX` needs, and the size is fixed at the tile size.

Only the `GoogleMapsCompatible` tile-matrix set is supported, because that is
the single grid the surrounding module implements. A WMTS published on any other
matrix set (NASA GIBS' `EPSG4326_250m`, a national grid such as EPSG:28992) will
return tiles that do not line up; see `WMTSProvider.tile_matrix_set`.

Importing this module pulls in nothing from the `[tiles]` extra -- neither
`xyzservices` nor `pyproj` is touched, and the tile-grid helpers it does use are
plain arithmetic. The extra is required only once you actually render, and
`add_tiles` raises the install hint itself if it is missing.
"""

from __future__ import annotations

import urllib.parse
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from types import MappingProxyType

from cleopatra.basemap.tiles import Tile, _tile_xy_bounds

#: The tile-matrix set this module's geometry assumes. Web Mercator, `2**z`
#: columns and rows, top-left origin at +/-20 037 508.34 m, 256 px square tiles
#: -- the grid `cleopatra.basemap.tiles` implements.
GOOGLE_MAPS_COMPATIBLE = "GoogleMapsCompatible"

#: WMS versions that send the CRS as `SRS=`. 1.3.0 renamed it to `CRS=` and made
#: the `BBOX` axis order follow the CRS's own declared order.
_WMS_SRS_VERSIONS = ("1.0.0", "1.1.0", "1.1.1")

#: WMS versions that send the CRS as `CRS=`.
_WMS_CRS_VERSIONS = ("1.3.0",)

#: The `{}`-delimited placeholders an OGC RESTful WMTS template may carry.
#: Presence of `{TileMatrix}` is what marks a `url` as RESTful rather than KVP.
_RESTFUL_MARKER = "{TileMatrix}"


def _validate_endpoint(url: str, field_name: str) -> None:
    """Reject an endpoint that could never be fetched.

    `tiles.fetch_single_tile` refuses a non-http(s) URL anyway, but it does so
    per tile, deep inside a render. Failing at construction names the field.

    Args:
        url: The endpoint or template to check.
        field_name: The dataclass field the value came from, for the message.

    Raises:
        ValueError: If `url` is empty or does not use the http(s) scheme.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError(f"{field_name} must be a non-empty string, got {url!r}.")
    scheme = urllib.parse.urlsplit(url).scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError(
            f"{field_name} must be an http(s) URL, got {url!r} (scheme {scheme!r})."
        )


def _validate_identifier(value: str, field_name: str) -> None:
    """Reject an empty layer/format identifier.

    Args:
        value: The value to check.
        field_name: The dataclass field the value came from, for the message.

    Raises:
        ValueError: If `value` is not a non-empty string.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string, got {value!r}.")


def _freeze_params(extra_params: Mapping[str, str]) -> Mapping[str, str]:
    """Copy `extra_params` behind a read-only view.

    A `frozen=True` dataclass blocks attribute assignment but not mutation of a
    dict it holds, so a caller's later edit would otherwise change the provider's
    URLs underneath it.

    Args:
        extra_params: The mapping the caller passed.

    Returns:
        Mapping[str, str]: A read-only view over a copy.

    Raises:
        TypeError: If `extra_params` is not a mapping.
    """
    if not isinstance(extra_params, Mapping):
        raise TypeError(
            f"extra_params must be a mapping of query parameters, "
            f"got {type(extra_params).__name__}."
        )
    return MappingProxyType({str(k): str(v) for k, v in extra_params.items()})


def _query(base: str, params: Mapping[str, str]) -> str:
    """Append `params` to `base`, preserving any query it already carries.

    Args:
        base: The endpoint, with or without an existing query string.
        params: The parameters to add, already ordered.

    Returns:
        str: The full URL.
    """
    encoded = urllib.parse.urlencode(params)
    if not encoded:
        return base
    separator = "&" if urllib.parse.urlsplit(base).query else "?"
    return f"{base}{separator}{encoded}"


def _hash_provider(provider: object) -> int:
    """Hash a provider by its fields, flattening the mapping one holds.

    `frozen=True` generates a `__hash__`, but it hashes the field tuple -- and
    `extra_params` is a mapping, which is unhashable. That made every provider
    unhashable even with the default empty mapping, so one could not be used as
    a dict key, a set member or an `lru_cache` argument, despite the class
    advertising itself as frozen. Flattening the mapping to its sorted items
    keeps the hash consistent with the generated `__eq__`, which compares those
    same fields by value.

    Args:
        provider: The dataclass instance to hash.

    Returns:
        int: A hash over the type and every field.
    """
    values = []
    for spec in fields(provider):
        value = getattr(provider, spec.name)
        values.append(
            tuple(sorted(value.items())) if isinstance(value, Mapping) else value
        )
    return hash((type(provider).__name__, *values))


@dataclass(frozen=True)
class WMTSProvider:
    """An OGC WMTS service addressed as a tile provider.

    Satisfies the `build_url(x=, y=, z=)` contract
    `cleopatra.basemap.tiles.fetch_single_tile` calls, so it can be handed
    straight to `add_tiles` as the `source`.

    Both request encodings are supported. If `url` contains the RESTful
    placeholder `{TileMatrix}` it is treated as a template and the placeholders
    are substituted; otherwise `url` is taken as a KVP endpoint and a `GetTile`
    query is built.

    Args:
        url: The `GetTile` KVP endpoint, or a RESTful template containing
            `{TileMatrix}`, `{TileRow}` and `{TileCol}` (and optionally
            `{TileMatrixSet}`, `{Layer}`, `{Style}`).
        layer: The `Layer` identifier to request.
        tile_matrix_set: The tile-matrix set identifier. Only
            `GoogleMapsCompatible` (or a service's own name for that same Web
            Mercator grid, e.g. `GoogleMapsCompatible_Level9`) lines up with
            this package's tile geometry -- any other grid returns tiles that
            will be placed wrongly.
        style: The `Style` identifier. Most services publish `"default"`.
        image_format: The `Format` to request. `tiles._looks_like_image`
            accepts PNG, JPEG, GIF and WebP, so a TIFF or SVG format will be
            rejected as an unreadable tile.
        version: The WMTS version, sent as `VERSION` in KVP requests.
        attribution: Credit line. `add_tiles(attribution=True)` reads this
            attribute and draws it on the axes.
        extra_params: Extra query parameters, merged last so they can also
            override a generated one. This is where an API key or token goes.

    Raises:
        ValueError: If `url` is not an http(s) URL, or if `layer`,
            `tile_matrix_set`, `style`, `image_format` or `version` is empty.
        TypeError: If `extra_params` is not a mapping.

    Examples:
        - Build a `GetTile` KVP request and read back the tile triple:
            ```python
            >>> from urllib.parse import parse_qs, urlsplit
            >>> from cleopatra.basemap.ogc import WMTSProvider
            >>> provider = WMTSProvider(
            ...     url="https://example.org/wmts",
            ...     layer="TrueColor",
            ... )
            >>> query = parse_qs(urlsplit(provider.build_url(x=4, y=2, z=3)).query)
            >>> query["TILEMATRIX"], query["TILEROW"], query["TILECOL"]
            (['3'], ['2'], ['4'])
            >>> query["REQUEST"], query["LAYER"]
            (['GetTile'], ['TrueColor'])

            ```
        - A RESTful template substitutes in place instead:
            ```python
            >>> from cleopatra.basemap.ogc import WMTSProvider
            >>> provider = WMTSProvider(
            ...     url="https://example.org/wmts/{Layer}/{TileMatrix}/{TileRow}/{TileCol}.png",
            ...     layer="TrueColor",
            ... )
            >>> provider.build_url(x=4, y=2, z=3)
            'https://example.org/wmts/TrueColor/3/2/4.png'

            ```
        - `extra_params` carries a key, and is escaped:
            ```python
            >>> from cleopatra.basemap.ogc import WMTSProvider
            >>> provider = WMTSProvider(
            ...     url="https://example.org/wmts",
            ...     layer="TrueColor",
            ...     extra_params={"api key": "a&b"},
            ... )
            >>> "api+key=a%26b" in provider.build_url(x=0, y=0, z=0)
            True

            ```
        - An unusable endpoint is refused at construction, not mid-render:
            ```python
            >>> from cleopatra.basemap.ogc import WMTSProvider
            >>> WMTSProvider(url="file:///tiles/wmts", layer="TrueColor")
            Traceback (most recent call last):
                ...
            ValueError: url must be an http(s) URL, got 'file:///tiles/wmts' (scheme 'file').

            ```

    See Also:
        WMSProvider: The same idea for a WMS `GetMap` service.
        cleopatra.basemap.tiles.add_tiles: The renderer both feed.
    """

    url: str
    layer: str
    tile_matrix_set: str = GOOGLE_MAPS_COMPATIBLE
    style: str = "default"
    image_format: str = "image/png"
    version: str = "1.0.0"
    attribution: str = ""
    extra_params: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the service description and freeze `extra_params`."""
        _validate_endpoint(self.url, "url")
        _validate_identifier(self.layer, "layer")
        _validate_identifier(self.tile_matrix_set, "tile_matrix_set")
        _validate_identifier(self.style, "style")
        _validate_identifier(self.image_format, "image_format")
        _validate_identifier(self.version, "version")
        object.__setattr__(self, "extra_params", _freeze_params(self.extra_params))

    def __hash__(self) -> int:
        """Hash the service description; see `_hash_provider`.

        Returns:
            int: A hash consistent with this dataclass's own equality.
        """
        return _hash_provider(self)

    @property
    def is_restful(self) -> bool:
        """Whether `url` is a RESTful template rather than a KVP endpoint.

        Returns:
            bool: `True` when the template carries `{TileMatrix}`.
        """
        return _RESTFUL_MARKER in self.url

    def build_url(self, *, x: int, y: int, z: int) -> str:
        """Return the `GetTile` URL for one tile.

        The WMTS tile triple is the slippy triple under another name:
        `z` is `TileMatrix`, `y` is `TileRow`, `x` is `TileCol`.

        Args:
            x: Tile column, i.e. `TileCol`.
            y: Tile row, i.e. `TileRow`.
            z: Zoom level, i.e. `TileMatrix`.

        Returns:
            str: The full request URL.
        """
        if self.is_restful:
            filled = self.url
            for placeholder, value in (
                ("{TileMatrixSet}", self.tile_matrix_set),
                ("{TileMatrix}", str(z)),
                ("{TileRow}", str(y)),
                ("{TileCol}", str(x)),
                ("{Layer}", self.layer),
                ("{Style}", self.style),
            ):
                filled = filled.replace(placeholder, value)
            return _query(filled, self.extra_params)

        params = {
            "SERVICE": "WMTS",
            "REQUEST": "GetTile",
            "VERSION": self.version,
            "LAYER": self.layer,
            "STYLE": self.style,
            "TILEMATRIXSET": self.tile_matrix_set,
            "TILEMATRIX": str(z),
            "TILEROW": str(y),
            "TILECOL": str(x),
            "FORMAT": self.image_format,
            **self.extra_params,
        }
        return _query(self.url, params)


@dataclass(frozen=True)
class WMSProvider:
    """An OGC WMS service addressed as a tile provider: one `GetMap` per tile.

    Satisfies the `build_url(x=, y=, z=)` contract
    `cleopatra.basemap.tiles.fetch_single_tile` calls, so it can be handed
    straight to `add_tiles` as the `source`.

    A WMS has no tile index; it answers a `BBOX` plus a pixel size. Each tile is
    therefore converted to its Web Mercator bounds with `tiles._tile_xy_bounds`
    and requested as its own image, and the mosaic is stitched as usual. Because
    the request is always pinned to EPSG:3857 -- the CRS of the tile grid -- the
    WMS 1.3.0 axis-order trap does not arise: 1.3.0 orders `BBOX` by the CRS's
    own axis order, which for EPSG:3857 is easting then northing, the same order
    1.1.1 always used. (It is EPSG:4326 under 1.3.0 that flips to
    latitude-first, and that CRS is never requested here.)

    A WMS covering a whole viewport in one request is more efficient than a
    mosaic of many. `add_tiles(..., min_tiles_across=1)` lowers the tile count
    towards one `GetMap` without needing a second code path.

    Args:
        url: The `GetMap` endpoint. An existing query string is preserved.
        layers: The comma-separated `Layers` value to request.
        styles: The comma-separated `Styles` value. Empty means the service's
            default style, which is what most callers want.
        version: The WMS version. `"1.3.0"` sends `CRS=`; `"1.1.1"` and older
            send `SRS=`.
        image_format: The `Format` to request. `tiles._looks_like_image`
            accepts PNG, JPEG, GIF and WebP, so a TIFF or SVG format will be
            rejected as an unreadable tile.
        transparent: Whether to request `TRANSPARENT=TRUE`, so an overlay layer
            composites over what is already on the axes.
        tile_size: The `WIDTH`/`HEIGHT` in pixels for each `GetMap`. Keep this
            at 256 unless the service refuses it -- `tiles.stitch_tiles` infers
            the mosaic's cell size from the first decoded image, so mixing
            sizes within one render would mis-stitch.
        attribution: Credit line. `add_tiles(attribution=True)` reads this
            attribute and draws it on the axes.
        extra_params: Extra query parameters, merged last so they can also
            override a generated one. This is where an API key or token goes.

    Raises:
        ValueError: If `url` is not an http(s) URL, if `layers`, `image_format`
            or `version` is empty, if `version` is not a supported WMS version,
            or if `tile_size` is not a positive integer.
        TypeError: If `extra_params` is not a mapping.

    Examples:
        - The `BBOX` of a request is the tile's own Web Mercator bounds:
            ```python
            >>> from urllib.parse import parse_qs, urlsplit
            >>> from cleopatra.basemap.ogc import WMSProvider
            >>> from cleopatra.basemap.tiles import Tile, _tile_xy_bounds
            >>> provider = WMSProvider(url="https://example.org/wms", layers="ortho")
            >>> query = parse_qs(urlsplit(provider.build_url(x=4, y=2, z=3)).query)
            >>> tuple(float(v) for v in query["BBOX"][0].split(",")) == _tile_xy_bounds(
            ...     Tile(4, 2, 3)
            ... )
            True
            >>> query["WIDTH"], query["HEIGHT"], query["REQUEST"]
            (['256'], ['256'], ['GetMap'])

            ```
        - 1.3.0 names the CRS `CRS`; 1.1.1 names it `SRS`:
            ```python
            >>> from urllib.parse import parse_qs, urlsplit
            >>> from cleopatra.basemap.ogc import WMSProvider
            >>> def keys(version):
            ...     provider = WMSProvider(
            ...         url="https://example.org/wms", layers="ortho", version=version
            ...     )
            ...     return parse_qs(urlsplit(provider.build_url(x=0, y=0, z=0)).query)
            >>> sorted(k for k in keys("1.3.0") if k in ("CRS", "SRS"))
            ['CRS']
            >>> sorted(k for k in keys("1.1.1") if k in ("CRS", "SRS"))
            ['SRS']

            ```
        - An unsupported version is refused at construction:
            ```python
            >>> from cleopatra.basemap.ogc import WMSProvider
            >>> WMSProvider(url="https://example.org/wms", layers="ortho", version="2.0.0")
            Traceback (most recent call last):
                ...
            ValueError: version must be one of '1.0.0', '1.1.0', '1.1.1', '1.3.0', got '2.0.0'.

            ```

    See Also:
        WMTSProvider: The same idea for a WMTS `GetTile` service.
        cleopatra.basemap.tiles.add_tiles: The renderer both feed.
    """

    url: str
    layers: str
    styles: str = ""
    version: str = "1.3.0"
    image_format: str = "image/png"
    transparent: bool = True
    tile_size: int = 256
    attribution: str = ""
    extra_params: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the service description and freeze `extra_params`."""
        _validate_endpoint(self.url, "url")
        _validate_identifier(self.layers, "layers")
        _validate_identifier(self.image_format, "image_format")
        _validate_identifier(self.version, "version")
        supported = _WMS_SRS_VERSIONS + _WMS_CRS_VERSIONS
        if self.version not in supported:
            listed = ", ".join(repr(v) for v in supported)
            raise ValueError(f"version must be one of {listed}, got {self.version!r}.")
        if (
            not isinstance(self.tile_size, int)
            or isinstance(self.tile_size, bool)
            or self.tile_size < 1
        ):
            raise ValueError(
                f"tile_size must be a positive int, got {self.tile_size!r}."
            )
        object.__setattr__(self, "extra_params", _freeze_params(self.extra_params))

    def __hash__(self) -> int:
        """Hash the service description; see `_hash_provider`.

        Returns:
            int: A hash consistent with this dataclass's own equality.
        """
        return _hash_provider(self)

    @property
    def crs_parameter(self) -> str:
        """The query key this WMS version uses for the CRS.

        Returns:
            str: `"CRS"` for 1.3.0, `"SRS"` for 1.1.1 and older.
        """
        return "CRS" if self.version in _WMS_CRS_VERSIONS else "SRS"

    def build_url(self, *, x: int, y: int, z: int) -> str:
        """Return the `GetMap` URL covering one tile.

        Args:
            x: Tile column.
            y: Tile row.
            z: Zoom level.

        Returns:
            str: The full request URL, with `BBOX` set to the tile's EPSG:3857
            bounds and `WIDTH`/`HEIGHT` to `tile_size`.
        """
        left, bottom, right, top = _tile_xy_bounds(Tile(x, y, z))
        params = {
            "SERVICE": "WMS",
            "REQUEST": "GetMap",
            "VERSION": self.version,
            "LAYERS": self.layers,
            "STYLES": self.styles,
            self.crs_parameter: "EPSG:3857",
            "BBOX": f"{left},{bottom},{right},{top}",
            "WIDTH": str(self.tile_size),
            "HEIGHT": str(self.tile_size),
            "FORMAT": self.image_format,
            "TRANSPARENT": "TRUE" if self.transparent else "FALSE",
            **self.extra_params,
        }
        return _query(self.url, params)
