"""OGC WMS and WMTS services addressed as tile providers.

`cleopatra.basemap.tiles` talks to a provider through exactly one method --
`build_url(x=, y=, z=)` returning an http(s) URL (`tiles.fetch_single_tile`).
Everything downstream of that call is service-agnostic: the fetch returns opaque
bytes, the mosaic is stitched from `dict[Tile, bytes]`, and the composite is
drawn with `ax.imshow`. So a service that is *not* an XYZ template needs no new
pipeline -- only a different way to turn a tile into a URL.

That is all this module is. `WMTSProvider` and `WMSProvider` are frozen
dataclasses satisfying that one-method contract, so they reach the renderer
through the same public entry point an XYZ provider does:

```python
from cleopatra.basemap.ogc import WMSProvider
from cleopatra.basemap.tiles import add_tiles

add_tiles(ax, WMSProvider(url="https://example.org/wms", layers="ortho"))
```

Both are checked at construction rather than mid-render, and both are hashable,
so a provider can key a dict or a cache. `extra_params` is copied behind a
read-only view, so nothing a caller edits afterwards can change the URLs an
already-built provider produces.

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

Two limits are worth knowing before you reach them. `tiles.world_texture` keys
its disk cache on `provider.get("name", ...)`, so it needs a Mapping-like
provider and raises `AttributeError` on these dataclasses -- use `add_tiles`,
or an XYZ provider for a world texture. And on the RESTful WMTS branch the
service fixes the format and version in its own template, so `image_format` and
`version` are validated but never sent; they apply to the KVP branch only.

Adding these providers did change one thing next door. Because `extra_params`
is documented here as the place to put an API key, `tiles.fetch_single_tile`
now logs a *redacted* URL when an attempt fails: `tiles._redact_url` keeps the
query parameter names -- which is what makes a failure diagnosable -- and drops
every value, so a credential no longer outlives the session in a debug log. The
request that goes on the wire is untouched.

Importing this module pulls in nothing from the `[tiles]` extra -- neither
`xyzservices` nor `pyproj` is touched, and the tile-grid helpers it does use are
plain arithmetic. The extra is required only once you actually render, and
`add_tiles` raises the install hint itself if it is missing.
"""

from __future__ import annotations

import re
import urllib.parse
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from decimal import Decimal
from types import MappingProxyType

from cleopatra.basemap.tiles import Tile, _tile_xy_bounds

#: The tile-matrix set this module's geometry assumes, and the default of
#: `WMTSProvider.tile_matrix_set`. Web Mercator, `2**z` columns and rows,
#: top-left origin at +/-20 037 508.34 m, 256 px square tiles -- the grid
#: `cleopatra.basemap.tiles` implements.
GOOGLE_MAPS_COMPATIBLE = "GoogleMapsCompatible"

#: WMS versions that send the CRS as `SRS=`. 1.3.0 renamed it to `CRS=` and made
#: the `BBOX` axis order follow the CRS's own declared order.
_WMS_SRS_VERSIONS = ("1.0.0", "1.1.0", "1.1.1")

#: WMS versions that send the CRS as `CRS=`.
_WMS_CRS_VERSIONS = ("1.3.0",)

#: The published WMTS versions. OGC has only ever issued 1.0.0; validating it
#: keeps the two providers consistent, rather than one refusing an unknown
#: version and the other accepting any non-empty string.
_WMTS_VERSIONS = ("1.0.0",)

#: Matches one `{}`-delimited placeholder in a RESTful WMTS template. The name
#: is ASCII letters only, so `{Tile_Row}` or `{Time2}` is not read as a
#: placeholder at all and survives substitution untouched, exactly like a
#: well-formed name this module does not own.
_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z]+)\}")

#: The placeholders a RESTful WMTS template may carry, lower-cased for
#: case-insensitive lookup. OGC spells them `{TileMatrix}` and so on, but
#: services are inconsistent about it and a mis-cased one used to fall through
#: to the KVP branch and ship a URL with literal braces in it.
_RESTFUL_FIELDS = (
    "tilematrixset",
    "tilematrix",
    "tilerow",
    "tilecol",
    "layer",
    "style",
)

#: The placeholders a RESTful template cannot do without. Without all three the
#: template addresses one fixed image, so every tile of the mosaic would be the
#: same picture -- silently, since the request itself succeeds.
_REQUIRED_RESTFUL_FIELDS = ("tilematrix", "tilerow", "tilecol")


def _validate_endpoint(url: str, field_name: str) -> None:
    """Reject an endpoint that could never be fetched.

    `tiles.fetch_single_tile` refuses a non-http(s) URL anyway, but it does so
    per tile, deep inside a render. Failing at construction names the field.

    Args:
        url: The endpoint or template to check.
        field_name: The dataclass field the value came from, for the message.

    Returns:
        None

    Raises:
        ValueError: If `url` is not a string, is empty or blank, carries
            leading or trailing whitespace, or does not use the `http` or
            `https` scheme. The message names `field_name`. Padding is refused
            rather than trimmed: the value is stored and sent verbatim, so
            trimming it silently would hide a copy-paste error.
    """
    if not isinstance(url, str) or not url.strip():
        raise ValueError(f"{field_name} must be a non-empty string, got {url!r}.")
    if url != url.strip():
        raise ValueError(
            f"{field_name} has leading or trailing whitespace, got {url!r}. It would "
            f"be sent verbatim, so it is refused rather than quietly trimmed."
        )
    scheme = urllib.parse.urlsplit(url).scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError(
            f"{field_name} must be an http(s) URL, got {url!r} (scheme {scheme!r})."
        )


def _validate_text(value: str, field_name: str) -> None:
    """Reject a non-string where an optional string is expected.

    `styles` and `attribution` may legitimately be empty, so they escaped the
    non-empty check -- and with it any type check at all. `styles=None` was
    then sent to the service as the four characters `None`, which is a style
    name it does not have, and the resulting service exception arrived here as
    an unreadable tile.

    Args:
        value: The value to check.
        field_name: The dataclass field the value came from, for the message.

    Returns:
        None

    Raises:
        ValueError: If `value` is not a string. The empty string passes -- that
            is the point of the check being separate from
            `_validate_identifier`. The message names `field_name`.
    """
    if not isinstance(value, str):
        raise ValueError(
            f"{field_name} must be a string (empty is allowed), got {value!r}."
        )


def _validate_identifier(value: str, field_name: str) -> None:
    """Reject an empty layer/format identifier.

    Only the identifiers a service actually requires go through here. A WMS
    `styles` may legitimately be empty -- that is how a caller asks for the
    service's own default -- so it is deliberately not checked.

    Args:
        value: The value to check.
        field_name: The dataclass field the value came from, for the message.

    Returns:
        None

    Raises:
        ValueError: If `value` is not a string, or is empty or whitespace only.
            The message names `field_name`.
    """
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string, got {value!r}.")


def _freeze_params(extra_params: Mapping[str, str]) -> Mapping[str, str]:
    """Copy `extra_params` behind a read-only view, coercing it to `str -> str`.

    A `frozen=True` dataclass blocks attribute assignment but not mutation of a
    dict it holds, so a caller's later edit would otherwise change the provider's
    URLs underneath it.

    Keys and values are passed through `str`, so a token or version id that
    arrives from JSON or YAML as an `int` is stored exactly as the equivalent
    string literal would be, and two providers configured alike compare equal
    whatever literal types built them. An already-frozen view -- what
    `dataclasses.replace` feeds back in -- is a `Mapping` too, so it is copied
    again rather than aliased into the new instance.

    Args:
        extra_params: The mapping the caller passed.

    Returns:
        Mapping[str, str]: A read-only `MappingProxyType` over a fresh copy,
        with every key and value coerced to `str`.

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

    The URL is taken apart and put back together rather than concatenated, so
    an endpoint that publishes a mandatory parameter of its own survives having
    more added after it, and two shapes that broke naive concatenation are
    handled: a trailing `?` (which carries an *empty* query, so testing the
    query for truthiness appended a second `?` and renamed the first parameter
    to `?SERVICE`), and a fragment (which would otherwise swallow the whole
    generated query, transmitting none of it). Keys and values are
    percent-encoded by `urllib.parse.urlencode`.

    Args:
        base: The endpoint, with or without an existing query string.
        params: The parameters to add, already ordered.

    Returns:
        str: The full URL. `base` is returned unchanged when `params` is empty.

    Examples:
        - A bare endpoint gets a `?`:
            ```python
            >>> from cleopatra.basemap.ogc import _query
            >>> _query("https://example.org/wms", {"REQUEST": "GetMap"})
            'https://example.org/wms?REQUEST=GetMap'

            ```
        - An endpoint that already carries a query gets a `&`, and the added
          value is escaped:
            ```python
            >>> from cleopatra.basemap.ogc import _query
            >>> _query("https://example.org/wms?map=/etc/base.map", {"token": "a&b"})
            'https://example.org/wms?map=/etc/base.map&token=a%26b'

            ```
        - A trailing `?` is absorbed rather than doubled:
            ```python
            >>> from cleopatra.basemap.ogc import _query
            >>> _query("https://example.org/wms?", {"SERVICE": "WMS"})
            'https://example.org/wms?SERVICE=WMS'

            ```
        - A fragment stays at the end, where it cannot swallow the query:
            ```python
            >>> from cleopatra.basemap.ogc import _query
            >>> _query("https://example.org/wms#layers", {"SERVICE": "WMS"})
            'https://example.org/wms?SERVICE=WMS#layers'

            ```
        - Nothing to add leaves the endpoint alone:
            ```python
            >>> from cleopatra.basemap.ogc import _query
            >>> _query("https://example.org/wms", {})
            'https://example.org/wms'

            ```
    """
    encoded = urllib.parse.urlencode(params)
    if not encoded:
        return base
    parts = urllib.parse.urlsplit(base)
    merged = f"{parts.query}&{encoded}" if parts.query else encoded
    return urllib.parse.urlunsplit(parts._replace(query=merged))


def _restful_placeholders(url: str) -> set[str]:
    """The known RESTful placeholder names a template carries, lower-cased.

    The whole URL is scanned, not only its path, because some services publish
    the tile triple in the query. Names this module does not own are dropped in
    the same pass, which is what stops a service's own `{Time}` from satisfying
    the required-placeholder check or dragging a KVP endpoint onto the RESTful
    branch.

    Args:
        url: The endpoint or template.

    Returns:
        set[str]: The subset of `_RESTFUL_FIELDS` present, however cased.

    Examples:
        - Casing does not matter, and the names come back lower-cased:
            ```python
            >>> from cleopatra.basemap.ogc import _restful_placeholders
            >>> found = _restful_placeholders("https://e.org/{TILEMATRIX}/{tilerow}/{TileCol}")
            >>> sorted(found)
            ['tilecol', 'tilematrix', 'tilerow']

            ```
        - A name this module does not own is filtered out, so a URL carrying
          only unknown ones reads as KVP:
            ```python
            >>> from cleopatra.basemap.ogc import _restful_placeholders
            >>> _restful_placeholders("https://e.org/{Time}.png")
            set()
            >>> sorted(_restful_placeholders("https://e.org/{Custom}/{TileMatrix}"))
            ['tilematrix']

            ```
        - The scan reaches the query, not just the path:
            ```python
            >>> from cleopatra.basemap.ogc import _restful_placeholders
            >>> sorted(_restful_placeholders("https://e.org/wmts?tm={TileMatrix}&r={TileRow}"))
            ['tilematrix', 'tilerow']

            ```
    """
    found = {match.group(1).lower() for match in _PLACEHOLDER_RE.finditer(url)}
    return found & set(_RESTFUL_FIELDS)


def _merge_params(
    generated: Mapping[str, str], extra: Mapping[str, str]
) -> dict[str, str]:
    """Merge `extra` over `generated`, matching keys case-insensitively.

    OGC parameter names are case-insensitive, so `{"format": "image/jpeg"}` is
    meant to replace the generated `FORMAT`. Merging with `**` only replaces on
    an exact match, which instead sent both `FORMAT=image/png` and
    `format=image/jpeg` and left the service to pick -- the opposite of the
    override the caller asked for.

    The reconciliation is between the generated parameters and `extra_params`
    only. A parameter baked into the endpoint's own query is kept verbatim by
    `_query` and is never overridden -- so an endpoint of
    `https://host/wms?FORMAT=image/jpeg` plus `extra_params={"format": ...}`
    sends both, and the service chooses. Put such a parameter in
    `extra_params` rather than in the `url` if you need to control it.

    Args:
        generated: The parameters this provider builds.
        extra: The caller's `extra_params`, which win over the generated ones.

    Returns:
        dict[str, str]: The merged parameters, in generated-then-extra order.

    Examples:
        - A differently-cased key replaces rather than duplicates:
            ```python
            >>> from cleopatra.basemap.ogc import _merge_params
            >>> _merge_params({"FORMAT": "image/png"}, {"format": "image/jpeg"})
            {'format': 'image/jpeg'}

            ```
        - An unrelated key is simply added:
            ```python
            >>> from cleopatra.basemap.ogc import _merge_params
            >>> _merge_params({"FORMAT": "image/png"}, {"token": "abc"})
            {'FORMAT': 'image/png', 'token': 'abc'}

            ```
    """
    overridden = {key.upper() for key in extra}
    merged = {
        key: value for key, value in generated.items() if key.upper() not in overridden
    }
    merged.update(extra)
    return merged


def _format_coordinate(value: float) -> str:
    """Render one BBOX coordinate in plain decimal notation, losing nothing.

    Python's default float formatting switches to an exponent for small
    magnitudes, and a tile adjacent to the projection origin has bounds like
    `-5.5e-10`. A `BBOX` carrying `-5.529727786779404e-10` is not what the WMS
    grammar asks for, and services vary between rejecting it, reading it as
    zero and reading it as garbage -- so a render centred on Greenwich and the
    equator failed for no visible reason.

    Rounding to a fixed number of decimals would fix that but cost the exact
    round trip the whole WMS adaptation rests on: the image is requested for
    precisely the bounds the mosaic will place it at. `Decimal` of a float is
    that float's exact binary value, so formatting it with `f` is both
    exponent-free and lossless -- `float(_format_coordinate(v)) == v` for every
    `v`. Trailing zeros are trimmed to keep the URL readable.

    Args:
        value: The coordinate in EPSG:3857 metres.

    Returns:
        str: The coordinate without an exponent, parsing back to exactly
        `value`.

    Examples:
        - A near-zero bound stays decimal instead of turning into an exponent:
            ```python
            >>> from cleopatra.basemap.ogc import _format_coordinate
            >>> rendered = _format_coordinate(-5.529727786779404e-10)
            >>> "e" in rendered
            False
            >>> float(rendered) == -5.529727786779404e-10
            True

            ```
        - An ordinary bound round-trips exactly:
            ```python
            >>> from cleopatra.basemap.ogc import _format_coordinate
            >>> float(_format_coordinate(5009377.085697311)) == 5009377.085697311
            True

            ```
        - A whole number loses its decimal point rather than gaining zeros:
            ```python
            >>> from cleopatra.basemap.ogc import _format_coordinate
            >>> _format_coordinate(-20037508.0)
            '-20037508'

            ```
    """
    text = format(Decimal(value), "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("", "-", "-0") else text


def _reduce_provider(provider: object) -> tuple:
    """Rebuild instructions for `pickle` and `copy`.

    `__post_init__` stores `extra_params` as a `MappingProxyType`, which cannot
    be pickled -- so `pickle.dumps` and `copy.deepcopy` both raised, even though
    the class documents itself as immutable and cache-friendly. Reconstructing
    through the constructor with a plain dict avoids that, and has the side
    benefit of re-running validation on the rebuilt copy.

    Args:
        provider: The dataclass instance to reduce.

    Returns:
        tuple: The `(callable, args)` pair `pickle` and `copy` use to rebuild.
    """
    values = []
    for spec in fields(provider):
        value = getattr(provider, spec.name)
        values.append(dict(value) if isinstance(value, Mapping) else value)
    return (type(provider), tuple(values))


def _hash_provider(provider: object) -> int:
    """Hash a provider by its fields, flattening the mapping one holds.

    `frozen=True` generates a `__hash__`, but it hashes the field tuple -- and
    `extra_params` is a mapping, which is unhashable. That made every provider
    unhashable even with the default empty mapping, so one could not be used as
    a dict key, a set member or an `lru_cache` argument, despite the class
    advertising itself as frozen. Flattening the mapping to its sorted items
    keeps the hash consistent with the generated `__eq__`, which compares those
    same fields by value. Every other field is a validated `str`, `bool` or
    `int`, so it is hashable as it stands.

    The type itself is part of the key, not its name: two providers in
    different modules could share a name, and `__eq__` already refuses to
    compare across types.

    Args:
        provider: The dataclass instance to hash.

    Returns:
        int: A hash over the provider's type and every field, with any mapping
        field flattened to its sorted items.
    """
    values = []
    for spec in fields(provider):
        value = getattr(provider, spec.name)
        values.append(
            tuple(sorted(value.items())) if isinstance(value, Mapping) else value
        )
    return hash((type(provider), *values))


@dataclass(frozen=True)
class WMTSProvider:
    """An OGC WMTS service addressed as a tile provider.

    Satisfies the `build_url(x=, y=, z=)` contract
    `cleopatra.basemap.tiles.fetch_single_tile` calls, so it can be handed
    straight to `add_tiles` as the `source`.

    Both request encodings are supported. If `url` carries any placeholder
    this module owns -- `{TileMatrix}`, `{TileRow}`, `{TileCol}`,
    `{TileMatrixSet}`, `{Layer}` or `{Style}`, in whatever casing the service
    publishes them -- it is treated as a RESTful template and those
    placeholders are substituted; otherwise `url` is taken as a KVP endpoint
    and a `GetTile` query is built. A template has to address a tile, so one
    carrying placeholders but missing `{TileMatrix}`, `{TileRow}` or
    `{TileCol}` is refused at construction rather than shipped as a mosaic of
    one repeated image.

    The service description is checked once, at construction -- including on a
    `dataclasses.replace` copy, which is the supported way to vary a frozen
    provider. The instance is then immutable and hashable, so it can key a dict
    or a cache; `extra_params` is stored as a read-only copy, so a caller's
    later edit to the dict they passed cannot rewrite the URLs.

    Args:
        url: The `GetTile` KVP endpoint, or a RESTful template containing
            `{TileMatrix}`, `{TileRow}` and `{TileCol}` -- all three, in any
            casing -- and optionally `{TileMatrixSet}`, `{Layer}` or
            `{Style}`. Leading or trailing whitespace is refused rather than
            trimmed, because the value is sent verbatim.
        layer: The `Layer` identifier to request.
        tile_matrix_set: The tile-matrix set identifier, defaulting to
            `GOOGLE_MAPS_COMPATIBLE`. Only `GoogleMapsCompatible` (or a
            service's own name for that same Web Mercator grid, e.g.
            `GoogleMapsCompatible_Level9`) lines up with this package's tile
            geometry -- any other grid returns tiles that will be placed
            wrongly. Nothing validates the grid beyond it being non-empty,
            because the identifier is the service's to name.
        style: The `Style` identifier. Most services publish `"default"`. It
            is percent-encoded where it fills a `{Style}` placeholder, so a
            name carrying `/`, `?` or a space cannot invent a path segment or
            start a query.
        image_format: The `Format` to request. `tiles._looks_like_image`
            accepts PNG, JPEG, GIF and WebP, so a TIFF or SVG format will be
            rejected as an unreadable tile. Sent on the KVP branch only -- a
            RESTful template fixes the format in its own path.
        version: The WMTS version. OGC has only ever published 1.0.0, so that
            is the only accepted value -- checked here as well, rather than
            `WMSProvider` refusing an unknown version while this one takes any
            non-empty string. Sent as `VERSION` on the KVP branch only; a
            RESTful template encodes it in the endpoint, but it is validated
            either way.
        attribution: Credit line, which may be empty but must be a string.
            `add_tiles(attribution=True)` reads this attribute and draws it on
            the axes.
        extra_params: Extra query parameters, merged last so they can also
            override a generated one -- matched case-insensitively, as OGC
            parameter names are, so `{"format": ...}` replaces the
            generated `FORMAT` rather than joining it. A parameter already
            in the `url`'s own query is kept verbatim and is *not*
            overridden this way. On the RESTful branch there are no generated
            parameters to override, so these are simply appended as the query.
            This is where an API key or token goes; a failed fetch logs the
            URL with every query value redacted, so the key does not reach the
            debug log. Keys and values are coerced to `str` and stored
            read-only.

    Raises:
        ValueError: If `url` is not a non-empty http(s) string or carries
            leading or trailing whitespace; if `layer`, `tile_matrix_set`,
            `style`, `image_format` or `version` is not a non-empty string; if
            `version` is anything but `1.0.0`; if `attribution` is not a string
            (empty is allowed); or if `url` carries RESTful placeholders
            without all of `{TileMatrix}`, `{TileRow}` and `{TileCol}`.
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
        - `extra_params` carries a credential, escaped into the query:
            ```python
            >>> from urllib.parse import parse_qs, urlsplit
            >>> from cleopatra.basemap.ogc import WMTSProvider
            >>> provider = WMTSProvider(
            ...     url="https://example.org/wmts",
            ...     layer="TrueColor",
            ...     extra_params={"api key": "a&b"},
            ... )
            >>> url = provider.build_url(x=0, y=0, z=0)
            >>> url.endswith("api+key=a%26b")
            True
            >>> parse_qs(urlsplit(url).query)["api key"]
            ['a&b']

            ```
        - A provider is frozen and hashable, so it can key a cache:
            ```python
            >>> from cleopatra.basemap.ogc import WMTSProvider
            >>> first = WMTSProvider(url="https://example.org/wmts", layer="TrueColor")
            >>> second = WMTSProvider(url="https://example.org/wmts", layer="TrueColor")
            >>> first == second
            True
            >>> len({first, second})
            1
            >>> {first: "imagery"}[second]
            'imagery'

            ```
        - An unusable endpoint is refused at construction, not mid-render:
            ```python
            >>> from cleopatra.basemap.ogc import WMTSProvider
            >>> WMTSProvider(url="file:///tiles/wmts", layer="TrueColor")
            Traceback (most recent call last):
                ...
            ValueError: url must be an http(s) URL, got 'file:///tiles/wmts' (scheme 'file').

            ```
        - So is a template that cannot address a tile -- every tile would
          otherwise resolve to the same picture, silently:
            ```python
            >>> from cleopatra.basemap.ogc import WMTSProvider
            >>> try:
            ...     WMTSProvider(url="https://example.org/w/{TileMatrix}.png", layer="L")
            ... except ValueError as error:
            ...     print(str(error).split(".")[0])
            a RESTful WMTS template must address a tile: url is missing {tilerow}, {tilecol}

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
        """Validate the service description and freeze `extra_params`.

        Runs on a `dataclasses.replace` copy as well as on a direct
        construction, so a copy is checked as thoroughly as the original. The
        version is matched against the one published WMTS version rather than
        merely required to be non-empty, so the two providers refuse the same
        class of mistake; `attribution` may be empty but is still required to
        be a string, since a non-string one would reach the axes as its `repr`;
        and a template carrying placeholders is required to carry the three
        that identify a tile.

        Returns:
            None

        Raises:
            ValueError: If `url` is not a non-empty http(s) string or carries
                leading or trailing whitespace; if `layer`, `tile_matrix_set`,
                `style`, `image_format` or `version` is not a non-empty string;
                if `version` is anything but `1.0.0`; if `attribution` is not a
                string (empty is allowed); or if `url` carries RESTful
                placeholders without all of `{TileMatrix}`, `{TileRow}` and
                `{TileCol}`.
            TypeError: If `extra_params` is not a mapping.
        """
        _validate_endpoint(self.url, "url")
        _validate_identifier(self.layer, "layer")
        _validate_identifier(self.tile_matrix_set, "tile_matrix_set")
        _validate_identifier(self.style, "style")
        _validate_identifier(self.image_format, "image_format")
        _validate_identifier(self.version, "version")
        if self.version not in _WMTS_VERSIONS:
            listed = ", ".join(repr(v) for v in _WMTS_VERSIONS)
            raise ValueError(f"version must be one of {listed}, got {self.version!r}.")
        _validate_text(self.style, "style")
        _validate_text(self.attribution, "attribution")
        present = _restful_placeholders(self.url)
        if present:
            missing = [f for f in _REQUIRED_RESTFUL_FIELDS if f not in present]
            if missing:
                listed = ", ".join(f"{{{f}}}" for f in missing)
                raise ValueError(
                    f"a RESTful WMTS template must address a tile: url is "
                    f"missing {listed}. Without them every tile resolves to "
                    f"the same URL, so the mosaic would repeat one image."
                )
        object.__setattr__(self, "extra_params", _freeze_params(self.extra_params))

    def __reduce__(self) -> tuple:
        """Rebuild through the constructor; see `_reduce_provider`.

        Returns:
            tuple: The `(callable, args)` pair `pickle` and `copy` use.
        """
        return _reduce_provider(self)

    def __hash__(self) -> int:
        """Hash the service description, flattening `extra_params`.

        `frozen=True` would generate a `__hash__` over the field tuple, which
        the `extra_params` mapping makes unhashable; `_hash_provider` hashes
        that mapping's sorted items instead, staying consistent with the
        generated `__eq__`.

        Returns:
            int: A hash consistent with this dataclass's own equality.
        """
        return _hash_provider(self)

    @property
    def is_restful(self) -> bool:
        """Whether `url` is a RESTful template rather than a KVP endpoint.

        Any placeholder this module owns is the marker -- `{TileMatrix}`,
        `{TileRow}`, `{TileCol}`, `{TileMatrixSet}`, `{Layer}` or `{Style}` --
        matched case-insensitively, because services are inconsistent about
        the spelling and a mis-cased one used to fail this test, fall through
        to the KVP branch and ship a URL with literal braces still in it. Its
        presence is what selects `build_url`'s substitution branch over the
        `GetTile` query branch.

        A name this module does not own -- a service's own `{Time}`, say --
        does not count, and neither does a query string in the endpoint.
        Because `__post_init__` refuses a template missing `{TileMatrix}`,
        `{TileRow}` or `{TileCol}`, a constructed provider that answers `True`
        always carries all three.

        Returns:
            bool: `True` when `url` carries at least one recognised
            placeholder.

        Examples:
            - A plain endpoint is KVP, so a `GetTile` query gets built:
                ```python
                >>> from cleopatra.basemap.ogc import WMTSProvider
                >>> provider = WMTSProvider(url="https://example.org/wmts", layer="L")
                >>> provider.is_restful
                False
                >>> "REQUEST=GetTile" in provider.build_url(x=1, y=1, z=1)
                True

                ```
            - A template carrying the marker is substituted in place instead:
                ```python
                >>> from cleopatra.basemap.ogc import WMTSProvider
                >>> provider = WMTSProvider(
                ...     url="https://example.org/{TileMatrix}/{TileRow}/{TileCol}.png",
                ...     layer="L",
                ... )
                >>> provider.is_restful
                True
                >>> provider.build_url(x=4, y=2, z=3)
                'https://example.org/3/2/4.png'

                ```
            - The marker is matched case-insensitively, so a lower-cased
              template is filled in rather than shipped with literal braces:
                ```python
                >>> from cleopatra.basemap.ogc import WMTSProvider
                >>> provider = WMTSProvider(
                ...     url="https://example.org/{tilematrix}/{tilerow}/{tilecol}.png",
                ...     layer="L",
                ... )
                >>> provider.is_restful
                True
                >>> provider.build_url(x=4, y=2, z=3)
                'https://example.org/3/2/4.png'

                ```
            - An endpoint that merely carries a query is still KVP, and so is
              one whose only placeholder this module does not own:
                ```python
                >>> from cleopatra.basemap.ogc import WMTSProvider
                >>> WMTSProvider(url="https://example.org/wmts?layer=x", layer="L").is_restful
                False
                >>> WMTSProvider(url="https://example.org/wmts/{Time}.png", layer="L").is_restful
                False

                ```
        """
        return bool(_restful_placeholders(self.url))

    def build_url(self, *, x: int, y: int, z: int) -> str:
        """Return the `GetTile` URL for one tile.

        The WMTS tile triple is the slippy triple under another name:
        `z` is `TileMatrix`, `y` is `TileRow`, `x` is `TileCol`.

        A RESTful `url` (see `is_restful`) has its placeholders filled in, in
        one left-to-right pass over the whole URL -- query included, not just
        the path. Four things follow from that. Names are matched
        case-insensitively, so whatever spelling the service publishes works.
        Each value is percent-encoded, so a layer or style name carrying `/`,
        `?` or a space cannot invent a path segment or start a query. The pass
        never re-scans what it has already written, so a field whose value
        itself looks like a placeholder stays data instead of being rewritten
        by a later field. And a placeholder this module does not own -- a
        service's own `{Time}` -- is left exactly as it stands, for the caller
        to see rather than silently mangled. `extra_params` is then appended as
        a query, with an `&` if the filled-in template already carries one.

        Otherwise a `GetTile` KVP query is appended to `url`, keeping any query
        it already carries, and `extra_params` comes last so it can override a
        generated parameter -- matched case-insensitively -- as well as add
        one.

        Args:
            x: Tile column, i.e. `TileCol`.
            y: Tile row, i.e. `TileRow`.
            z: Zoom level, i.e. `TileMatrix`.

        Returns:
            str: The full request URL.

        Examples:
            - A KVP endpoint keeps the query it already carries and gains the
              `GetTile` parameters:
                ```python
                >>> from urllib.parse import parse_qs, urlsplit
                >>> from cleopatra.basemap.ogc import WMTSProvider
                >>> provider = WMTSProvider(
                ...     url="https://example.org/wmts?map=/etc/base.map",
                ...     layer="TrueColor",
                ... )
                >>> query = parse_qs(urlsplit(provider.build_url(x=1, y=1, z=1)).query)
                >>> query["map"]
                ['/etc/base.map']
                >>> query["SERVICE"], query["TILEMATRIXSET"]
                (['WMTS'], ['GoogleMapsCompatible'])

                ```
            - A template is filled in as far as its placeholders go, and
              `extra_params` follows it as a query:
                ```python
                >>> from cleopatra.basemap.ogc import WMTSProvider
                >>> provider = WMTSProvider(
                ...     url="https://example.org/wmts/{TileMatrix}/{TileRow}/{TileCol}.png",
                ...     layer="TrueColor",
                ...     extra_params={"token": "abc"},
                ... )
                >>> provider.build_url(x=4, y=2, z=3)
                'https://example.org/wmts/3/2/4.png?token=abc'

                ```
            - `extra_params` wins over a parameter the provider generates, so a
              service demanding its own spelling needs no code change:
                ```python
                >>> from urllib.parse import parse_qs, urlsplit
                >>> from cleopatra.basemap.ogc import WMTSProvider
                >>> provider = WMTSProvider(
                ...     url="https://example.org/wmts",
                ...     layer="TrueColor",
                ...     extra_params={"FORMAT": "image/jpeg"},
                ... )
                >>> parse_qs(urlsplit(provider.build_url(x=0, y=0, z=0)).query)["FORMAT"]
                ['image/jpeg']

                ```
            - Casing does not matter and every substituted value is
              percent-encoded, so a layer name cannot break out of its segment:
                ```python
                >>> from cleopatra.basemap.ogc import WMTSProvider
                >>> provider = WMTSProvider(
                ...     url="https://example.org/w/{layer}/{tilematrix}/{tilerow}/{tilecol}.png",
                ...     layer="a b/c",
                ... )
                >>> provider.build_url(x=4, y=2, z=3)
                'https://example.org/w/a%20b%2Fc/3/2/4.png'

                ```
            - A placeholder this module does not own survives untouched, and a
              value that merely looks like one is not substituted a second
              time:
                ```python
                >>> from cleopatra.basemap.ogc import WMTSProvider
                >>> provider = WMTSProvider(
                ...     url="https://example.org/{Custom}/{Layer}/{TileMatrix}/{TileRow}/{TileCol}.png",
                ...     layer="{TileRow}",
                ... )
                >>> provider.build_url(x=4, y=2, z=3)
                'https://example.org/{Custom}/%7BTileRow%7D/3/2/4.png'

                ```
        """
        if self.is_restful:
            values = {
                "tilematrixset": self.tile_matrix_set,
                "tilematrix": str(z),
                "tilerow": str(y),
                "tilecol": str(x),
                "layer": self.layer,
                "style": self.style,
            }

            def substitute(match: re.Match[str]) -> str:
                """Replace one placeholder, leaving an unknown one alone."""
                name = match.group(1).lower()
                if name not in values:
                    return match.group(0)
                return urllib.parse.quote(values[name], safe="")

            # One pass, so a value that happens to look like a placeholder is
            # not rewritten again by a later field.
            filled = _PLACEHOLDER_RE.sub(substitute, self.url)
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
        }
        return _query(self.url, _merge_params(params, self.extra_params))


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

    The service description is checked once, at construction -- including on a
    `dataclasses.replace` copy, which is the supported way to vary a frozen
    provider. The instance is then immutable and hashable, so it can key a dict
    or a cache; `extra_params` is stored as a read-only copy, so a caller's
    later edit to the dict they passed cannot rewrite the URLs.

    Args:
        url: The `GetMap` endpoint. An existing query string is preserved.
            Leading or trailing whitespace is refused rather than trimmed,
            because the value is sent verbatim.
        layers: The comma-separated `Layers` value to request.
        styles: The comma-separated `Styles` value. Empty means the service's
            default style, which is what most callers want -- so this is the
            one identifier that is allowed to be empty, and it is still sent.
            It must still be a string: `None` used to reach the service as the
            four characters `None`, and came back as a service exception
            disguised as an unreadable tile.
        version: The WMS version. `"1.3.0"` sends `CRS=`; `"1.1.1"` and older
            send `SRS=`.
        image_format: The `Format` to request. `tiles._looks_like_image`
            accepts PNG, JPEG, GIF and WebP, so a TIFF or SVG format will be
            rejected as an unreadable tile.
        transparent: Whether to request `TRANSPARENT=TRUE`, so an overlay layer
            composites over what is already on the axes. It must be an actual
            `bool`: any truthy value would otherwise send `TRANSPARENT=TRUE`,
            so a string like `"no"` would mean its own opposite.
        tile_size: The `WIDTH`/`HEIGHT` in pixels for each `GetMap`. Keep this
            at 256 unless the service refuses it -- `tiles.stitch_tiles` infers
            the mosaic's cell size from the first decoded image, so mixing
            sizes within one render would mis-stitch.
        attribution: Credit line, which may be empty but must be a string.
            `add_tiles(attribution=True)` reads this attribute and draws it on
            the axes.
        extra_params: Extra query parameters, merged last so they can also
            override a generated one -- matched case-insensitively, as OGC
            parameter names are, so `{"format": ...}` replaces the
            generated `FORMAT` rather than joining it. A parameter already
            in the `url`'s own query is kept verbatim and is *not*
            overridden this way. This is where an API key or token goes; a
            failed fetch logs the URL with every query value redacted, so the
            key does not reach the debug log. Keys and values are coerced to
            `str` and stored read-only.

    Raises:
        ValueError: If `url` is not a non-empty http(s) string or carries
            leading or trailing whitespace; if `layers`, `image_format` or
            `version` is not a non-empty string; if `version` is not one of
            `1.0.0`, `1.1.0`, `1.1.1` or `1.3.0`; if `styles` or `attribution`
            is not a string (empty is allowed); if `transparent` is not a
            `bool`; or if `tile_size` is not a positive `int` (`bool` is
            rejected too).
        TypeError: If `extra_params` is not a mapping.

    Examples:
        - One tile becomes one `GetMap`, with the tile's Web Mercator bounds
          as `BBOX` in `left,bottom,right,top` metres:
            ```python
            >>> from urllib.parse import parse_qs, urlsplit
            >>> from cleopatra.basemap.ogc import WMSProvider
            >>> provider = WMSProvider(url="https://example.org/wms", layers="ortho")
            >>> query = parse_qs(urlsplit(provider.build_url(x=4, y=2, z=3)).query)
            >>> [round(float(v)) for v in query["BBOX"][0].split(",")]
            [0, 5009377, 5009377, 10018754]
            >>> query["WIDTH"], query["HEIGHT"], query["REQUEST"]
            (['256'], ['256'], ['GetMap'])

            ```
        - 1.3.0 names the CRS `CRS` and 1.1.1 names it `SRS`, but the value is
          `EPSG:3857` either way:
            ```python
            >>> from urllib.parse import parse_qs, urlsplit
            >>> from cleopatra.basemap.ogc import WMSProvider
            >>> latest = WMSProvider(url="https://example.org/wms", layers="ortho")
            >>> latest.version, latest.crs_parameter
            ('1.3.0', 'CRS')
            >>> parse_qs(urlsplit(latest.build_url(x=0, y=0, z=0)).query)["CRS"]
            ['EPSG:3857']
            >>> older = WMSProvider(
            ...     url="https://example.org/wms", layers="ortho", version="1.1.1"
            ... )
            >>> parse_qs(urlsplit(older.build_url(x=0, y=0, z=0)).query)["SRS"]
            ['EPSG:3857']

            ```
        - An unsupported version is refused at construction:
            ```python
            >>> from cleopatra.basemap.ogc import WMSProvider
            >>> WMSProvider(url="https://example.org/wms", layers="ortho", version="2.0.0")
            Traceback (most recent call last):
                ...
            ValueError: version must be one of '1.0.0', '1.1.0', '1.1.1', '1.3.0', got '2.0.0'.

            ```
        - A near-miss on a flag is refused rather than read as truthy:
            ```python
            >>> from cleopatra.basemap.ogc import WMSProvider
            >>> try:
            ...     WMSProvider(
            ...         url="https://example.org/wms", layers="ortho", transparent="no"
            ...     )
            ... except ValueError as error:
            ...     print(str(error).split(".")[0])
            transparent must be a bool, got 'no'

            ```
        - `styles` and `attribution` may be empty, but not `None`:
            ```python
            >>> from cleopatra.basemap.ogc import WMSProvider
            >>> blank = WMSProvider(
            ...     url="https://example.org/wms", layers="ortho", styles="", attribution=""
            ... )
            >>> blank.styles, blank.attribution
            ('', '')
            >>> try:
            ...     WMSProvider(url="https://example.org/wms", layers="ortho", styles=None)
            ... except ValueError as error:
            ...     print(error)
            styles must be a string (empty is allowed), got None.

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
        """Validate the service description and freeze `extra_params`.

        Runs on a `dataclasses.replace` copy as well as on a direct
        construction, so a copy is checked as thoroughly as the original. The
        version is matched against the supported set rather than guessed at,
        because choosing `CRS=` or `SRS=` wrongly would misplace every tile
        silently. `styles` and `attribution` may be empty but must still be
        strings -- being allowed to be empty is how they escaped the non-empty
        check and, with it, any type check at all -- and `transparent` must be
        a real `bool` rather than merely truthy.

        Returns:
            None

        Raises:
            ValueError: If `url` is not a non-empty http(s) string or carries
                leading or trailing whitespace; if `layers`, `image_format` or
                `version` is not a non-empty string; if `version` is not one of
                `1.0.0`, `1.1.0`, `1.1.1` or `1.3.0`; if `styles` or
                `attribution` is not a string (empty is allowed); if
                `transparent` is not a `bool`; or if `tile_size` is not a
                positive `int` (`bool` is rejected too).
            TypeError: If `extra_params` is not a mapping.
        """
        _validate_endpoint(self.url, "url")
        _validate_identifier(self.layers, "layers")
        _validate_identifier(self.image_format, "image_format")
        _validate_identifier(self.version, "version")
        supported = _WMS_SRS_VERSIONS + _WMS_CRS_VERSIONS
        if self.version not in supported:
            listed = ", ".join(repr(v) for v in supported)
            raise ValueError(f"version must be one of {listed}, got {self.version!r}.")
        _validate_text(self.styles, "styles")
        _validate_text(self.attribution, "attribution")
        if not isinstance(self.transparent, bool):
            raise ValueError(
                f"transparent must be a bool, got {self.transparent!r}. Any truthy "
                f"value would otherwise send TRANSPARENT=TRUE."
            )
        if (
            not isinstance(self.tile_size, int)
            or isinstance(self.tile_size, bool)
            or self.tile_size < 1
        ):
            raise ValueError(
                f"tile_size must be a positive int, got {self.tile_size!r}."
            )
        object.__setattr__(self, "extra_params", _freeze_params(self.extra_params))

    def __reduce__(self) -> tuple:
        """Rebuild through the constructor; see `_reduce_provider`.

        Returns:
            tuple: The `(callable, args)` pair `pickle` and `copy` use.
        """
        return _reduce_provider(self)

    def __hash__(self) -> int:
        """Hash the service description, flattening `extra_params`.

        `frozen=True` would generate a `__hash__` over the field tuple, which
        the `extra_params` mapping makes unhashable; `_hash_provider` hashes
        that mapping's sorted items instead, staying consistent with the
        generated `__eq__`.

        Returns:
            int: A hash consistent with this dataclass's own equality.
        """
        return _hash_provider(self)

    @property
    def crs_parameter(self) -> str:
        """The query key this WMS version uses for the CRS.

        1.3.0 renamed 1.1.1's `SRS=` to `CRS=`. Only the key changes: the value
        `build_url` sends is `EPSG:3857` either way, because that is the CRS of
        the tile grid. Pinning it there is also why 1.3.0's axis-order rule
        never bites -- it orders `BBOX` by the CRS's own declared axis order,
        which for EPSG:3857 is easting then northing, the same order 1.1.1
        always used. The CRS that flips to latitude-first under 1.3.0 is
        EPSG:4326, and this module never asks for it.

        Returns:
            str: `"CRS"` for 1.3.0, `"SRS"` for 1.1.1 and older.

        Examples:
            - 1.3.0 is the default, and it names the key `CRS`:
                ```python
                >>> from urllib.parse import parse_qs, urlsplit
                >>> from cleopatra.basemap.ogc import WMSProvider
                >>> provider = WMSProvider(url="https://example.org/wms", layers="ortho")
                >>> provider.crs_parameter
                'CRS'
                >>> parse_qs(urlsplit(provider.build_url(x=0, y=0, z=0)).query)["CRS"]
                ['EPSG:3857']

                ```
            - An older version carries the same value under `SRS`, and sends no
              `CRS` at all:
                ```python
                >>> from urllib.parse import parse_qs, urlsplit
                >>> from cleopatra.basemap.ogc import WMSProvider
                >>> provider = WMSProvider(
                ...     url="https://example.org/wms", layers="ortho", version="1.1.1"
                ... )
                >>> provider.crs_parameter
                'SRS'
                >>> query = parse_qs(urlsplit(provider.build_url(x=0, y=0, z=0)).query)
                >>> query["SRS"]
                ['EPSG:3857']
                >>> "CRS" in query
                False

                ```
            - 1.3.0 is the only supported version on the `CRS` side of the
              rename:
                ```python
                >>> from cleopatra.basemap.ogc import WMSProvider
                >>> [
                ...     WMSProvider(
                ...         url="https://example.org/wms", layers="ortho", version=v
                ...     ).crs_parameter
                ...     for v in ("1.0.0", "1.1.0", "1.1.1", "1.3.0")
                ... ]
                ['SRS', 'SRS', 'SRS', 'CRS']

                ```
        """
        return "CRS" if self.version in _WMS_CRS_VERSIONS else "SRS"

    def build_url(self, *, x: int, y: int, z: int) -> str:
        """Return the `GetMap` URL covering one tile.

        The tile index itself is never sent. It is turned into the tile's
        EPSG:3857 bounds with `tiles._tile_xy_bounds` and passed as `BBOX`,
        with `WIDTH`/`HEIGHT` fixed at `tile_size`. Any query `url` already
        carries is kept, and `extra_params` comes last, so it can override a
        generated parameter as well as add one.

        Args:
            x: Tile column.
            y: Tile row.
            z: Zoom level.

        Returns:
            str: The full request URL, with `BBOX` set to the tile's EPSG:3857
            bounds and `WIDTH`/`HEIGHT` to `tile_size`.

        Examples:
            - The `BBOX` is exactly the bounds the tile grid gives that tile:
                ```python
                >>> from urllib.parse import parse_qs, urlsplit
                >>> from cleopatra.basemap.ogc import WMSProvider
                >>> from cleopatra.basemap.tiles import Tile, _tile_xy_bounds
                >>> provider = WMSProvider(url="https://example.org/wms", layers="ortho")
                >>> query = parse_qs(urlsplit(provider.build_url(x=4, y=2, z=3)).query)
                >>> sent = tuple(float(v) for v in query["BBOX"][0].split(","))
                >>> sent == _tile_xy_bounds(Tile(4, 2, 3))
                True
                >>> [round(v) for v in sent]
                [0, 5009377, 5009377, 10018754]

                ```
            - `tile_size` is the pixel size each `GetMap` asks for:
                ```python
                >>> from urllib.parse import parse_qs, urlsplit
                >>> from cleopatra.basemap.ogc import WMSProvider
                >>> provider = WMSProvider(
                ...     url="https://example.org/wms", layers="ortho", tile_size=512
                ... )
                >>> query = parse_qs(urlsplit(provider.build_url(x=0, y=0, z=0)).query)
                >>> query["WIDTH"], query["HEIGHT"]
                (['512'], ['512'])

                ```
            - `transparent=False` asks for an opaque image, for a base layer
              rather than an overlay:
                ```python
                >>> from urllib.parse import parse_qs, urlsplit
                >>> from cleopatra.basemap.ogc import WMSProvider
                >>> opaque = WMSProvider(
                ...     url="https://example.org/wms", layers="ortho", transparent=False
                ... )
                >>> query = parse_qs(urlsplit(opaque.build_url(x=0, y=0, z=0)).query)
                >>> query["TRANSPARENT"], query["LAYERS"]
                (['FALSE'], ['ortho'])

                ```
        """
        bounds = _tile_xy_bounds(Tile(x, y, z))
        bbox = ",".join(_format_coordinate(value) for value in bounds)
        params = {
            "SERVICE": "WMS",
            "REQUEST": "GetMap",
            "VERSION": self.version,
            "LAYERS": self.layers,
            "STYLES": self.styles,
            self.crs_parameter: "EPSG:3857",
            "BBOX": bbox,
            "WIDTH": str(self.tile_size),
            "HEIGHT": str(self.tile_size),
            "FORMAT": self.image_format,
            "TRANSPARENT": "TRUE" if self.transparent else "FALSE",
        }
        return _query(self.url, _merge_params(params, self.extra_params))
