"""Tests for `cleopatra.basemap.ogc` -- WMS and WMTS addressed as tile providers.

URL construction is pure string work and needs nothing installed. The two
end-to-end classes drive the real `add_tiles` pipeline with only the HTTP layer
mocked, which is what proves `build_url` is actually the seam the renderer uses
-- patching `fetch_tiles` instead (as `test_tiles.py` does for other cases)
would never call the provider at all.
"""

from __future__ import annotations

import base64
import dataclasses
from dataclasses import replace
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlsplit

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from cleopatra.basemap import tiles as tiles_mod
from cleopatra.basemap.ogc import (
    WMSProvider,
    WMTSProvider,
    _format_coordinate,
    _merge_params,
    _query,
)
from cleopatra.basemap.tiles import (
    _TILES_AVAILABLE,
    Tile,
    _tile_xy_bounds,
    add_tiles,
    fetch_single_tile,
)

pytestmark = pytest.mark.plot

#: Skips the rendering classes when the optional `[tiles]` extra is absent. The
#: URL-construction classes above deliberately do not use it -- they need none
#: of Pillow, pyproj or xyzservices.
requires_tiles = pytest.mark.skipif(
    not _TILES_AVAILABLE, reason="the [tiles] extra is not installed"
)

#: A real, decodable 1x1 PNG. Hard-coded rather than built with Pillow so this
#: module imports without the extra; `stitch_tiles` takes the mosaic cell size
#: from the first decoded tile, so a 1x1 tile stitches just as faithfully.
ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmM"
    "IQAAAABJRU5ErkJggg=="
)

#: A real, decodable 4x4 PNG, for the one test where the mosaic's cell size is
#: the subject rather than incidental: `stitch_tiles` takes that size from the
#: decoded image, so a 1x1 tile cannot tell a correct grid from a collapsed one.
FOUR_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAYAAACp8Z5+AAAAFUlEQVR4nGPkEpH7z4AEmJA5xAkA"
    "AFB4AUOwjzs/AAAAAElFTkSuQmCC"
)


@pytest.fixture
def wmts():
    """Provide a WMTS provider on a KVP endpoint.

    Returns:
        WMTSProvider: A provider with an attribution line.
    """
    return WMTSProvider(
        url="https://example.org/wmts",
        layer="TrueColor",
        attribution="Imagery provider",
    )


@pytest.fixture
def wms():
    """Provide a WMS provider on a GetMap endpoint.

    Returns:
        WMSProvider: A provider with an attribution line.
    """
    return WMSProvider(
        url="https://example.org/wms",
        layers="ortho",
        attribution="Ortho provider",
    )


@pytest.fixture
def recorded_urls():
    """Patch the HTTP opener to record URLs and return a decodable PNG.

    Yields:
        list[str]: The URLs the pipeline requested, filled in as it runs.
    """
    urls: list[str] = []

    def fake_urlopen(request, timeout=None):
        urls.append(request.full_url)
        response = MagicMock()
        response.read.return_value = ONE_PIXEL_PNG
        return response

    with patch.object(tiles_mod, "urlopen_http", side_effect=fake_urlopen):
        yield urls


@pytest.fixture
def recorded_urls_4px():
    """Patch the HTTP opener to record URLs and return a decodable 4x4 PNG.

    Serves an image whose size matches a `tile_size=4` request, so a render can
    be asserted on the dimensions of the mosaic it stitches.

    Yields:
        list[str]: The URLs the pipeline requested, filled in as it runs.
    """
    urls: list[str] = []

    def fake_urlopen(request, timeout=None):
        urls.append(request.full_url)
        response = MagicMock()
        response.read.return_value = FOUR_PIXEL_PNG
        return response

    with patch.object(tiles_mod, "urlopen_http", side_effect=fake_urlopen):
        yield urls


def query_of(url: str) -> dict[str, list[str]]:
    """Parse a URL's query string.

    Args:
        url: The URL to parse.

    Returns:
        dict[str, list[str]]: The decoded query parameters.
    """
    return parse_qs(urlsplit(url).query)


class TestWMTSProviderBuildUrl:
    """`WMTSProvider.build_url` turns a tile into a GetTile request."""

    def test_tile_triple_maps_to_matrix_row_col(self, wmts):
        """`z`/`y`/`x` become `TileMatrix`/`TileRow`/`TileCol`.

        Args:
            wmts: The KVP provider fixture.

        Test scenario:
            The WMTS triple is the slippy triple renamed, so the mapping is the
            identity. Distinct values for all three catch a transposition.
        """
        query = query_of(wmts.build_url(x=4, y=2, z=3))
        assert query["TILEMATRIX"] == ["3"], f"TileMatrix wrong: {query}"
        assert query["TILEROW"] == ["2"], f"TileRow wrong: {query}"
        assert query["TILECOL"] == ["4"], f"TileCol wrong: {query}"

    def test_service_parameters_are_sent(self, wmts):
        """The request carries the fixed GetTile parameters.

        Args:
            wmts: The KVP provider fixture.

        Test scenario:
            A service rejects a request missing SERVICE/REQUEST/VERSION, so
            their presence is part of the contract, not decoration.
        """
        query = query_of(wmts.build_url(x=0, y=0, z=0))
        assert query["SERVICE"] == ["WMTS"], f"SERVICE wrong: {query}"
        assert query["REQUEST"] == ["GetTile"], f"REQUEST wrong: {query}"
        assert query["VERSION"] == ["1.0.0"], f"VERSION wrong: {query}"
        assert query["LAYER"] == ["TrueColor"], f"LAYER wrong: {query}"
        assert query["STYLE"] == ["default"], f"STYLE wrong: {query}"
        assert query["TILEMATRIXSET"] == ["GoogleMapsCompatible"], f"set: {query}"
        assert query["FORMAT"] == ["image/png"], f"FORMAT wrong: {query}"

    def test_restful_template_is_substituted(self):
        """A template carrying `{TileMatrix}` is filled in, not queried.

        Test scenario:
            RESTful WMTS puts the triple in the path. Building a KVP query for
            such an endpoint would request the wrong URL entirely.
        """
        provider = WMTSProvider(
            url="https://example.org/wmts/{TileMatrixSet}/{Layer}/{Style}"
            "/{TileMatrix}/{TileRow}/{TileCol}.png",
            layer="TrueColor",
        )
        assert provider.build_url(x=4, y=2, z=3) == (
            "https://example.org/wmts/GoogleMapsCompatible/TrueColor/default/3/2/4.png"
        ), f"template not substituted: {provider.build_url(x=4, y=2, z=3)}"

    def test_extra_params_are_merged_and_escaped(self):
        """`extra_params` reaches the query, percent-encoded.

        Test scenario:
            This is the credential seam; a key containing a space or an
            ampersand must not corrupt the query it is added to.
        """
        provider = WMTSProvider(
            url="https://example.org/wmts",
            layer="TrueColor",
            extra_params={"api key": "a&b"},
        )
        assert "api+key=a%26b" in provider.build_url(x=0, y=0, z=0), (
            "extra_params not escaped into the query"
        )

    def test_extra_params_override_a_generated_parameter(self):
        """A caller can force a parameter the provider also generates.

        Test scenario:
            `extra_params` is merged last precisely so a service demanding a
            non-standard spelling can be accommodated without a code change.
        """
        provider = WMTSProvider(
            url="https://example.org/wmts",
            layer="TrueColor",
            extra_params={"FORMAT": "image/jpeg"},
        )
        assert query_of(provider.build_url(x=0, y=0, z=0))["FORMAT"] == [
            "image/jpeg"
        ], "extra_params did not win over the generated FORMAT"

    def test_an_existing_query_string_is_preserved(self):
        """An endpoint that already carries a query keeps it.

        Test scenario:
            Several services publish an endpoint with a mandatory parameter
            baked in; clobbering it with `?` would drop it.
        """
        provider = WMTSProvider(
            url="https://example.org/wmts?map=/etc/base.map", layer="TrueColor"
        )
        query = query_of(provider.build_url(x=1, y=1, z=1))
        assert query["map"] == ["/etc/base.map"], f"pre-existing query lost: {query}"
        assert query["REQUEST"] == ["GetTile"], f"generated query lost: {query}"

    def test_a_partial_restful_template_substitutes_only_its_placeholders(self):
        """A template carrying just the tile triple is filled in as far as it goes.

        Test scenario:
            Plenty of services bake the layer and style into the path they
            publish, leaving a template of `{TileMatrix}/{TileRow}/{TileCol}`
            and nothing else. Substitution must still happen, and the absent
            placeholders must not be invented anywhere in the URL.
        """
        provider = WMTSProvider(
            url="https://example.org/TrueColor/default"
            "/{TileMatrix}/{TileRow}/{TileCol}.png",
            layer="TrueColor",
        )
        url = provider.build_url(x=4, y=2, z=3)
        assert url == "https://example.org/TrueColor/default/3/2/4.png", (
            f"partial template not substituted: {url}"
        )
        assert "{" not in url, f"an unsubstituted placeholder survived: {url}"

    def test_a_restful_template_appends_extra_params_as_a_query(self):
        """`extra_params` is appended to a substituted path, not dropped.

        Test scenario:
            The RESTful branch returns from its own code path, so the
            credential seam has to be proved twice over: a token silently lost
            here turns every tile of a render into a 401.
        """
        provider = WMTSProvider(
            url="https://example.org/wmts/{TileMatrix}/{TileRow}/{TileCol}.png",
            layer="TrueColor",
            extra_params={"token": "abc"},
        )
        url = provider.build_url(x=4, y=2, z=3)
        assert url == "https://example.org/wmts/3/2/4.png?token=abc", (
            f"extra_params not appended to the substituted path: {url}"
        )

    def test_a_restful_template_with_its_own_query_keeps_it(self):
        """A template that already carries a query gets `&`, not a second `?`.

        Test scenario:
            The separator is chosen from the *substituted* URL, so a template
            publishing a mandatory parameter of its own must survive having a
            token added after it -- a second `?` makes the whole tail garbage.
        """
        provider = WMTSProvider(
            url="https://example.org/wmts/{TileMatrix}/{TileRow}/{TileCol}.png?flat=1",
            layer="TrueColor",
            extra_params={"token": "abc"},
        )
        url = provider.build_url(x=4, y=2, z=3)
        assert url == "https://example.org/wmts/3/2/4.png?flat=1&token=abc", (
            f"separator or pre-existing query wrong: {url}"
        )

    def test_extra_params_are_coerced_to_strings(self):
        """Non-string keys and values are stringified when frozen.

        Test scenario:
            A key or token read out of JSON or YAML arrives as an `int` as
            readily as a `str`. Coercing both halves keeps the stored mapping
            uniformly `str -> str`, so two providers configured the same way
            compare equal whatever literal types were used to build them.
        """
        provider = WMTSProvider(
            url="https://example.org/wmts",
            layer="TrueColor",
            extra_params={"version_id": 42},
        )
        assert dict(provider.extra_params) == {"version_id": "42"}, (
            f"extra_params not coerced: {dict(provider.extra_params)}"
        )
        assert query_of(provider.build_url(x=0, y=0, z=0))["version_id"] == ["42"], (
            "the coerced value did not reach the query"
        )


class TestCaseInsensitiveOverride:
    """`extra_params` replaces a generated parameter whatever its casing."""

    @pytest.mark.parametrize(
        "kind, key, value, generated",
        [
            ("wmts", "format", "image/jpeg", "FORMAT"),
            ("wmts", "Layer", "Other", "LAYER"),
            ("wms", "transparent", "FALSE", "TRANSPARENT"),
            ("wms", "styles", "shaded", "STYLES"),
        ],
    )
    def test_a_differently_cased_key_replaces_rather_than_duplicates(
        self, kind, key, value, generated, wmts, wms
    ):
        """The service receives one value for the parameter, not two.

        Args:
            kind: Which provider to check.
            key: The caller's spelling of the parameter.
            value: The value they want sent.
            generated: The provider's own spelling of the same parameter.
            wmts: The WMTS fixture.
            wms: The WMS fixture.

        Test scenario:
            OGC parameter names are case-insensitive, so merging with `**` on an
            exact match sent both `FORMAT=image/png` and `format=image/jpeg` and
            left the service to choose -- the opposite of the documented
            override.
        """
        base = wmts if kind == "wmts" else wms
        provider = replace(base, extra_params={key: value})
        query = query_of(provider.build_url(x=0, y=0, z=0))
        present = [k for k in query if k.upper() == generated]
        assert present == [key], f"expected only {key!r}, got {present}"
        assert query[key] == [value], f"the caller's value did not win: {query}"

    def test_an_unrelated_key_is_still_added(self, wms):
        """Overriding does not stop `extra_params` adding new parameters.

        Args:
            wms: The GetMap provider fixture.

        Test scenario:
            The credential case: a token shares no name with anything the
            provider generates and must simply arrive.
        """
        provider = replace(wms, extra_params={"token": "abc"})
        query = query_of(provider.build_url(x=0, y=0, z=0))
        assert query["token"] == ["abc"], f"token missing: {query}"
        assert query["FORMAT"] == ["image/png"], f"generated key lost: {query}"


class TestMergeParams:
    """`_merge_params` is the case-insensitive merge both providers use."""

    def test_case_insensitive_replacement(self):
        """A lower-case key displaces the generated upper-case one.

        Test scenario:
            The unit-level statement of the rule, so a future caller of the
            helper inherits it rather than re-deriving it.
        """
        merged = _merge_params({"FORMAT": "image/png"}, {"format": "image/jpeg"})
        assert merged == {"format": "image/jpeg"}, f"not replaced: {merged}"

    def test_generated_order_is_kept(self):
        """Untouched generated keys stay in front, in their original order.

        Test scenario:
            A stable parameter order keeps the URLs diffable and the doctests
            deterministic.
        """
        merged = _merge_params({"A": "1", "B": "2"}, {"c": "3"})
        assert list(merged) == ["A", "B", "c"], f"order changed: {list(merged)}"


class TestQueryAssembly:
    """`_query` puts a URL back together rather than concatenating onto it."""

    def test_a_trailing_question_mark_is_not_doubled(self):
        """An endpoint ending in `?` gets one query, not two.

        Test scenario:
            `urlsplit("https://h/wms?").query` is the empty string, so choosing
            the separator by truthiness appended a second `?` and renamed the
            first parameter to `?SERVICE`. WMTS requires `SERVICE`, so every
            tile of a MapServer/GeoServer-style `.../service?` endpoint failed
            with a misleading `ConnectionError`.
        """
        built = _query("https://example.org/service?", {"SERVICE": "WMTS"})
        assert built == "https://example.org/service?SERVICE=WMTS", (
            f"trailing '?' mishandled: {built!r}"
        )

    def test_a_fragment_does_not_swallow_the_query(self):
        """The query goes before the fragment, where it is transmitted.

        Test scenario:
            Appending `?a=b` to `https://h/wms#frag` puts the whole query
            inside the fragment, so the server receives none of it.
        """
        built = _query("https://example.org/wms#layers", {"SERVICE": "WMS"})
        assert urlsplit(built).query == "SERVICE=WMS", f"query lost: {built!r}"
        assert urlsplit(built).fragment == "layers", f"fragment lost: {built!r}"

    def test_an_existing_query_is_extended(self):
        """A mandatory parameter already on the endpoint survives.

        Test scenario:
            Some services publish an endpoint with a parameter baked in;
            replacing rather than extending its query would drop it.
        """
        built = _query("https://example.org/wms?map=/etc/base.map", {"A": "1"})
        query = parse_qs(urlsplit(built).query)
        assert query["map"] == ["/etc/base.map"], f"pre-existing lost: {built!r}"
        assert query["A"] == ["1"], f"added parameter lost: {built!r}"

    def test_no_parameters_leaves_the_url_untouched(self):
        """An empty mapping returns the endpoint verbatim.

        Test scenario:
            The early return keeps a RESTful template free of a stray `?`.
        """
        assert _query("https://example.org/wms?", {}) == "https://example.org/wms?", (
            "an empty mapping should not rewrite the URL"
        )


class TestProvidersHandleAwkwardEndpoints:
    """Both providers survive endpoint shapes that broke naive concatenation."""

    def test_wmts_on_a_trailing_question_mark_endpoint(self):
        """A `.../service?` WMTS endpoint still sends `SERVICE`.

        Test scenario:
            The end-to-end form of the `_query` case: WMTS `GetTile` mandates
            `SERVICE`, and mangling it to `?SERVICE` is a hard failure.
        """
        provider = WMTSProvider(url="https://example.org/service?", layer="L")
        query = parse_qs(urlsplit(provider.build_url(x=1, y=2, z=3)).query)
        assert query["SERVICE"] == ["WMTS"], f"SERVICE mangled: {query}"
        assert "?SERVICE" not in query, f"a bogus '?SERVICE' key was produced: {query}"

    def test_wms_on_a_fragment_endpoint(self):
        """A fragment on the endpoint does not cost the GetMap its parameters.

        Test scenario:
            Every GetMap parameter has to reach the service; losing them all
            to the fragment would return the service's default image, or an
            exception, rather than the requested tile.
        """
        provider = WMSProvider(url="https://example.org/wms#layers", layers="ortho")
        query = parse_qs(urlsplit(provider.build_url(x=0, y=0, z=0)).query)
        assert query["REQUEST"] == ["GetMap"], (
            f"parameters lost to the fragment: {query}"
        )


class TestWMTSProviderIsRestful:
    """`WMTSProvider.is_restful` decides which encoding `build_url` uses."""

    @pytest.mark.parametrize(
        "url, expected",
        [
            ("https://example.org/wmts", False),
            ("https://example.org/{TileMatrix}/{TileRow}/{TileCol}.png", True),
            ("https://example.org/wmts?layer=x", False),
        ],
    )
    def test_marker_decides(self, url, expected):
        """`{TileMatrix}` in the URL marks it RESTful.

        Args:
            url: The endpoint or template under test.
            expected: Whether it should be read as RESTful.

        Test scenario:
            One marker decides the branch, so it is worth pinning directly
            rather than only through `build_url`'s output.
        """
        provider = WMTSProvider(url=url, layer="L")
        assert provider.is_restful is expected, f"{url!r} classified wrongly"


class TestRestfulTemplateSubstitution:
    """The RESTful branch substitutes once, case-insensitively, and escapes."""

    def test_a_template_missing_row_or_col_is_refused(self):
        """A template that cannot address a tile is rejected at construction.

        Test scenario:
            With only `{TileMatrix}` every tile resolves to the same URL, so the
            mosaic is one image repeated -- and silently, since each request
            succeeds. Refusing the template is the only point at which this is
            visible.
        """
        with pytest.raises(ValueError, match="must address a tile"):
            WMTSProvider(url="https://example.org/wmts/{TileMatrix}.png", layer="L")

    @pytest.mark.parametrize(
        "template",
        [
            "https://example.org/w/{tilematrix}/{tilerow}/{tilecol}.png",
            "https://example.org/w/{TILEMATRIX}/{TILEROW}/{TILECOL}.png",
            "https://example.org/w/{TileMatrix}/{TileRow}/{TileCol}.png",
        ],
    )
    def test_placeholder_casing_does_not_matter(self, template):
        """Any casing of the three placeholders substitutes.

        Args:
            template: The template under test.

        Test scenario:
            Services are inconsistent about the spelling. A mis-cased
            placeholder used to fail the RESTful test entirely, so the KVP
            branch ran and shipped a URL with literal braces still in it.
        """
        provider = WMTSProvider(url=template, layer="L")
        assert provider.build_url(x=4, y=2, z=3) == "https://example.org/w/3/2/4.png", (
            f"{template} did not substitute: {provider.build_url(x=4, y=2, z=3)}"
        )

    def test_substituted_values_are_percent_encoded(self):
        """A layer name with URL-significant characters cannot break the path.

        Test scenario:
            A raw `/` would invent a path segment and a raw `?` would start a
            query, so an unescaped layer name silently requests something else.
        """
        provider = WMTSProvider(
            url="https://example.org/w/{Layer}/{TileMatrix}/{TileRow}/{TileCol}.png",
            layer="a b/c?d",
        )
        assert provider.build_url(x=4, y=2, z=3) == (
            "https://example.org/w/a%20b%2Fc%3Fd/3/2/4.png"
        ), f"not escaped: {provider.build_url(x=4, y=2, z=3)}"

    def test_a_placeholder_shaped_value_is_not_re_substituted(self):
        """A field whose value looks like a placeholder is left as data.

        Test scenario:
            The old chain of `str.replace` calls re-scanned what it had already
            written, so a layer literally named `{TileRow}` was rewritten by the
            next step into the row number.
        """
        provider = WMTSProvider(
            url="https://example.org/{Layer}/{TileMatrix}/{TileRow}/{TileCol}.png",
            layer="{TileRow}",
        )
        built = provider.build_url(x=4, y=2, z=3)
        assert built == "https://example.org/%7BTileRow%7D/3/2/4.png", (
            f"a placeholder-shaped value was re-substituted: {built}"
        )

    def test_an_unknown_placeholder_is_left_alone(self):
        """A placeholder this package does not own is not touched.

        Test scenario:
            A service template may carry its own; silently deleting or
            mangling it would be worse than leaving it for the caller to see.
        """
        provider = WMTSProvider(
            url="https://example.org/{Custom}/{TileMatrix}/{TileRow}/{TileCol}.png",
            layer="L",
        )
        assert "{Custom}" in provider.build_url(x=4, y=2, z=3), (
            "an unknown placeholder was altered"
        )

    def test_each_tile_gets_its_own_url(self):
        """Distinct tiles produce distinct URLs.

        Test scenario:
            The property the missing-placeholder check exists to protect: a
            mosaic of N tiles must make N different requests.
        """
        provider = WMTSProvider(
            url="https://example.org/w/{TileMatrix}/{TileRow}/{TileCol}.png", layer="L"
        )
        built = {provider.build_url(x=i, y=i, z=3) for i in range(4)}
        assert len(built) == 4, f"tiles collapsed to {len(built)} URL(s): {built}"


class TestWMTSProviderValidation:
    """`WMTSProvider` refuses an unusable service description at construction."""

    @pytest.mark.parametrize(
        "bad", ["file:///tiles/wmts", "ftp://example.org", "", "  "]
    )
    def test_a_non_http_endpoint_raises(self, bad):
        """Only http(s) endpoints are accepted.

        Args:
            bad: The rejected URL.

        Test scenario:
            `fetch_single_tile` would reject these too, but only per tile and
            deep inside a render; failing here names the field.
        """
        with pytest.raises(ValueError, match="url must be"):
            WMTSProvider(url=bad, layer="TrueColor")

    @pytest.mark.parametrize(
        "field_name, kwargs",
        [
            ("layer", {"layer": ""}),
            ("tile_matrix_set", {"layer": "L", "tile_matrix_set": ""}),
            ("style", {"layer": "L", "style": ""}),
            ("image_format", {"layer": "L", "image_format": ""}),
            ("version", {"layer": "L", "version": ""}),
        ],
    )
    def test_an_empty_identifier_raises(self, field_name, kwargs):
        """Each identifier must be a non-empty string.

        Args:
            field_name: The field expected in the message.
            kwargs: The constructor arguments that should be rejected.

        Test scenario:
            An empty layer produces a request the service answers with an XML
            exception, which the image sniffer then reports as a network
            failure -- a long way from the real cause.
        """
        with pytest.raises(ValueError, match=field_name):
            WMTSProvider(url="https://example.org/wmts", **kwargs)

    def test_non_mapping_extra_params_raises(self):
        """`extra_params` must be a mapping.

        Test scenario:
            A list of pairs is the plausible mistake, and it would otherwise
            fail much later inside `urlencode`.
        """
        with pytest.raises(TypeError, match="extra_params must be a mapping"):
            WMTSProvider(
                url="https://example.org/wmts",
                layer="L",
                extra_params=[("a", "b")],
            )

    @pytest.mark.parametrize(
        "field_name, value",
        [
            ("url", None),
            ("url", 123),
            ("layer", None),
            ("tile_matrix_set", 3857),
            ("style", None),
            ("image_format", 0),
            ("version", 1.0),
        ],
    )
    def test_a_non_string_field_raises(self, field_name, value):
        """Every validated field rejects a non-string as well as an empty one.

        Args:
            field_name: The field replaced with a non-string value.
            value: The rejected value.

        Test scenario:
            The empty-string table above never reaches the `isinstance` half of
            the guard, so line and branch coverage read the same with it and
            without it. Dropping it would cost the `ValueError` naming the
            field and buy an `AttributeError` from `.strip()` -- which is why
            these inputs are worth pinning even at 100% coverage.
        """
        kwargs = {"url": "https://example.org/wmts", "layer": "TrueColor"}
        kwargs[field_name] = value
        expected = f"{field_name} must be a non-empty string"
        with pytest.raises(ValueError, match=expected):
            WMTSProvider(**kwargs)


class TestWMSProviderBuildUrl:
    """`WMSProvider.build_url` turns a tile into a GetMap request."""

    @pytest.mark.parametrize("tile", [Tile(4, 2, 3), Tile(0, 0, 0), Tile(7, 5, 4)])
    def test_bbox_is_the_tile_bounds(self, wms, tile):
        """The `BBOX` is exactly the tile's Web Mercator bounds.

        Args:
            wms: The GetMap provider fixture.
            tile: The tile under test.

        Test scenario:
            This is the whole WMS adaptation: the mosaic only lines up if the
            requested image covers precisely the tile the stitcher will place
            it at. Compared against the pipeline's own helper, not a literal.
        """
        query = query_of(wms.build_url(x=tile.x, y=tile.y, z=tile.z))
        sent = tuple(float(v) for v in query["BBOX"][0].split(","))
        assert sent == _tile_xy_bounds(tile), (
            f"BBOX {sent} != tile bounds {_tile_xy_bounds(tile)}"
        )

    def test_size_is_the_tile_size(self, wms):
        """`WIDTH`/`HEIGHT` come from `tile_size`.

        Args:
            wms: The GetMap provider fixture.

        Test scenario:
            A GetMap has no tile index, so the pixel size is the only thing
            fixing the resolution of the returned image.
        """
        query = query_of(wms.build_url(x=0, y=0, z=0))
        assert query["WIDTH"] == ["256"], f"WIDTH wrong: {query}"
        assert query["HEIGHT"] == ["256"], f"HEIGHT wrong: {query}"

    def test_service_parameters_are_sent(self, wms):
        """The request carries the fixed GetMap parameters.

        Args:
            wms: The GetMap provider fixture.

        Test scenario:
            SERVICE/REQUEST/VERSION/LAYERS/STYLES are all mandatory; a missing
            one is a service-side exception rather than an image.
        """
        query = query_of(wms.build_url(x=0, y=0, z=0))
        assert query["SERVICE"] == ["WMS"], f"SERVICE wrong: {query}"
        assert query["REQUEST"] == ["GetMap"], f"REQUEST wrong: {query}"
        assert query["LAYERS"] == ["ortho"], f"LAYERS wrong: {query}"
        assert query["FORMAT"] == ["image/png"], f"FORMAT wrong: {query}"
        assert query["TRANSPARENT"] == ["TRUE"], f"TRANSPARENT wrong: {query}"

    @pytest.mark.parametrize(
        "version, expected_key, absent_key",
        [
            ("1.3.0", "CRS", "SRS"),
            ("1.1.1", "SRS", "CRS"),
            ("1.1.0", "SRS", "CRS"),
            ("1.0.0", "SRS", "CRS"),
        ],
    )
    def test_crs_parameter_follows_the_version(self, version, expected_key, absent_key):
        """1.3.0 sends `CRS=`; older versions send `SRS=`.

        Args:
            version: The WMS version requested.
            expected_key: The query key that must be present.
            absent_key: The query key that must not be.

        Test scenario:
            Sending the wrong spelling makes the service ignore the CRS and
            answer in its own default, which silently misplaces every tile.
        """
        provider = WMSProvider(
            url="https://example.org/wms", layers="ortho", version=version
        )
        query = query_of(provider.build_url(x=0, y=0, z=0))
        assert query[expected_key] == ["EPSG:3857"], f"{expected_key} missing: {query}"
        assert absent_key not in query, f"{absent_key} should not be sent: {query}"

    def test_transparent_can_be_turned_off(self):
        """`transparent=False` sends `TRANSPARENT=FALSE`.

        Test scenario:
            A base layer wants an opaque image; only an overlay wants alpha.
        """
        provider = WMSProvider(
            url="https://example.org/wms", layers="ortho", transparent=False
        )
        assert query_of(provider.build_url(x=0, y=0, z=0))["TRANSPARENT"] == [
            "FALSE"
        ], "transparent=False was not honoured"

    def test_custom_tile_size_reaches_the_request(self):
        """A non-default `tile_size` sets both dimensions.

        Test scenario:
            Some services cap or require a particular size, so it is settable
            -- and both dimensions must move together or the mosaic skews.
        """
        provider = WMSProvider(
            url="https://example.org/wms", layers="ortho", tile_size=512
        )
        query = query_of(provider.build_url(x=0, y=0, z=0))
        assert query["WIDTH"] == ["512"] and query["HEIGHT"] == ["512"], (
            f"tile_size not applied: {query}"
        )

    def test_styles_is_sent_even_when_empty(self, wms):
        """The mandatory `STYLES` key is present, with an empty value by default.

        Args:
            wms: The GetMap provider fixture.

        Test scenario:
            WMS requires `STYLES`; empty means "the service's default", and
            omitting it altogether is a service exception. `parse_qs` drops
            blank values, so none of the sibling tests can see this key at all
            -- it has to be read off the raw query string.
        """
        url = wms.build_url(x=0, y=0, z=0)
        assert "STYLES=&" in url, f"STYLES is not sent when empty: {url}"
        assert "STYLES" in parse_qs(urlsplit(url).query, keep_blank_values=True), (
            f"STYLES missing from the query: {url}"
        )

    def test_non_default_styles_reach_the_request(self):
        """A comma-separated `styles` value is passed through verbatim.

        Test scenario:
            `styles` pairs positionally with `layers`, so a two-layer request
            needs two style names in the same order. The comma has to survive
            as a separator inside one parameter rather than being split into
            two, which would silently style the wrong layer.
        """
        provider = WMSProvider(
            url="https://example.org/wms", layers="ortho,roads", styles="raw,thin"
        )
        query = query_of(provider.build_url(x=0, y=0, z=0))
        assert query["STYLES"] == ["raw,thin"], f"styles not passed through: {query}"
        assert query["LAYERS"] == ["ortho,roads"], f"layers not passed through: {query}"

    @pytest.mark.parametrize(
        "tile", [Tile(419234, 280456, 19), Tile(524287, 524287, 19)]
    )
    def test_bbox_survives_the_float_to_string_round_trip(self, wms, tile):
        """At the deepest zoom the `BBOX` text still reproduces the bounds exactly.

        Args:
            wms: The GetMap provider fixture.
            tile: A zoom-19 tile, including the grid's far south-east corner.

        Test scenario:
            A zoom-19 tile is about 0.07 m across yet sits up to 2e7 m from the
            origin, so its bounds need every significant digit a float has.
            Formatting them with `%f`, or rounding them for tidiness, would
            pass every low-zoom case above and misplace the image only here.
        """
        query = query_of(wms.build_url(x=tile.x, y=tile.y, z=tile.z))
        sent = tuple(float(v) for v in query["BBOX"][0].split(","))
        assert sent == _tile_xy_bounds(tile), (
            f"BBOX {sent} != tile bounds {_tile_xy_bounds(tile)}"
        )


class TestBboxNotation:
    """A `BBOX` never reaches a service in scientific notation."""

    @pytest.mark.parametrize("z", [0, 1, 3, 10, 19])
    def test_no_tile_produces_an_exponent(self, wms, z):
        """Every tile at every zoom formats as plain decimal.

        Args:
            wms: The GetMap provider fixture.
            z: The zoom level under test.

        Test scenario:
            Tiles adjacent to the projection origin have bounds around 1e-10,
            which Python renders with an exponent by default. The WMS grammar
            does not ask for that, and services vary between rejecting it and
            misparsing it -- so a render centred on Greenwich and the equator
            broke for no visible reason. The corners and the middle of each
            level are checked, since the origin-adjacent tiles are the middle.
        """
        span = 2**z
        corners = [(0, 0), (span - 1, span - 1), (span // 2, span // 2)]
        for x, y in corners:
            bbox = query_of(wms.build_url(x=x, y=y, z=z))["BBOX"][0]
            assert "e" not in bbox.lower(), f"exponent at z={z} x={x} y={y}: {bbox}"

    def test_the_bbox_still_round_trips_to_the_tile_bounds(self, wms):
        """Formatting does not cost the coordinates their accuracy.

        Args:
            wms: The GetMap provider fixture.

        Test scenario:
            The deepest tile this package reaches is ~0.075 m across, sitting
            2e7 m from the origin, so a careless format would place it in the
            wrong tile entirely. Six decimals is sub-micrometre.
        """
        tile = Tile(2**19 - 1, 2**19 - 1, 19)
        sent = [
            float(v)
            for v in query_of(wms.build_url(x=tile.x, y=tile.y, z=tile.z))["BBOX"][
                0
            ].split(",")
        ]
        expected = _tile_xy_bounds(tile)
        assert sent == pytest.approx(expected, abs=1e-6), (
            f"BBOX {sent} drifted from the tile bounds {expected}"
        )


class TestFormatCoordinate:
    """`_format_coordinate` renders one bound for the `BBOX`."""

    @pytest.mark.parametrize(
        "value, expected",
        [(0.0, "0"), (-0.0, "0"), (-20037508.0, "-20037508"), (1.5, "1.5")],
    )
    def test_tidy_values_render_tidily(self, value, expected):
        """A whole or short value does not gain noise.

        Args:
            value: The coordinate to render.
            expected: Its expected text form.

        Test scenario:
            Negative zero is in the table because `-0` is a legal result of the
            trimming but a strange thing to put in a `BBOX`.
        """
        assert _format_coordinate(value) == expected, (
            f"{value!r} rendered as {_format_coordinate(value)!r}"
        )

    @pytest.mark.parametrize(
        "value",
        [-5.529727786779404e-10, 5009377.085697311, -20037508.342789244, 1e-7, 1e17],
    )
    def test_every_value_is_exponent_free_and_lossless(self, value):
        """The rendering neither uses an exponent nor loses a bit.

        Args:
            value: The coordinate to render.

        Test scenario:
            These are the two properties in tension: rounding to fixed decimals
            kills the exponent but costs the exact round trip the mosaic
            alignment rests on, and plain `str()` keeps the value but emits an
            exponent near the origin. Both must hold at once.
        """
        rendered = _format_coordinate(value)
        assert "e" not in rendered.lower(), f"{value!r} rendered as {rendered!r}"
        assert float(rendered) == value, (
            f"{rendered!r} does not parse back to {value!r}"
        )


class TestWMSProviderCrsParameter:
    """`WMSProvider.crs_parameter` names the version's CRS query key."""

    @pytest.mark.parametrize(
        "version, expected", [("1.3.0", "CRS"), ("1.1.1", "SRS"), ("1.0.0", "SRS")]
    )
    def test_key_matches_the_version(self, version, expected):
        """The property agrees with what `build_url` sends.

        Args:
            version: The WMS version.
            expected: The query key it should use.

        Test scenario:
            The property is the readable form of the rule; pinning it directly
            documents the 1.3.0 rename.
        """
        provider = WMSProvider(
            url="https://example.org/wms", layers="ortho", version=version
        )
        assert provider.crs_parameter == expected, f"{version} -> {expected} expected"


class TestWMSProviderValidation:
    """`WMSProvider` refuses an unusable service description at construction."""

    @pytest.mark.parametrize("bad", ["file:///wms", "ftp://example.org", "", "  "])
    def test_a_non_http_endpoint_raises(self, bad):
        """Only http(s) endpoints are accepted.

        Args:
            bad: The rejected URL.

        Test scenario:
            Same reasoning as the WMTS case -- fail where the field is named.
        """
        with pytest.raises(ValueError, match="url must be"):
            WMSProvider(url=bad, layers="ortho")

    def test_an_empty_layers_raises(self):
        """`layers` must be a non-empty string.

        Test scenario:
            A GetMap with no LAYERS is a service exception, which arrives here
            as an unreadable image and reads as a network failure.
        """
        with pytest.raises(ValueError, match="layers"):
            WMSProvider(url="https://example.org/wms", layers="")

    def test_an_unsupported_version_raises(self):
        """A version whose CRS spelling is unknown is refused.

        Test scenario:
            Guessing between `CRS=` and `SRS=` for an unknown version would
            misplace every tile silently; the supported set is small and
            explicit instead.
        """
        with pytest.raises(ValueError, match="version must be one of"):
            WMSProvider(url="https://example.org/wms", layers="ortho", version="2.0.0")

    @pytest.mark.parametrize("bad", [0, -1, 2.5, "256", True])
    def test_a_bad_tile_size_raises(self, bad):
        """`tile_size` must be a positive int.

        Args:
            bad: The rejected size.

        Test scenario:
            `True` is in the table deliberately: it is an `int` subclass, so a
            bare `isinstance` check would accept it as a 1-pixel request.
        """
        with pytest.raises(ValueError, match="tile_size must be a positive int"):
            WMSProvider(url="https://example.org/wms", layers="ortho", tile_size=bad)

    def test_non_mapping_extra_params_raises(self):
        """`extra_params` must be a mapping.

        Test scenario:
            Same trap as the WMTS case.
        """
        with pytest.raises(TypeError, match="extra_params must be a mapping"):
            WMSProvider(
                url="https://example.org/wms", layers="ortho", extra_params=[("a", "b")]
            )

    @pytest.mark.parametrize(
        "field_name, value",
        [
            ("url", None),
            ("url", 123),
            ("layers", None),
            ("image_format", 0),
            ("version", 1.3),
        ],
    )
    def test_a_non_string_field_raises(self, field_name, value):
        """Every validated field rejects a non-string as well as an empty one.

        Args:
            field_name: The field replaced with a non-string value.
            value: The rejected value.

        Test scenario:
            The same blind spot as the WMTS case -- the empty-string tests
            exercise only the `.strip()` half of the guard. `version=1.3` is
            the plausible slip here: a float literal looks like a version, and
            without the type check it would reach the supported-versions
            comparison as a non-string and be reported as an unknown version.
        """
        kwargs = {"url": "https://example.org/wms", "layers": "ortho"}
        kwargs[field_name] = value
        expected = f"{field_name} must be a non-empty string"
        with pytest.raises(ValueError, match=expected):
            WMSProvider(**kwargs)


class TestOptionalFieldsAreStillTyped:
    """A field that may be empty is still required to be the right type."""

    @pytest.mark.parametrize(
        "kwargs, field_name",
        [
            ({"styles": None}, "styles"),
            ({"styles": 5}, "styles"),
            ({"attribution": None}, "attribution"),
        ],
    )
    def test_a_non_string_optional_field_raises(self, kwargs, field_name):
        """`styles` and `attribution` accept empty, not `None`.

        Args:
            kwargs: The constructor argument that should be rejected.
            field_name: The field expected in the message.

        Test scenario:
            Both may legitimately be empty, so they escaped the non-empty check
            and with it any type check -- `styles=None` was sent to the service
            as the four characters `None`, and came back as a service exception
            disguised as an unreadable tile.
        """
        with pytest.raises(ValueError, match=field_name):
            WMSProvider(url="https://example.org/wms", layers="ortho", **kwargs)

    def test_empty_is_still_accepted(self):
        """The empty string remains valid for both.

        Test scenario:
            Tightening the type must not narrow the contract: an empty
            `styles` means "the service's default", which is the common case.
        """
        provider = WMSProvider(
            url="https://example.org/wms", layers="ortho", styles="", attribution=""
        )
        # `parse_qs` drops blank values unless asked to keep them, so an empty
        # STYLES is invisible to `query_of` -- which is how it went untested.
        query = parse_qs(
            urlsplit(provider.build_url(x=0, y=0, z=0)).query, keep_blank_values=True
        )
        assert query["STYLES"] == [""], f"an empty STYLES was not sent: {query}"

    @pytest.mark.parametrize("bad", ["no", 1, 0, None])
    def test_a_non_bool_transparent_raises(self, bad):
        """`transparent` must be an actual bool.

        Args:
            bad: The rejected value.

        Test scenario:
            Any truthy value would send `TRANSPARENT=TRUE`, so `"no"` would
            have meant its own opposite.
        """
        with pytest.raises(ValueError, match="transparent must be a bool"):
            WMSProvider(url="https://example.org/wms", layers="ortho", transparent=bad)

    @pytest.mark.parametrize(
        "padded", [" https://example.org/wms", "https://example.org/wms "]
    )
    def test_a_whitespace_padded_url_raises(self, padded):
        """A padded endpoint is refused rather than quietly trimmed.

        Args:
            padded: The rejected URL.

        Test scenario:
            The emptiness check used `.strip()` but the value was stored and
            sent verbatim, so the padding reached the request. Trimming it
            silently would hide a copy-paste error the caller should see.
        """
        with pytest.raises(ValueError, match="whitespace"):
            WMSProvider(url=padded, layers="ortho")

    def test_an_unsupported_wmts_version_raises(self):
        """`WMTSProvider.version` is validated like the WMS one.

        Test scenario:
            One provider refusing an unknown version while the other accepted
            any non-empty string was an inconsistency a caller would meet only
            by accident. OGC has only ever published WMTS 1.0.0.
        """
        with pytest.raises(ValueError, match="version must be one of"):
            WMTSProvider(url="https://example.org/wmts", layer="L", version="2.0.0")


class TestProvidersAreImmutable:
    """Both dataclasses are frozen, including the mapping they hold."""

    @pytest.mark.parametrize("kind", ["wmts", "wms"])
    def test_attributes_cannot_be_reassigned(self, kind, wmts, wms):
        """A frozen dataclass rejects assignment.

        Args:
            kind: Which provider to check.
            wmts: The WMTS fixture.
            wms: The WMS fixture.

        Test scenario:
            The provider is shared across every tile of a render, so a mutation
            part-way through would produce a mosaic from two services.
        """
        provider = wmts if kind == "wmts" else wms
        with pytest.raises(AttributeError):
            provider.url = "https://elsewhere.invalid/"

    @pytest.mark.parametrize("kind", ["wmts", "wms"])
    def test_a_provider_is_hashable(self, kind, wmts, wms):
        """A frozen provider can be used as a dict key or a set member.

        Args:
            kind: Which provider to check.
            wmts: The WMTS fixture.
            wms: The WMS fixture.

        Test scenario:
            `frozen=True` generates a `__hash__`, but it hashes the field tuple
            -- and `extra_params` is a mapping. Without the flattening every
            provider raised `TypeError: unhashable type: 'dict'`, even with the
            default empty mapping, while still advertising itself as frozen.
        """
        provider = wmts if kind == "wmts" else wms
        assert isinstance(hash(provider), int), "provider is not hashable"
        assert len({provider, provider}) == 1, "provider does not de-duplicate"

    def test_equal_providers_hash_equal(self):
        """Two separately built, equal providers share a hash.

        Test scenario:
            The hash has to agree with the generated `__eq__`, which compares
            the same fields by value -- including `extra_params`, which is a
            `MappingProxyType` over a fresh copy in each instance.
        """
        first = WMSProvider(
            url="https://example.org/wms", layers="ortho", extra_params={"t": "1"}
        )
        second = WMSProvider(
            url="https://example.org/wms", layers="ortho", extra_params={"t": "1"}
        )
        assert first == second, "precondition: the two providers compare equal"
        assert hash(first) == hash(second), "equal providers hashed differently"

    def test_extra_params_participate_in_the_hash(self):
        """Two providers differing only in `extra_params` do not collide.

        Test scenario:
            Skipping the mapping entirely would be the easy way to make the
            hash work, and would silently merge a keyed provider with an
            unkeyed one in a cache.
        """
        plain = WMSProvider(url="https://example.org/wms", layers="ortho")
        keyed = WMSProvider(
            url="https://example.org/wms", layers="ortho", extra_params={"t": "1"}
        )
        assert plain != keyed, "precondition: the two providers differ"
        assert hash(plain) != hash(keyed), "extra_params was left out of the hash"

    def test_mutating_the_callers_dict_does_not_change_the_provider(self):
        """`extra_params` is copied, not aliased.

        Test scenario:
            `frozen=True` blocks assignment but not mutation of a held dict, so
            without the copy a caller's later edit would rewrite the URLs of an
            already-built provider.
        """
        params = {"token": "first"}
        provider = WMSProvider(
            url="https://example.org/wms", layers="ortho", extra_params=params
        )
        params["token"] = "second"
        assert query_of(provider.build_url(x=0, y=0, z=0))["token"] == ["first"], (
            "the provider aliased the caller's dict"
        )

    def test_the_stored_mapping_is_read_only(self, wms):
        """The stored `extra_params` itself refuses writes.

        Args:
            wms: The GetMap provider fixture.

        Test scenario:
            The other half of the copy: reaching into the attribute must fail
            too, not silently succeed on a private copy.
        """
        with pytest.raises(TypeError):
            wms.extra_params["token"] = "sneaked in"

    def test_two_identically_configured_providers_are_equal(self):
        """Separately built providers with the same fields compare equal.

        Test scenario:
            `__post_init__` swaps `extra_params` for a `MappingProxyType`, so
            `==` runs through that view rather than the dict the caller passed.
            A view comparing by identity would make every such comparison false
            -- including the ones a caching or de-duplicating layer relies on.
        """
        first = WMTSProvider(
            url="https://example.org/wmts", layer="L", extra_params={"k": "v"}
        )
        second = WMTSProvider(
            url="https://example.org/wmts", layer="L", extra_params={"k": "v"}
        )
        assert first == second, "identically configured providers compared unequal"
        assert first != WMTSProvider(url="https://example.org/wmts", layer="Other"), (
            "providers differing in a field compared equal"
        )

    @pytest.mark.parametrize("kind", ["wmts", "wms"])
    def test_replace_revalidates_the_copy(self, kind, wmts, wms):
        """`dataclasses.replace` re-runs the validation on the new instance.

        Args:
            kind: Which provider to check.
            wmts: The WMTS fixture.
            wms: The WMS fixture.

        Test scenario:
            `replace` is the only supported way to vary a frozen provider, and
            it is where a bad value would plausibly slip past: the copy is
            assembled field by field rather than through the caller's own
            constructor call.
        """
        provider = wmts if kind == "wmts" else wms
        with pytest.raises(ValueError, match="url must be an http"):
            dataclasses.replace(provider, url="file:///tiles")

    def test_replace_refreezes_the_copied_params(self):
        """A replaced copy holds its own read-only `extra_params`.

        Test scenario:
            `replace` feeds the original's `MappingProxyType` straight back in.
            That is a `Mapping`, so it passes the type check unchanged -- the
            copy must still end up behind its own frozen view rather than an
            alias a write could reach the original through.
        """
        original = WMSProvider(
            url="https://example.org/wms", layers="ortho", extra_params={"token": "t"}
        )
        copy = dataclasses.replace(original, layers="roads")
        assert dict(copy.extra_params) == {"token": "t"}, (
            f"extra_params lost in the copy: {dict(copy.extra_params)}"
        )
        with pytest.raises(TypeError):
            copy.extra_params["token"] = "sneaked in"


@requires_tiles
class TestProvidersDriveFetchSingleTile:
    """Both providers satisfy the contract `fetch_single_tile` calls."""

    @pytest.mark.parametrize("kind", ["wmts", "wms"])
    def test_a_tile_is_fetched_from_the_built_url(self, kind, wmts, wms, recorded_urls):
        """The bytes come back and the URL is the provider's own.

        Args:
            kind: Which provider to check.
            wmts: The WMTS fixture.
            wms: The WMS fixture.
            recorded_urls: The patched opener's URL log.

        Test scenario:
            This is the one seam the whole design rests on; asserting the
            request URL proves `build_url` was what produced it.
        """
        provider = wmts if kind == "wmts" else wms
        tile, data = fetch_single_tile(Tile(4, 2, 3), provider, timeout=1, retries=0)

        assert data == ONE_PIXEL_PNG, "the fetched bytes were not passed through"
        assert tile == Tile(4, 2, 3), f"the tile was not echoed back: {tile}"
        assert recorded_urls == [provider.build_url(x=4, y=2, z=3)], (
            f"requested {recorded_urls}, not the provider's URL"
        )


@requires_tiles
class TestProvidersDriveAddTiles:
    """Both providers render through the unchanged public entry point."""

    @pytest.mark.parametrize("kind", ["wmts", "wms"])
    def test_a_basemap_is_drawn(self, kind, wmts, wms, recorded_urls):
        """`add_tiles` composes a mosaic from the provider's tiles.

        Args:
            kind: Which provider to check.
            wmts: The WMTS fixture.
            wms: The WMS fixture.
            recorded_urls: The patched opener's URL log.

        Test scenario:
            The claim the module exists to make good on: an OGC service needs
            no pipeline change, only a different `build_url`. Nothing in
            `tiles.py` is patched here beyond the HTTP opener.
        """
        provider = wmts if kind == "wmts" else wms
        fig, ax = plt.subplots()
        ax.set_xlim(1_000_000.0, 1_200_000.0)
        ax.set_ylim(6_000_000.0, 6_200_000.0)

        add_tiles(ax, provider, crs=3857, zoom=3)

        assert len(ax.images) == 1, f"no mosaic was drawn: {ax.images}"
        assert recorded_urls, "no tile was requested"
        plt.close(fig)

    @pytest.mark.parametrize(
        "kind, marker",
        [("wmts", "REQUEST=GetTile"), ("wms", "REQUEST=GetMap")],
    )
    def test_the_requests_are_ogc_requests(
        self, kind, marker, wmts, wms, recorded_urls
    ):
        """Every request the pipeline makes is the OGC form.

        Args:
            kind: Which provider to check.
            marker: The query fragment that identifies the service kind.
            wmts: The WMTS fixture.
            wms: The WMS fixture.
            recorded_urls: The patched opener's URL log.

        Test scenario:
            Drawing an image is not enough on its own -- it would also hold if
            the renderer had silently fallen back to a default XYZ provider.
        """
        provider = wmts if kind == "wmts" else wms
        fig, ax = plt.subplots()
        ax.set_xlim(1_000_000.0, 1_200_000.0)
        ax.set_ylim(6_000_000.0, 6_200_000.0)

        add_tiles(ax, provider, crs=3857, zoom=3)

        assert all(marker in url for url in recorded_urls), (
            f"not every request was a {marker}: {recorded_urls[:2]}"
        )
        plt.close(fig)

    @pytest.mark.parametrize("kind", ["wmts", "wms"])
    def test_the_attribution_is_drawn(self, kind, wmts, wms, recorded_urls):
        """`attribution=True` picks up the dataclass field.

        Args:
            kind: Which provider to check.
            wmts: The WMTS fixture.
            wms: The WMS fixture.
            recorded_urls: The patched opener's URL log.

        Test scenario:
            `add_tiles` reads `provider.attribution` defensively with `getattr`,
            so the credit line comes for free -- but only if the field is
            actually named what the renderer looks for.
        """
        provider = wmts if kind == "wmts" else wms
        fig, ax = plt.subplots()
        ax.set_xlim(1_000_000.0, 1_200_000.0)
        ax.set_ylim(6_000_000.0, 6_200_000.0)

        add_tiles(ax, provider, crs=3857, zoom=3, attribution=True)

        drawn = [text.get_text() for text in ax.texts]
        assert drawn == [provider.attribution], f"attribution not drawn: {drawn}"
        plt.close(fig)

    def test_a_restful_template_renders_through_the_pipeline(self, recorded_urls):
        """A path-substituted WMTS template is what the pipeline actually fetches.

        Args:
            recorded_urls: The patched opener's URL log.

        Test scenario:
            The RESTful branch is the one whose URL is mostly path rather than
            query, so it is the shape most likely to come apart downstream --
            `fetch_single_tile` re-checks the scheme of whatever `build_url`
            handed back. Every request must be a fully substituted path with
            the token still attached, and no placeholder left behind.
        """
        provider = WMTSProvider(
            url="https://example.org/wmts/{Layer}/{TileMatrix}/{TileRow}/{TileCol}.png",
            layer="TrueColor",
            extra_params={"token": "abc"},
        )
        fig, ax = plt.subplots()
        ax.set_xlim(1_000_000.0, 1_200_000.0)
        ax.set_ylim(6_000_000.0, 6_200_000.0)

        add_tiles(ax, provider, crs=3857, zoom=8)

        assert recorded_urls, "no tile was requested"
        assert all("{" not in url for url in recorded_urls), (
            f"a placeholder survived into a request: {sorted(recorded_urls)[:2]}"
        )
        assert all(
            url.startswith("https://example.org/wmts/TrueColor/8/")
            and url.endswith(".png?token=abc")
            for url in recorded_urls
        ), f"not every request was a substituted template: {sorted(recorded_urls)[:2]}"
        plt.close(fig)

    def test_an_endpoint_query_survives_the_pipeline(self, recorded_urls):
        """A `GetMap` endpoint's own query reaches every request unharmed.

        Args:
            recorded_urls: The patched opener's URL log.

        Test scenario:
            MapServer publishes its endpoint with a mandatory `map=` parameter
            baked in. The unit test above proves `build_url` keeps it; this
            proves nothing between `build_url` and the opener re-encodes or
            drops it -- and it checks every tile of the mosaic, not just one.
        """
        provider = WMSProvider(
            url="https://example.org/wms?map=/etc/base.map", layers="ortho"
        )
        fig, ax = plt.subplots()
        ax.set_xlim(1_000_000.0, 1_200_000.0)
        ax.set_ylim(6_000_000.0, 6_200_000.0)

        add_tiles(ax, provider, crs=3857, zoom=8)

        assert len(recorded_urls) > 1, f"expected a tile grid: {recorded_urls}"
        assert all(
            query_of(url).get("map") == ["/etc/base.map"] for url in recorded_urls
        ), f"the endpoint's own parameter was lost: {sorted(recorded_urls)[:2]}"
        plt.close(fig)

    def test_min_tiles_across_one_lowers_the_request_count(self, wms, recorded_urls):
        """`min_tiles_across=1` asks a WMS for fewer, coarser `GetMap` images.

        Args:
            wms: The GetMap provider fixture.
            recorded_urls: The patched opener's URL log.

        Test scenario:
            The class docstring recommends this as the efficient shape for a
            WMS, which only holds if the knob reaches `auto_zoom` and lowers
            the zoom. Both renders leave `zoom` at its `"auto"` default, since
            an explicit zoom bypasses the floor entirely. On this extent the
            counts are 16 and 4; the assertion is on the relation rather than
            the literals, which belong to the zoom heuristic in `tiles`.
        """
        fig, ax = plt.subplots()
        ax.set_xlim(1_000_000.0, 1_200_000.0)
        ax.set_ylim(6_000_000.0, 6_200_000.0)
        add_tiles(ax, wms, crs=3857)
        default_count = len(recorded_urls)

        add_tiles(ax, wms, crs=3857, min_tiles_across=1)
        floored_count = len(recorded_urls) - default_count

        assert default_count > 1, f"the default floor fetched {default_count} tiles"
        assert 0 < floored_count < default_count, (
            f"min_tiles_across=1 requested {floored_count} images, "
            f"not fewer than the default {default_count}"
        )
        plt.close(fig)

    def test_the_mosaic_cell_size_follows_the_requested_tile_size(
        self, recorded_urls_4px
    ):
        """A non-default `tile_size` sizes the mosaic, not just the query string.

        Args:
            recorded_urls_4px: The patched opener's URL log, serving 4x4 PNGs.

        Test scenario:
            `stitch_tiles` reads the cell size off the first decoded image, not
            off the provider, so `tile_size` only works end to end because the
            service honours the `WIDTH` it was asked for. This renders a 2x2
            grid of 4-pixel tiles and requires an 8x8 mosaic: a single-tile
            zoom would give a square image whatever the grid arithmetic did.
        """
        provider = WMSProvider(
            url="https://example.org/wms", layers="ortho", tile_size=4
        )
        fig, ax = plt.subplots()
        ax.set_xlim(1_000_000.0, 1_200_000.0)
        ax.set_ylim(6_000_000.0, 6_200_000.0)

        add_tiles(ax, provider, crs=3857, zoom=8)

        requested = {query_of(url)["WIDTH"][0] for url in recorded_urls_4px}
        assert requested == {"4"}, f"WIDTH was not the tile_size: {requested}"
        assert len(recorded_urls_4px) == 4, (
            f"expected a 2x2 grid, got {len(recorded_urls_4px)} tiles"
        )
        assert ax.images[0].get_array().shape[:2] == (8, 8), (
            f"mosaic is not 2x2 cells of 4 px: {ax.images[0].get_array().shape}"
        )
        plt.close(fig)
