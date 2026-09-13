"""Tests for `cleopatra.basemap.ogc` -- WMS and WMTS addressed as tile providers.

URL construction is pure string work and needs nothing installed. The two
end-to-end classes drive the real `add_tiles` pipeline with only the HTTP layer
mocked, which is what proves `build_url` is actually the seam the renderer uses
-- patching `fetch_tiles` instead (as `test_tiles.py` does for other cases)
would never call the provider at all.
"""

from __future__ import annotations

import base64
import copy
import dataclasses
import logging
import pickle
from dataclasses import dataclass, field, replace
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
    _restful_placeholders,
)
from cleopatra.basemap.tiles import (
    _TILES_AVAILABLE,
    Tile,
    _redact_url,
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

    def test_an_override_does_not_reach_a_parameter_baked_into_the_endpoint(self):
        """The case-insensitive override covers generated parameters only.

        Test scenario:
            `extra_params` is documented as winning over a generated parameter,
            and it does -- but an endpoint publishing `FORMAT` in its own query
            sits outside that reconciliation, because `_query` keeps the
            endpoint's query verbatim. The request then carries the key twice
            with different values and the service picks between them, so this
            pins which of the two shapes a caller actually gets rather than
            leaving it to be discovered against a live server.
        """
        provider = WMSProvider(
            url="https://example.org/wms?FORMAT=image/jpeg",
            layers="ortho",
            extra_params={"format": "image/gif"},
        )
        query = query_of(provider.build_url(x=0, y=0, z=0))
        assert query["FORMAT"] == ["image/jpeg"], (
            f"the endpoint's own FORMAT was dropped or joined by the generated one: {query}"
        )
        assert query["format"] == ["image/gif"], (
            f"extra_params did not reach the query alongside it: {query}"
        )


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

    def test_a_key_the_endpoint_already_carries_is_appended_beside_it(self):
        """A parameter baked into the endpoint is kept, not replaced.

        Test scenario:
            `_query` reassembles the URL, so it sees the endpoint's query only
            as opaque text -- `_merge_params` reconciles the generated
            parameters against `extra_params`, never against the endpoint's
            own. Sending the same key twice is therefore the deliberate
            outcome, and the order (endpoint first, added second) is what
            settles it on a service that takes the last occurrence.
        """
        built = _query("https://example.org/wms?SERVICE=WMS", {"SERVICE": "WMTS"})
        assert built == "https://example.org/wms?SERVICE=WMS&SERVICE=WMTS", (
            f"the endpoint's own parameter was not kept ahead of the added one: {built!r}"
        )
        assert parse_qs(urlsplit(built).query)["SERVICE"] == ["WMS", "WMTS"], (
            f"both occurrences should reach the service, in order: {built!r}"
        )

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

    def test_placeholders_are_found_wherever_they_sit(self):
        """`_restful_placeholders` scans the whole URL, not only its path.

        Test scenario:
            Nothing in the pattern is anchored to a path segment, and some
            services publish the tile triple in the query instead. Unknown
            names are filtered out in the same pass, so a template carrying one
            of each yields only the known ones -- which is what stops a
            service's own `{Time}` from satisfying the required-placeholder
            check, or from dragging a KVP endpoint onto the RESTful branch.
        """
        in_query = _restful_placeholders(
            "https://example.org/wmts?tm={TileMatrix}&r={TileRow}&c={TileCol}"
        )
        assert in_query == {"tilematrix", "tilerow", "tilecol"}, (
            f"placeholders sitting in the query were not found: {in_query}"
        )
        mixed = _restful_placeholders(
            "https://example.org/{Custom}/{TileMatrixSet}/{TileMatrix}"
            "/{TileRow}/{TileCol}.png"
        )
        assert mixed == {"tilematrixset", "tilematrix", "tilerow", "tilecol"}, (
            f"an unknown placeholder leaked into the known set: {mixed}"
        )
        assert _restful_placeholders("https://example.org/{Time}.png") == set(), (
            "a URL carrying only unknown placeholders should read as KVP"
        )

    def test_a_template_with_its_placeholders_in_the_query_is_filled_in(self):
        """A query-borne template substitutes and still takes `extra_params`.

        Test scenario:
            The RESTful branch substitutes across the whole URL and only then
            hands the result to `_query`, so a template whose triple lives in
            the query must come back filled in *and* gain the token with an
            `&`. Substituting only the path would ship literal braces to the
            service; appending with a second `?` would make the tail garbage.
        """
        provider = WMTSProvider(
            url="https://example.org/wmts?tm={TileMatrix}&r={TileRow}&c={TileCol}",
            layer="L",
            extra_params={"token": "abc"},
        )
        built = provider.build_url(x=4, y=2, z=3)
        assert built == "https://example.org/wmts?tm=3&r=2&c=4&token=abc", (
            f"a query-borne template was not filled in correctly: {built}"
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
        # `span // 2 - 1` is the tile whose far edge sits *on* the projection
        # origin -- the residual that produced the exponent lives there, not at
        # the corners, so a grid that skips it would not have caught M1 at all.
        middle = max(span // 2 - 1, 0)
        corners = [
            (0, 0),
            (span - 1, span - 1),
            (span // 2, span // 2),
            (middle, middle),
            (middle, span // 2),
            (span // 2, middle),
        ]
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
        # Exact, not approximate: `_format_coordinate` renders through `Decimal`
        # precisely so the bound parses back bit-for-bit. A tolerance here would
        # accept the rounding implementation its sibling test rules out.
        assert sent == list(expected), (
            f"BBOX {sent} does not round-trip to the tile bounds {expected}"
        )

    def test_the_tile_on_the_projection_origin_sends_a_bare_zero(self, wms):
        """The zero bound reaches the service as `0`, not `0.0` or `-0`.

        Args:
            wms: The GetMap provider fixture.

        Test scenario:
            `_format_coordinate`'s zero handling is not hypothetical: the tile
            whose corner sits on the projection origin is a real tile of any
            render over Greenwich and the equator, and both of its zero bounds
            go through the trimming that can otherwise leave `-0` or an empty
            string in the `BBOX`.
        """
        bbox = query_of(wms.build_url(x=2**18, y=2**18, z=19))["BBOX"][0]
        left, _bottom, _right, top = bbox.split(",")
        assert (left, top) == ("0", "0"), (
            f"the origin bounds were not sent as a bare '0': {bbox}"
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

    @pytest.mark.parametrize("z", [0, 19])
    def test_the_bounds_the_grid_actually_produces_round_trip_exactly(self, z):
        """Every bound reachable at the grid's zoom limits survives verbatim.

        Args:
            z: The zoom level under test.

        Test scenario:
            `auto_zoom` clamps to 0--19, so these are the two ends of what a
            render can ask for, and the WMS adaptation rests on requesting
            precisely the bounds the mosaic will then place the image at. The
            assertion is exact equality rather than a tolerance on purpose: an
            implementation rounding to six decimals is exponent-free and lands
            within a micrometre of every bound, so it passes the round-trip
            check made through `build_url` while quietly dropping the guarantee
            that check exists to protect.
        """
        span = 2**z
        corners = sorted({0, span - 1, span // 2})
        for x in corners:
            for y in corners:
                for value in _tile_xy_bounds(Tile(x, y, z)):
                    rendered = _format_coordinate(value)
                    assert "e" not in rendered.lower(), (
                        f"exponent at z={z} x={x} y={y}: {rendered!r}"
                    )
                    assert float(rendered) == value, (
                        f"z={z} x={x} y={y}: {rendered!r} lost {value!r} exactly"
                    )


class TestWMSProviderCrsParameter:
    """`WMSProvider.crs_parameter` names the version's CRS query key."""

    @pytest.mark.parametrize(
        "version, expected", [("1.3.0", "CRS"), ("1.1.1", "SRS"), ("1.1.0", "SRS")]
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


class TestProvidersSurviveSerialisation:
    """A provider can be pickled and copied, as an immutable value should be."""

    @pytest.mark.parametrize("kind", ["wmts", "wms"])
    def test_pickle_round_trips(self, kind, wmts, wms):
        """`pickle` rebuilds an equal provider.

        Args:
            kind: Which provider to check.
            wmts: The WMTS fixture.
            wms: The WMS fixture.

        Test scenario:
            `extra_params` is stored as a `MappingProxyType`, which pickle
            cannot handle, so every provider raised -- while the class
            documented itself as immutable and cache-friendly. Anything that
            caches a render keyed on its provider would have hit this.
        """
        provider = wmts if kind == "wmts" else wms
        assert pickle.loads(pickle.dumps(provider)) == provider, (
            "the provider did not survive a pickle round trip"
        )

    def test_extra_params_survive_the_round_trip(self):
        """The mapping comes back with its contents and its read-only view.

        Test scenario:
            Rebuilding through the constructor re-runs `__post_init__`, so the
            copy must be as frozen as the original rather than holding a bare
            dict.
        """
        provider = WMSProvider(
            url="https://example.org/wms", layers="ortho", extra_params={"t": "1"}
        )
        rebuilt = pickle.loads(pickle.dumps(provider))
        assert dict(rebuilt.extra_params) == {"t": "1"}, "extra_params lost"
        with pytest.raises(TypeError, match="does not support item assignment"):
            rebuilt.extra_params["t"] = "2"

    @pytest.mark.parametrize("clone", [copy.copy, copy.deepcopy])
    def test_copying_works(self, clone, wms):
        """Both shallow and deep copies rebuild an equal provider.

        Args:
            clone: The copy function under test.
            wms: The GetMap provider fixture.

        Test scenario:
            `deepcopy` failed for the same reason pickling did, which would
            surprise anyone holding a provider inside a larger config object.
        """
        assert clone(wms) == wms, f"{clone.__name__} did not preserve the provider"

    def test_two_types_with_the_same_fields_do_not_collide(self):
        """The hash keys on the type itself, not its name.

        Test scenario:
            Keying on `type(...).__name__` would let two same-named classes
            from different modules collide in a dict.
        """
        wmts = WMTSProvider(url="https://example.org/x", layer="L")
        wms = WMSProvider(url="https://example.org/x", layers="L")
        assert wmts != wms, "precondition: the two providers are not equal"
        assert len({wmts, wms}) == 2, "two provider kinds collided in a set"


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


@dataclass(frozen=True)
class _ProviderWithDerivedField(WMSProvider):
    """A provider carrying a field the constructor does not take.

    Exists only so the `spec.init` filter in `_reduce_provider` can be proved
    rather than assumed -- no field on the real providers is `init=False`, so
    the guard is unreachable through them.
    """

    derived: str = field(default="computed", init=False)


class TestRoundTwoHardening:
    """The guards round 2 added, each pinned to the defect that prompted it."""

    def test_an_xyz_template_is_refused(self):
        """A `{z}/{x}/{y}` template is not silently sent with literal braces.

        Test scenario:
            Round 1 made placeholder matching case-insensitive but still fell
            through to KVP when *none* of the placeholders were ours -- so
            pasting an XYZ template in shipped `{z}` on the wire and 404ed
            every tile with no warning.
        """
        with pytest.raises(ValueError, match="placeholders this module does not fill"):
            WMTSProvider(url="https://example.org/{z}/{x}/{y}.png", layer="L")

    def test_an_unowned_placeholder_beside_the_required_ones_is_allowed(self):
        """A service's own placeholder is still tolerated alongside ours.

        Test scenario:
            The refusal must catch the "none of them are mine" case only; a
            template that does address a tile may carry extra names.
        """
        provider = WMTSProvider(
            url="https://e.org/{Time}/{TileMatrix}/{TileRow}/{TileCol}.png", layer="L"
        )
        assert "{Time}" in provider.build_url(x=4, y=2, z=3), (
            "an unowned placeholder beside the required three should survive"
        )

    @pytest.mark.parametrize("bad", ["https://e.org/w\tms", "https://e.org/w\nms"])
    def test_interior_whitespace_in_a_url_is_refused(self, bad):
        """A tab or newline inside the URL is rejected, not silently stripped.

        Args:
            bad: The rejected URL.

        Test scenario:
            The old check only looked at the ends. `urlsplit` strips interior
            whitespace, so the request differed from what the caller wrote --
            and differently again depending on whether `extra_params` was
            empty, since only then was the URL rebuilt.
        """
        with pytest.raises(ValueError, match="whitespace"):
            WMSProvider(url=bad, layers="ortho")

    @pytest.mark.parametrize("key", ["BBOX", "bbox", "WIDTH", "REQUEST", "TILEROW"])
    def test_extra_params_may_not_displace_a_tile_identifying_parameter(self, key):
        """Overriding what identifies the tile is refused.

        Args:
            key: The protected parameter the caller tried to set.

        Test scenario:
            The case-insensitive merge round 1 introduced made these reachable.
            Overriding `BBOX` detaches every image from the position the mosaic
            pastes it at, and since each request still succeeds the result is a
            plausible-looking picture made of the wrong tiles.
        """
        with pytest.raises(ValueError, match="identifies the tile"):
            WMSProvider(
                url="https://example.org/wms", layers="ortho", extra_params={key: "x"}
            )

    def test_two_extra_params_differing_only_by_case_are_refused(self):
        """`extra_params` may not name one parameter twice.

        Test scenario:
            Round 1 reconciled casing between the generated parameters and
            `extra_params`, but not *within* `extra_params` -- so both were
            sent and the service chose.
        """
        with pytest.raises(ValueError, match="name the same case-insensitive"):
            WMSProvider(
                url="https://example.org/wms",
                layers="ortho",
                extra_params={"format": "a", "FORMAT": "b"},
            )

    def test_a_tuneable_parameter_can_still_be_overridden(self):
        """The guard does not block the overrides that are the point.

        Test scenario:
            `FORMAT`, `STYLES` and a credential all remain settable; only the
            parameters that address the tile are protected.
        """
        provider = WMSProvider(
            url="https://example.org/wms",
            layers="ortho",
            extra_params={"format": "image/gif", "token": "t"},
        )
        query = query_of(provider.build_url(x=0, y=0, z=0))
        assert query["format"] == ["image/gif"], f"override lost: {query}"
        assert query["token"] == ["t"], f"credential lost: {query}"

    def test_a_non_init_field_does_not_break_pickling(self):
        """A field the constructor does not take is skipped when reducing.

        Test scenario:
            `_reduce_provider` rebuilds positionally through the constructor, so
            passing a non-init field along would raise `TypeError` on unpickle.
            No field on the real providers is `init=False` today; the guard is
            there so adding one cannot break pickling silently, and this is what
            proves the guard rather than assuming it.
        """
        provider = _ProviderWithDerivedField(
            url="https://example.org/wms", layers="ortho"
        )
        rebuilt = pickle.loads(pickle.dumps(provider))
        assert rebuilt == provider, "a non-init field broke the pickle round trip"
        assert rebuilt.derived == "computed", f"derived field lost: {rebuilt.derived!r}"

    def test_wms_1_0_0_is_not_claimed(self):
        """WMS 1.0.0 is refused rather than built with 1.1.x spellings.

        Test scenario:
            1.0.0 used `WMTVER`, `REQUEST=map` and `FORMAT=PNG`, none of which
            this module emits -- so accepting the version promised a request it
            never built.
        """
        with pytest.raises(ValueError, match="version must be one of"):
            WMSProvider(url="https://example.org/wms", layers="ortho", version="1.0.0")

    @pytest.mark.parametrize("kind", ["wmts", "wms"])
    def test_a_percent_encoded_space_is_still_accepted(self, kind):
        """`%20` is not whitespace, so an escaped path segment survives.

        Args:
            kind: Which provider to build.

        Test scenario:
            The whitespace rule was widened from the ends to any character, so
            it has to be read on the URL as written -- not on its decoded form.
            An endpoint whose path legitimately carries an escaped space is a
            normal published URL; refusing it would make the guard reject valid
            services while catching nothing extra.
        """
        url = "https://example.org/my%20wms"
        provider = (
            WMTSProvider(url=url, layer="L")
            if kind == "wmts"
            else WMSProvider(url=url, layers="ortho")
        )
        built = provider.build_url(x=0, y=0, z=0)
        assert built.startswith(f"{url}?"), (
            f"the escaped space did not survive: {built}"
        )

    @pytest.mark.parametrize("key", ["TileMatrix", "tilerow", "TILECOL", "SERVICE"])
    def test_a_protected_parameter_is_refused_on_the_wmts_provider_too(self, key):
        """The guard is wired into both constructors, not just `WMSProvider`.

        Args:
            key: The protected parameter the caller tried to set.

        Test scenario:
            `WMTSProvider` is where the tile triple is actually generated, so
            an override of `TILEROW` there is the version of this mistake that
            silently repaints the mosaic from the wrong tiles. Every other
            protected-parameter test builds a `WMSProvider`, so leaving the
            call out of this `__post_init__` would go unnoticed.
        """
        with pytest.raises(ValueError, match="identifies the tile"):
            WMTSProvider(
                url="https://example.org/wmts", layer="L", extra_params={key: "1"}
            )

    def test_a_non_string_key_is_coerced_before_it_is_validated(self):
        """Validation reads the frozen copy, so a non-`str` key does not blow up.

        Test scenario:
            The protected and duplicate checks both call `key.upper()`. Running
            them on the caller's raw mapping would raise `AttributeError` on an
            `int` key -- the very type `_freeze_params` exists to absorb, since
            a token or version id read out of JSON arrives that way. Ordering
            the freeze first is what keeps the two features compatible.
        """
        provider = WMSProvider(
            url="https://example.org/wms", layers="ortho", extra_params={5: "x"}
        )
        assert dict(provider.extra_params) == {"5": "x"}, (
            f"the key was not coerced: {dict(provider.extra_params)}"
        )
        assert query_of(provider.build_url(x=0, y=0, z=0))["5"] == ["x"], (
            "the coerced key did not reach the query"
        )

    def test_keys_that_coerce_to_one_string_collapse_rather_than_conflict(self):
        """`{5: ..., "5": ...}` is one parameter, so the duplicate guard is silent.

        Test scenario:
            The case-duplicate guard exists to stop two spellings of one OGC
            parameter both going on the wire. Coercion happens first and a dict
            cannot hold `5` and `"5"` at once, so the pair is already one entry
            by the time the guard looks -- last one wins, exactly as a repeated
            literal key would. Nothing ambiguous reaches the service, which is
            why this is a collapse and not a refusal.
        """
        provider = WMSProvider(
            url="https://example.org/wms",
            layers="ortho",
            extra_params={5: "x", "5": "y"},
        )
        assert dict(provider.extra_params) == {"5": "y"}, (
            f"expected the later value to win: {dict(provider.extra_params)}"
        )

    @pytest.mark.parametrize(
        "template",
        [
            "https://example.org/wmts/{Layer}.png",
            "https://example.org/wmts/{Style}/{TileMatrixSet}.png",
        ],
    )
    def test_a_template_of_only_optional_placeholders_is_refused(self, template):
        """Owning a placeholder is not the same as addressing a tile.

        Args:
            template: The template under test.

        Test scenario:
            `{Layer}`, `{Style}` and `{TileMatrixSet}` are ours, so the new
            "none of these are mine" refusal does not fire and the template
            reaches the required-placeholder check instead. That check has to
            name all three missing ones -- the caller who wrote `{Layer}` alone
            needs to be told what to add, not that the module does not fill it.
        """
        with pytest.raises(ValueError, match="must address a tile") as error:
            WMTSProvider(url=template, layer="L")
        message = str(error.value)
        assert "does not fill" not in message, (
            f"the wrong refusal fired for an owned placeholder: {message}"
        )
        for missing in ("{tilematrix}", "{tilerow}", "{tilecol}"):
            assert missing in message, f"{missing} not named in: {message}"


class TestCredentialsAreNotInTheRepr:
    """`repr()` does not carry what `extra_params` was documented to hold."""

    @pytest.mark.parametrize("kind", ["wmts", "wms"])
    def test_the_value_is_masked_and_the_name_kept(self, kind, wmts, wms):
        """A token survives as a name, not as a value.

        Args:
            kind: Which provider to check.
            wmts: The WMTS fixture.
            wms: The WMS fixture.

        Test scenario:
            The URL redaction cannot help here -- a traceback, a pytest failure
            line or `logger.info("using %s", provider)` prints the object long
            before any URL exists, and that is the sink hardest to control.
        """
        provider = replace(
            wmts if kind == "wmts" else wms, extra_params={"token": "S3CRET"}
        )
        assert "S3CRET" not in repr(provider), "the credential is in the repr"
        assert "'token': '...'" in repr(provider), (
            f"the parameter name should survive: {repr(provider)}"
        )

    def test_the_other_fields_are_still_shown(self, wms):
        """Masking the mapping does not blank the rest of the repr.

        Args:
            wms: The GetMap provider fixture.

        Test scenario:
            The repr still has to be useful for debugging; only the one field
            documented to hold a secret is masked.
        """
        rendered = repr(wms)
        assert "https://example.org/wms" in rendered, f"url missing: {rendered}"
        assert "ortho" in rendered, f"layers missing: {rendered}"

    def test_a_structured_value_cannot_smuggle_the_secret_out(self):
        """A nested container is flattened to a string and then masked like any other.

        Test scenario:
            A credential loaded from YAML or JSON often arrives nested, as
            `{"auth": {"token": ...}}`. Masking keyed on the value being a
            `str` would print such a value verbatim; masking every entry of the
            mapping regardless of what `_freeze_params` coerced it into is what
            closes that.
        """
        provider = WMSProvider(
            url="https://example.org/wms",
            layers="ortho",
            extra_params={"auth": {"token": "S3CRET"}},
        )
        rendered = repr(provider)
        assert "S3CRET" not in rendered, (
            f"the nested credential is in the repr: {rendered}"
        )
        assert "'auth': '...'" in rendered, (
            f"the parameter name should survive: {rendered}"
        )

    def test_the_object_is_safe_in_a_log_line(self, caplog):
        """`%s` formatting reaches `__repr__`, so the documented sink is covered.

        Args:
            caplog: pytest's log capture fixture.

        Test scenario:
            The masking is only worth anything if it is on the *default* way
            the object renders. `logger.info("using %s", provider)` is the sink
            the module names, and it goes through `__str__`, which a dataclass
            leaves pointing at `__repr__` -- so a masking helper callers had to
            opt into would leave this line leaking.
        """
        provider = WMSProvider(
            url="https://example.org/wms",
            layers="ortho",
            extra_params={"token": "S3CRET-IN-A-LOG"},
        )

        with caplog.at_level(logging.INFO, logger=__name__):
            logging.getLogger(__name__).info("using %s", provider)

        assert "S3CRET-IN-A-LOG" not in caplog.text, "the credential reached the log"
        assert "'token': '...'" in caplog.text, (
            f"the parameter name should survive for diagnosis: {caplog.text[:200]}"
        )


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
        with pytest.raises(AttributeError, match="cannot assign to field"):
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
        with pytest.raises(TypeError, match="does not support item assignment"):
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
        with pytest.raises(TypeError, match="does not support item assignment"):
            copy.extra_params["token"] = "sneaked in"

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            ({"url": " https://example.org/wms"}, "whitespace"),
            ({"version": "2.0.0"}, "version must be one of"),
            ({"transparent": 1}, "transparent must be a bool"),
            ({"tile_size": 0}, "tile_size must be a positive int"),
            ({"styles": None}, "styles must be a string"),
            ({"attribution": None}, "attribution must be a string"),
        ],
    )
    def test_replace_revalidates_every_wms_check(self, wms, kwargs, message):
        """Each WMS check fires on a `replace` copy, not only on construction.

        Args:
            wms: The GetMap provider fixture.
            kwargs: The field the copy tries to change, and its bad value.
            message: The fragment expected in the error.

        Test scenario:
            These checks were added after the class already existed, so the
            standing risk is one written against the constructor and never
            reached by `replace` -- which is the supported way to vary a frozen
            provider and assembles the copy field by field. The scheme check is
            covered above; this is the rest, one row per validator.
        """
        with pytest.raises(ValueError, match=message):
            dataclasses.replace(wms, **kwargs)

    @pytest.mark.parametrize(
        "kwargs, message",
        [
            (
                {"url": "https://example.org/wmts/{TileMatrix}.png"},
                "must address a tile",
            ),
            ({"version": "2.0.0"}, "version must be one of"),
            ({"style": ""}, "style must be a non-empty string"),
            ({"attribution": None}, "attribution must be a string"),
        ],
    )
    def test_replace_revalidates_every_wmts_check(self, wmts, kwargs, message):
        """Each WMTS check fires on a `replace` copy too.

        Args:
            wmts: The KVP provider fixture.
            kwargs: The field the copy tries to change, and its bad value.
            message: The fragment expected in the error.

        Test scenario:
            The template check matters most here: `replace(provider, url=...)`
            is exactly how a caller repoints an existing provider at a RESTful
            template, and a half-written one collapses the whole mosaic onto a
            single repeated image without erroring at fetch time.
        """
        with pytest.raises(ValueError, match=message):
            dataclasses.replace(wmts, **kwargs)


class TestCredentialsAreNotLogged:
    """A token in `extra_params` does not reach the debug log."""

    def test_a_failed_fetch_logs_a_redacted_url(self, caplog):
        """The retry log keeps the parameter names but not their values.

        Args:
            caplog: pytest's log capture fixture.

        Test scenario:
            This module documents `extra_params` as the place to put an API
            key, and `fetch_single_tile` wrote the full URL to the debug log on
            every failed attempt -- so a flaky service quietly persisted the
            credential wherever the logs go.
        """
        provider = WMSProvider(
            url="https://example.org/wms",
            layers="ortho",
            extra_params={"token": "S3CRET-VALUE"},
        )

        def explode(request, timeout=None):
            raise OSError("connection reset")

        with (
            caplog.at_level("DEBUG", logger="cleopatra.basemap.tiles"),
            patch.object(tiles_mod, "urlopen_http", side_effect=explode),
            pytest.raises(ConnectionError),
        ):
            fetch_single_tile(Tile(0, 0, 0), provider, timeout=1, retries=0)

        assert "S3CRET-VALUE" not in caplog.text, "the credential reached the log"
        assert "token=..." in caplog.text, (
            f"the parameter name should survive for diagnosis: {caplog.text[:200]}"
        )


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
        # Asserted against an independently spelled-out URL rather than
        # `provider.build_url(...)`, which would only prove the pipeline called
        # the method -- not that it called it with the right tile.
        expected = {
            "wmts": (
                "https://example.org/wmts?SERVICE=WMTS&REQUEST=GetTile"
                "&VERSION=1.0.0&LAYER=TrueColor&STYLE=default"
                "&TILEMATRIXSET=GoogleMapsCompatible&TILEMATRIX=3&TILEROW=2"
                "&TILECOL=4&FORMAT=image%2Fpng"
            ),
            "wms": (
                "https://example.org/wms?SERVICE=WMS&REQUEST=GetMap"
                "&VERSION=1.3.0&LAYERS=ortho&STYLES=&CRS=EPSG%3A3857"
                "&BBOX=0%2C5009377.085697310976684093475341796875"
                "%2C5009377.085697310976684093475341796875"
                "%2C10018754.17139462195336818695068359375"
                "&WIDTH=256&HEIGHT=256&FORMAT=image%2Fpng&TRANSPARENT=TRUE"
            ),
        }[kind]
        assert recorded_urls == [expected], (
            f"requested {recorded_urls}, expected [{expected}]"
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
