"""Tests for `cleopatra.basemap.ogc` -- WMS and WMTS addressed as tile providers.

URL construction is pure string work and needs nothing installed. The two
end-to-end classes drive the real `add_tiles` pipeline with only the HTTP layer
mocked, which is what proves `build_url` is actually the seam the renderer uses
-- patching `fetch_tiles` instead (as `test_tiles.py` does for other cases)
would never call the provider at all.
"""

from __future__ import annotations

import base64
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs, urlsplit

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from cleopatra.basemap import tiles as tiles_mod
from cleopatra.basemap.ogc import WMSProvider, WMTSProvider
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
