"""Tests for `cleopatra.basemap._net`.

The shared opener is what makes the scheme restriction structural rather
than a prefix check every call site has to remember: only the HTTP(S)
handlers are registered, so no handler claims a `file://` / `ftp://` /
`data:` request and `OpenerDirector.open` refuses it outright. These tests
never touch the network -- every URL they pass is rejected before a socket
is opened.
"""

from __future__ import annotations

import urllib.error

import pytest

from cleopatra.basemap._net import urlopen_http


class TestUrlopenHttp:
    """`urlopen_http` opens only http/https requests."""

    def test_file_scheme_is_refused(self, tmp_path):
        """A `file://` URL cannot be read through the shared opener.

        Test scenario:
            A readable local file is addressed as `file://...`. Plain
            `urllib.request.urlopen` would happily return its contents;
            the restricted opener has no `FileHandler`, so the request
            raises `URLError` instead.
        """
        target = tmp_path / "local.txt"
        target.write_text("not reachable over http", encoding="utf-8")

        with pytest.raises(urllib.error.URLError):
            urlopen_http(target.as_uri())

    @pytest.mark.parametrize(
        "url",
        ["ftp://example.invalid/tile.png", "data:text/plain,inline"],
        ids=["ftp", "data"],
    )
    def test_other_non_http_schemes_are_refused(self, url):
        """`ftp://` and `data:` are refused for the same structural reason."""
        with pytest.raises(urllib.error.URLError):
            urlopen_http(url)

    def test_http_scheme_is_claimed_by_a_handler(self):
        """An `http://` URL reaches the HTTP handler (it fails on DNS, not scheme).

        Test scenario:
            `.invalid` is reserved and never resolves, so the call fails --
            but with a *transport* error, proving the request got past the
            scheme gate rather than being rejected as an unknown type.
        """
        with pytest.raises(urllib.error.URLError) as excinfo:
            urlopen_http("http://cleopatra.invalid/0/0/0.png", timeout=1)

        assert "unknown url type" not in str(excinfo.value), (
            "an http URL must be claimed by the HTTP handler, not rejected as "
            f"an unknown scheme; got {excinfo.value}"
        )
