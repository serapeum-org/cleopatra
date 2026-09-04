"""Tests for `cleopatra.basemap._net`.

The shared opener is what makes the scheme restriction structural rather
than a prefix check every call site has to remember: only the HTTP(S)
handlers are registered, so no handler claims a `file://` / `ftp://` /
`data:` request and `OpenerDirector.open` refuses it outright. These tests
never leave the loopback interface: the refused schemes never reach a
socket, and the one test that must get past the scheme gate connects to a
closed port on 127.0.0.1. That address is a literal IP, so it is resolved
locally without a DNS query -- no resolver can make the result vary, which a
hostname (even a reserved one) would allow.
"""

import socket
import urllib.error

import pytest

from cleopatra.basemap._net import _HTTP_ONLY_OPENER, urlopen_http


def _closed_loopback_url() -> str:
    """Return an http URL for a loopback port that is guaranteed closed.

    Binding port 0 lets the OS pick a free port; closing the socket
    immediately leaves it unused, so a connect attempt is *refused* rather
    than filtered. A hard-coded port such as 1 can be silently dropped by a
    hardened container or an endpoint agent, which would hang for the whole
    timeout and then raise `TimeoutError` instead of `ConnectionRefusedError`.

    Returns:
        str: An `http://127.0.0.1:<port>/...` URL nothing is listening on.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    return f"http://127.0.0.1:{port}/0/0/0.png"


class TestOpenerComposition:
    """The opener is built with only the HTTP(S) handlers registered."""

    def test_http_handlers_are_present(self):
        """`HTTPHandler` and `HTTPSHandler` are what claim a request.

        Test scenario:
            The permissive half of the guarantee. Asserting only that other
            schemes are absent would also pass for an opener that can open
            nothing at all.
        """
        registered = {type(handler).__name__ for handler in _HTTP_ONLY_OPENER.handlers}

        assert {"HTTPHandler", "HTTPSHandler"} <= registered, (
            f"http(s) must be openable; registered handlers: {sorted(registered)}"
        )

    @pytest.mark.parametrize(
        "handler", ["FileHandler", "FTPHandler", "DataHandler", "UnknownHandler"]
    )
    def test_other_scheme_handlers_are_absent(self, handler):
        """No handler for any other scheme is registered.

        Test scenario:
            This is the structural half of the guarantee: a scheme cannot be
            opened if nothing can claim it. `UnknownHandler` is checked too --
            it is registered on the *opener class* to raise, and must never be
            replaced by something that resolves a request.
        """
        registered = {type(h).__name__ for h in _HTTP_ONLY_OPENER.handlers}

        if handler == "UnknownHandler":
            assert handler in registered, "the refusing catch-all must stay registered"
        else:
            assert handler not in registered, f"{handler} must not be registered"


class TestUrlopenHttp:
    """`urlopen_http` opens only http/https requests."""

    def test_file_scheme_is_refused(self, tmp_path):
        """A `file://` URL cannot be read through the shared opener.

        Test scenario:
            The file is really written, so the refusal cannot be mistaken for
            a "file not found" -- plain `urllib.request.urlopen` would return
            its contents. The restricted opener has no `FileHandler`, so the
            request is refused as an unknown scheme instead.
        """
        target = tmp_path / "local.txt"
        target.write_text("not reachable over http", encoding="utf-8")
        assert target.exists(), "the file must exist for the refusal to be meaningful"
        uri = target.as_uri()

        with pytest.raises(urllib.error.URLError, match="unknown url type"):
            urlopen_http(uri)

    @pytest.mark.parametrize(
        "url",
        ["ftp://example.invalid/tile.png", "data:text/plain,inline"],
        ids=["ftp", "data"],
    )
    def test_other_non_http_schemes_are_refused(self, url):
        """`ftp://` and `data:` are refused for the same structural reason.

        Test scenario:
            The message is asserted, not just the exception type: "unknown url
            type" is what proves no handler claimed the scheme, as opposed to a
            handler claiming it and then failing to connect.
        """
        with pytest.raises(urllib.error.URLError, match="unknown url type"):
            urlopen_http(url)

    def test_http_scheme_is_claimed_by_a_handler(self):
        """An `http://` URL gets past the scheme gate and is really attempted.

        Test scenario:
            Connecting to a closed loopback port fails at `connect()`, which
            only the HTTP handler can reach -- a scheme the opener refused
            would never open a socket at all. Asserting on the wrapped
            `ConnectionRefusedError` (rather than on the absence of a message)
            is what distinguishes the two.
        """
        url = _closed_loopback_url()

        with pytest.raises(urllib.error.URLError) as excinfo:
            urlopen_http(url, timeout=5)

        assert isinstance(excinfo.value.reason, ConnectionRefusedError), (
            "an http URL must fail at the transport layer, not at the scheme "
            f"gate; got {excinfo.value.reason!r}"
        )
