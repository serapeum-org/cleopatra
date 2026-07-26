"""Scheme-restricted HTTP fetching shared by `reference` and `tiles`.

Both modules fetch fixed http(s) resources (basemap tiles / static reference
assets). Calling `urllib.request.urlopen` directly accepts *any* scheme the
URL happens to carry (`file://`, `ftp://`, `data:`, ...); a string prefix
check alone only guards whichever call sites remember to add it. Building a
dedicated `urllib.request.OpenerDirector` with *only* the HTTP(S) handlers
registered makes the restriction structural instead: no handler claims a
`file://`/`ftp://`/`data:` request, so `OpenerDirector.open` raises
`URLError('unknown url type: ...')` for it rather than silently honouring it.
"""

from __future__ import annotations

import urllib.error
import urllib.request

#: Opener with only the HTTP(S) handlers registered (plus the supporting
#: redirect/error-processing handlers every `urlopen` call needs) -- no
#: FileHandler, FTPHandler, or DataHandler is present, so those schemes
#: cannot be opened through it at all.
_HTTP_ONLY_OPENER = urllib.request.OpenerDirector()
for _handler in (
    urllib.request.HTTPHandler(),
    urllib.request.HTTPSHandler(),
    urllib.request.HTTPErrorProcessor(),
    urllib.request.HTTPRedirectHandler(),
    urllib.request.HTTPDefaultErrorHandler(),
    urllib.request.UnknownHandler(),
):
    _HTTP_ONLY_OPENER.add_handler(_handler)


def urlopen_http(
    request: urllib.request.Request | str, *, timeout: float | None = None
):
    """Open `request`, structurally refusing any scheme but http/https.

    Args:
        request: A URL string or `urllib.request.Request`.
        timeout: Socket timeout in seconds; `None` waits indefinitely.

    Returns:
        http.client.HTTPResponse: The opened response (use as a context
            manager, as with `urllib.request.urlopen`).

    Raises:
        urllib.error.URLError: If `request`'s scheme is not http/https
            (no handler claims it), or the request otherwise fails.
    """
    return _HTTP_ONLY_OPENER.open(request, timeout=timeout)
