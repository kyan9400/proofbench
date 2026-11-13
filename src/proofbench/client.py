"""How the runner reaches shopdesk.

In CI and on a laptop the service runs in-process: no port, no container, no
waiting for a health check. In the deployed demo it is a real HTTP service on the
other side of a network. Both are the same `httpx.Client` to everything above, so
the tests exercise the same code path the demo does.
"""

from __future__ import annotations

import httpx


def shopdesk_client(base_url: str | None = None, timeout_s: float = 30.0) -> httpx.Client:
    if base_url:
        return httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout_s)

    # Imported lazily: a deployed runner has no reason to load the service it is
    # pointing at, and importing it would give it a second, unused world.
    from fastapi.testclient import TestClient

    from shopdesk.app import app

    return TestClient(app)
