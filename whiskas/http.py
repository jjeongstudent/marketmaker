"""A small retrying HTTP helper shared by all API clients."""

from __future__ import annotations

import random
import threading
import time

import requests

_local = threading.local()

RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504, 520, 522, 524}


def session() -> requests.Session:
    """One keep-alive session per thread (requests.Session is not thread-safe)."""
    s = getattr(_local, "session", None)
    if s is None:
        s = requests.Session()
        s.headers["User-Agent"] = "whiskas-research/0.1"
        adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=4)
        s.mount("https://", adapter)
        _local.session = s
    return s


def get_json(url: str, params: dict | None = None, *, timeout: float = 60, retries: int = 6):
    """GET `url` and decode JSON, retrying on transient failures with exponential backoff."""
    delay = 1.0
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            r = session().get(url, params=params, timeout=timeout)
            if r.status_code in RETRY_STATUS:
                raise requests.HTTPError(f"{r.status_code} {r.text[:200]}", response=r)
            r.raise_for_status()
            return r.json()
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status is not None and status not in RETRY_STATUS:
                raise
            last_exc = exc
            time.sleep(delay + random.random() * delay)
            delay = min(delay * 2, 30)
    raise RuntimeError(f"GET {url} failed after {retries} attempts: {last_exc}")
