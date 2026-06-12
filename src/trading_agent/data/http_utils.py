"""HTTP utilities with retry logic."""
from __future__ import annotations

import time
import urllib.request
from typing import Callable


def fetch_with_retry(
    url: str,
    *,
    timeout: float = 20.0,
    max_retries: int = 2,
    retry_base_delay: float = 1.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> bytes:
    """Fetch URL with exponential backoff retry.

    Parameters
    ----------
    url : str
        The URL to fetch.
    timeout : float
        Request timeout in seconds.
    max_retries : int
        Maximum number of retry attempts.
    retry_base_delay : float
        Base delay between retries (doubled each attempt).
    sleep_fn : callable
        Sleep function (injectable for testing).

    Returns
    -------
    bytes
        The response body.

    Raises
    ------
    urllib.error.URLError
        If all retry attempts fail.
    """
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return resp.read()
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                sleep_fn(retry_base_delay * (2 ** attempt))
    raise last_error  # type: ignore[misc]
