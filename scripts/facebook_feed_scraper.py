#!/usr/bin/env python3
"""Scroll a logged-in Facebook home feed (headless Playwright) and harvest event links.

This module is intentionally best-effort and self-contained. It never raises for
the common "can't run" outcomes — Playwright not installed, no saved session,
login/checkpoint wall, or a network hiccup. Instead it returns a
:class:`FeedScrapeResult` whose ``reason`` explains any zero-yield result so the
calling collector can record a diagnostic rather than fail the whole pipeline.

Authentication uses a Playwright ``storageState`` captured once from a real,
logged-in browser (see the README for how to export it). Supplying credentials
is deliberately not supported: Facebook challenges datacenter logins with
CAPTCHAs/checkpoints and may lock the account.

The heavy Playwright import lives inside :func:`scrape_feed` on purpose, so the
pure helpers below (:func:`load_storage_state`, :func:`harvest_event_hrefs`,
:func:`canonical_event_url`) import and unit-test without the browser runtime.
"""

from __future__ import annotations

import base64
import json
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
from urllib.parse import urlparse

# Facebook's "Most Recent" feed. The default landing feed is re-ranked and
# heavily deduplicated; "h_chr" (chronological) surfaces more distinct posts,
# which means more distinct event shares per scroll.
DEFAULT_FEED_URL = "https://www.facebook.com/?sk=h_chr"

# If a feed navigation lands on any of these, the saved session is dead and the
# user needs to re-export it.
_LOGIN_WALL_MARKERS = ("/login", "/checkpoint", "/recover", "login.php")

_EVENT_ID_RE = re.compile(r"/events/(\d+)")
_EVENT_PHP_EID_RE = re.compile(r"[?&]eid=(\d+)")


def env_flag(name: str, default: bool = False) -> bool:
    """Interpret an env var as a boolean toggle."""
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "y", "on"}


def load_storage_state(spec: Optional[str] = None) -> Optional[object]:
    """Resolve a Playwright ``storage_state`` from the environment.

    Resolution order:
      1. ``FACEBOOK_SESSION_STATE_PATH`` pointing at an existing file.
      2. ``spec`` / ``FACEBOOK_SESSION_STATE`` as a filesystem path.
      3. ``spec`` / ``FACEBOOK_SESSION_STATE`` as raw JSON text.
      4. ``spec`` / ``FACEBOOK_SESSION_STATE`` as base64-encoded JSON
         (convenient for storing multi-line JSON in a single CI secret).

    Returns a value ready to hand to Playwright's ``storage_state`` argument —
    either a path string or a parsed ``dict`` — or ``None`` if nothing usable is
    configured.
    """
    explicit_path = (os.getenv("FACEBOOK_SESSION_STATE_PATH") or "").strip()
    if explicit_path and os.path.isfile(explicit_path):
        return explicit_path

    raw = spec if spec is not None else (os.getenv("FACEBOOK_SESSION_STATE") or "")
    raw = raw.strip()
    if not raw:
        return None

    # A bare filesystem path handed in via FACEBOOK_SESSION_STATE.
    if os.path.isfile(raw):
        return raw

    # Raw JSON text.
    if raw.startswith("{"):
        try:
            return json.loads(raw)
        except Exception:
            return None

    # base64-encoded JSON.
    try:
        decoded = base64.b64decode(raw, validate=False).decode("utf-8", "ignore").strip()
    except Exception:
        return None
    if decoded.startswith("{"):
        try:
            return json.loads(decoded)
        except Exception:
            return None
    return None


def canonical_event_url(href: str) -> str:
    """Reduce a Facebook anchor href to a canonical event URL, or "" if it is not one.

    Handles ``/events/<id>``, ``/events/<id>/permalink/...`` and
    ``event.php?eid=<id>`` shapes. Non-event ``/events/*`` paths such as
    ``/events/create`` or ``/events/birthdays`` yield "" because they carry no
    numeric id.
    """
    s = (href or "").strip()
    if not s:
        return ""
    if "/events/" not in s and "event.php" not in s:
        return ""
    match = _EVENT_ID_RE.search(s) or _EVENT_PHP_EID_RE.search(s)
    if not match:
        return ""
    return f"https://www.facebook.com/events/{match.group(1)}/"


def harvest_event_hrefs(hrefs) -> List[str]:
    """From arbitrary anchor hrefs, return canonical event URLs, order-preserving + deduped."""
    out: List[str] = []
    seen = set()
    for href in hrefs or []:
        canonical = canonical_event_url(href)
        if not canonical or canonical in seen:
            continue
        seen.add(canonical)
        out.append(canonical)
    return out


def is_login_wall(url: str) -> bool:
    """True if the current URL indicates the saved session is invalid/expired."""
    raw = (url or "").lower()
    path = (urlparse(raw).path or "") if "://" in raw else raw
    return any(marker in raw or marker in path for marker in _LOGIN_WALL_MARKERS)


@dataclass
class FeedScrapeResult:
    """Outcome of one feed-scroll pass.

    Attributes:
        event_urls: canonical event URLs we attempted to fetch (bounded by ``max_events``).
        pages: canonical event URL -> fetched page HTML (missing/failed fetches are absent).
        discovered: total distinct event URLs seen while scrolling (may exceed ``event_urls``).
        scrolls_done: how many scroll steps actually ran.
        reason: short machine-readable explanation when the result is empty/partial.
    """

    event_urls: List[str] = field(default_factory=list)
    pages: Dict[str, str] = field(default_factory=dict)
    discovered: int = 0
    scrolls_done: int = 0
    reason: str = ""


def _noop_logger(_msg: str) -> None:  # pragma: no cover - trivial
    pass


def scrape_feed(
    *,
    storage_state: object,
    feed_url: str = DEFAULT_FEED_URL,
    scrolls: int = 20,
    scroll_pause_ms: int = 2000,
    max_events: int = 40,
    nav_timeout_ms: int = 60000,
    headless: bool = True,
    fetch_event_pages: bool = True,
    logger: Optional[Callable[[str], None]] = None,
) -> FeedScrapeResult:
    """Open the logged-in feed, scroll it, harvest event links, and fetch their HTML.

    Returns a :class:`FeedScrapeResult`. Never raises for the expected failure
    modes (Playwright missing, dead session, navigation error): the reason is
    reported on the result instead.
    """
    log = logger or _noop_logger
    result = FeedScrapeResult()

    if not storage_state:
        result.reason = "disabled_missing_session"
        return result

    try:
        from playwright.sync_api import sync_playwright
    except Exception as ex:  # pragma: no cover - depends on optional dep
        result.reason = "playwright_unavailable"
        log(f"⚠️ Facebook feed scroll needs Playwright (pip install playwright && playwright install chromium): {ex}")
        return result

    scrolls = max(1, int(scrolls))
    max_events = max(1, int(max_events))
    scroll_pause_ms = max(250, int(scroll_pause_ms))

    discovered: List[str] = []
    discovered_seen = set()

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=headless)
            try:
                context = browser.new_context(
                    storage_state=storage_state,
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    ),
                    viewport={"width": 1280, "height": 1600},
                    locale="en-US",
                )
                page = context.new_page()
                page.set_default_timeout(nav_timeout_ms)

                try:
                    page.goto(feed_url, timeout=nav_timeout_ms, wait_until="domcontentloaded")
                except Exception as ex:
                    result.reason = "feed_navigation_failed"
                    log(f"⚠️ Facebook feed navigation failed: {ex}")
                    return result

                page.wait_for_timeout(min(scroll_pause_ms, 3000))
                if is_login_wall(page.url):
                    result.reason = "session_invalid_or_expired"
                    log("⚠️ Facebook session invalid/expired (redirected to login/checkpoint). Re-export FACEBOOK_SESSION_STATE.")
                    return result

                # The feed is virtualized: posts scrolled well past unload from the
                # DOM, so harvest after every scroll step and accumulate.
                for step in range(scrolls):
                    try:
                        hrefs = page.eval_on_selector_all(
                            "a[href*='/events/'], a[href*='event.php']",
                            "els => els.map(e => e.href)",
                        )
                    except Exception:
                        hrefs = []
                    for canonical in harvest_event_hrefs(hrefs):
                        if canonical not in discovered_seen:
                            discovered_seen.add(canonical)
                            discovered.append(canonical)

                    result.scrolls_done = step + 1
                    try:
                        page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
                    except Exception:
                        break
                    page.wait_for_timeout(scroll_pause_ms)

                result.discovered = len(discovered)
                log(
                    f"ℹ️ Facebook feed scroll: {result.scrolls_done} scrolls, "
                    f"{result.discovered} distinct event links discovered."
                )

                event_urls = discovered[:max_events]
                result.event_urls = event_urls
                if result.discovered > len(event_urls):
                    log(
                        f"ℹ️ Facebook feed scroll capped parsing at max_events={max_events} "
                        f"({result.discovered - len(event_urls)} discovered links not fetched this run)."
                    )

                if fetch_event_pages:
                    for event_url in event_urls:
                        try:
                            page.goto(event_url, timeout=nav_timeout_ms, wait_until="domcontentloaded")
                            page.wait_for_timeout(min(scroll_pause_ms, 1500))
                            result.pages[event_url] = page.content()
                        except Exception as ex:
                            log(f"⚠️ Facebook event page fetch failed ({event_url}): {ex}")

                if not event_urls and not result.reason:
                    result.reason = "no_event_links_in_feed"
                return result
            finally:
                browser.close()
    except Exception as ex:  # pragma: no cover - defensive; keep the pipeline alive
        result.reason = result.reason or "scrape_failed"
        log(f"⚠️ Facebook feed scroll failed: {ex}")
        return result
