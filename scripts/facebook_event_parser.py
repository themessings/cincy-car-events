#!/usr/bin/env python3
"""Parse Facebook event pages from HTML/metadata with optional Playwright fallback."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta
from typing import Callable, Optional

import dateparser
import requests
from bs4 import BeautifulSoup
from dateutil import tz


def parse_dt(text: str) -> Optional[datetime]:
    if not text:
        return None
    return dateparser.parse(
        text,
        settings={
            "RETURN_AS_TIMEZONE_AWARE": True,
            "TIMEZONE": "America/New_York",
            "TO_TIMEZONE": "America/New_York",
        },
    )


def _clean_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _extract_json_ld(soup: BeautifulSoup) -> list[dict]:
    out = []
    for script in soup.select('script[type="application/ld+json"]'):
        raw = script.get_text(" ", strip=True)
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        if isinstance(payload, dict):
            out.append(payload)
        elif isinstance(payload, list):
            out.extend([x for x in payload if isinstance(x, dict)])
    return out


def _json_dicts(node):
    if isinstance(node, dict):
        yield node
        for v in node.values():
            yield from _json_dicts(v)
    elif isinstance(node, list):
        for item in node:
            yield from _json_dicts(item)


def parse_facebook_event_html(event_url: str, html: str) -> tuple[Optional[dict], str]:
    if not html:
        return None, "empty_html"

    soup = BeautifulSoup(html, "html.parser")
    canonical = _clean_ws((soup.select_one('link[rel="canonical"]') or {}).get("href", "")) if soup else ""
    canonical_url = canonical or event_url

    og_title = _clean_ws((soup.select_one('meta[property="og:title"]') or {}).get("content", ""))
    og_url = _clean_ws((soup.select_one('meta[property="og:url"]') or {}).get("content", ""))
    og_desc = _clean_ws((soup.select_one('meta[property="og:description"]') or {}).get("content", ""))

    title = og_title
    start_dt = None
    end_dt = None
    location = ""
    host = ""
    address = ""

    for payload in _extract_json_ld(soup):
        for node in _json_dicts(payload):
            t = node.get("@type")
            types = [str(x).lower() for x in t] if isinstance(t, list) else [str(t).lower()]
            if "event" not in types:
                continue
            title = title or _clean_ws(node.get("name") or "")
            start_dt = start_dt or parse_dt(_clean_ws(node.get("startDate") or ""))
            end_dt = end_dt or parse_dt(_clean_ws(node.get("endDate") or ""))
            loc = node.get("location") or {}
            if isinstance(loc, dict):
                location = location or _clean_ws(loc.get("name") or "")
                addr = loc.get("address") or {}
                if isinstance(addr, dict):
                    address = address or _clean_ws(" ".join(str(addr.get(k) or "") for k in ["streetAddress", "addressLocality", "addressRegion", "postalCode"]))
            org = node.get("organizer") or {}
            if isinstance(org, dict):
                host = host or _clean_ws(org.get("name") or "")

    if not start_dt:
        for patt in [r'"start_time"\s*:\s*"([^"]+)"', r'"event_start_time"\s*:\s*"([^"]+)"']:
            m = re.search(patt, html)
            if m:
                start_dt = parse_dt(m.group(1))
                if start_dt:
                    break

    if not end_dt:
        for patt in [r'"end_time"\s*:\s*"([^"]+)"', r'"event_end_time"\s*:\s*"([^"]+)"']:
            m = re.search(patt, html)
            if m:
                end_dt = parse_dt(m.group(1))
                if end_dt:
                    break

    if not location and og_desc:
        location = og_desc[:220]

    if not title:
        title = _clean_ws((soup.title.get_text(" ") if soup.title else "").replace("| Facebook", ""))

    if not title or not start_dt:
        return None, "missing_required_title_or_start"

    return {
        "title": title,
        "start_dt": start_dt,
        "end_dt": end_dt or (start_dt + timedelta(hours=2)),
        "location": location,
        "address": address,
        "host": host,
        "url": og_url or canonical_url,
        "canonical_url": canonical_url,
    }, "ok"


def fetch_facebook_event_page(event_url: str, timeout_s: int = 30) -> tuple[Optional[str], str]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        r = requests.get(event_url, headers=headers, timeout=timeout_s)
        if r.status_code != 200:
            return None, f"http_{r.status_code}"
        return r.text, "ok"
    except Exception as ex:
        return None, f"request_failed:{ex}"


def fetch_with_playwright_if_enabled(event_url: str, logger: Callable[[str], None]) -> tuple[Optional[str], str]:
    enabled = (os.getenv("ENABLE_PLAYWRIGHT_FB") or "0").strip().lower() in {"1", "true", "yes"}
    if not enabled:
        return None, "playwright_disabled"

    try:
        from playwright.sync_api import sync_playwright
    except Exception as ex:
        return None, f"playwright_unavailable:{ex}"

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(event_url, timeout=60000, wait_until="networkidle")
            content = page.content()
            browser.close()
            logger(f"ℹ️ Playwright fetch used for Facebook event URL: {event_url}")
            return content, "ok"
    except Exception as ex:
        return None, f"playwright_failed:{ex}"


def parse_facebook_event_page(event_url: str, logger: Callable[[str], None]) -> tuple[Optional[dict], str]:
    html, fetch_reason = fetch_facebook_event_page(event_url)
    if html:
        parsed, reason = parse_facebook_event_html(event_url, html)
        if parsed:
            return parsed, "ok"
        logger(f"⚠️ Facebook HTML parse incomplete for {event_url}: {reason}")

    html_pw, pw_reason = fetch_with_playwright_if_enabled(event_url, logger)
    if html_pw:
        parsed, reason = parse_facebook_event_html(event_url, html_pw)
        if parsed:
            return parsed, "ok_playwright"
        return None, f"playwright_parse_failed:{reason}"

    return None, f"fetch_failed:{fetch_reason};{pw_reason}"


def format_timestamp_et(ts: Optional[int]) -> str:
    if not ts:
        return "unknown"
    return datetime.fromtimestamp(ts, tz=tz.gettz("America/New_York")).isoformat()
