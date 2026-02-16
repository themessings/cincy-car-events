#!/usr/bin/env python3
import csv
import json
import os
import re
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urljoin, urlparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import dateparser
import requests
import yaml
from bs4 import BeautifulSoup
from dateutil import tz
from icalendar import Calendar
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from scripts.facebook_event_parser import parse_facebook_event_page
from scripts.facebook_token_manager import TokenManager

# -------------------------
# Paths
# -------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")
RUNS_DIR = os.path.join(DATA_DIR, "runs")
CONFIG_PATH = os.path.join(ROOT, "config", "sources.yml")
EVENTS_JSON_PATH = os.path.join(DATA_DIR, "events.json")
EVENTS_CSV_PATH = os.path.join(DATA_DIR, "events.csv")
GEOCODE_CACHE_PATH = os.path.join(DATA_DIR, "geocode_cache.json")
URL_CACHE_PATH = os.path.join(DATA_DIR, "url_cache.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)

DEFAULT_APEX_FOLDER_ID = "1bd9LR2JfE7AJm9Z5dwlIc_u5WpO3atQM"
APEX_PARENT_FOLDER_ID = os.getenv("APEX_PARENT_FOLDER_ID", DEFAULT_APEX_FOLDER_ID)
APEX_PARENT_FOLDER_URL = os.getenv("APEX_PARENT_FOLDER_URL")
APEX_SUBFOLDER_NAME = os.getenv("APEX_SUBFOLDER_NAME", "Apex events")
APEX_SPREADSHEET_NAME = os.getenv("APEX_SPREADSHEET_NAME", "Apex Events")
APEX_SHARED_DRIVE_ID = os.getenv("APEX_SHARED_DRIVE_ID")
APEX_SHARE_WITH_EMAIL = os.getenv("APEX_SHARE_WITH_EMAIL")
APEX_SPREADSHEET_ID = os.getenv("APEX_SPREADSHEET_ID")

FACEBOOK_ACCESS_TOKEN = os.getenv("FACEBOOK_ACCESS_TOKEN")
FACEBOOK_APP_ID = os.getenv("FACEBOOK_APP_ID")
FACEBOOK_APP_SECRET = os.getenv("FACEBOOK_APP_SECRET")

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
SERPAPI_LOCATION = (os.getenv("SERPAPI_LOCATION", "Cincinnati, OH") or "").strip()
SERPAPI_GL = (os.getenv("SERPAPI_GL", "us") or "").strip()
SERPAPI_HL = (os.getenv("SERPAPI_HL", "en") or "").strip()
SERPAPI_EVENTS_DATE_FILTER = (os.getenv("SERPAPI_EVENTS_DATE_FILTER", "date:month") or "").strip()

FACEBOOK_GRAPH_VERSION = "v18.0"
DEFAULT_FACEBOOK_PAGES_TAB = os.getenv("APEX_FACEBOOK_PAGES_TAB", "Pages")
DEFAULT_HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
SOURCE_DIAGNOSTICS: Dict[str, dict] = {}
FACEBOOK_TARGETS_CACHE: Optional[List[dict]] = None
FACEBOOK_GRAPH_RUNTIME: Dict[str, object] = {"checked": False, "valid": None, "reason": "unchecked"}
FACEBOOK_COVERAGE: Dict[str, object] = {"token": {}, "serp_queries": [], "urls_discovered": 0, "urls_parsed": 0, "urls_failed": 0, "failure_reasons": Counter(), "page_events": 0}
TOKEN_MANAGER: Optional[TokenManager] = None


def get_token_manager() -> TokenManager:
    global TOKEN_MANAGER
    if TOKEN_MANAGER is None:
        TOKEN_MANAGER = TokenManager(log)
    return TOKEN_MANAGER


def get_facebook_access_token() -> str:
    return clean_ws(os.getenv("FACEBOOK_ACCESS_TOKEN") or FACEBOOK_ACCESS_TOKEN or "")


def extract_google_spreadsheet_id(value: str) -> str:
    raw = clean_ws(value)
    if not raw:
        return ""
    if "docs.google.com/spreadsheets" in raw:
        m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", raw)
        if m:
            return m.group(1)
    return raw


def normalize_facebook_pages_sheet_id(raw_value: str, context: str = "collector") -> str:
    normalized = extract_google_spreadsheet_id(raw_value)
    if raw_value and normalized and raw_value != normalized:
        log(f"ℹ️ {context}: APEX_FACEBOOK_PAGES_SHEET_ID provided as URL; using ID '{normalized}'.")
    if normalized and re.fullmatch(r"[a-zA-Z0-9-_]{20,}", normalized):
        log(f"ℹ️ {context}: Facebook pages spreadsheet ID resolved to '{normalized}'.")
        return normalized
    if raw_value:
        log(
            f"⚠️ {context}: Could not parse APEX_FACEBOOK_PAGES_SHEET_ID='{raw_value}' into a valid spreadsheet ID."
        )
    return ""


def decode_serpapi_candidate_url(url: str) -> str:
    """Decode common Google/FB redirect wrappers down to a direct target URL when possible."""
    raw = clean_ws(url)
    if not raw:
        return ""

    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    host = (parsed.netloc or "").lower()

    if host.endswith("google.com") and parsed.path == "/url":
        q_target = clean_ws((parse_qs(parsed.query).get("q") or [""])[0])
        if q_target:
            return decode_serpapi_candidate_url(unquote(q_target))

    if host in {"l.facebook.com", "lm.facebook.com"} and parsed.path.startswith("/l.php"):
        u_target = clean_ws((parse_qs(parsed.query).get("u") or [""])[0])
        if u_target:
            return decode_serpapi_candidate_url(unquote(u_target))

    return raw


def extract_facebook_event_id(url: str) -> Optional[str]:
    """Extract event ID from facebook events/event.php URLs."""
    if not url:
        return None

    decoded = decode_serpapi_candidate_url(url)
    parsed = urlparse(decoded if "://" in decoded else f"https://{decoded}")
    host = (parsed.netloc or "").lower()
    if "facebook.com" not in host:
        return None

    path = clean_ws(parsed.path or "")
    m = re.search(r"/events/(\d+)", path)
    if m:
        return m.group(1)

    if path.rstrip("/").lower().endswith("/event.php"):
        eid = clean_ws((parse_qs(parsed.query).get("eid") or [""])[0])
        if eid.isdigit():
            return eid

    m = re.search(r"facebook\.com/events/(\d+)", decoded)
    if m:
        return m.group(1)
    return None


def normalize_facebook_event_url(url: str) -> str:
    raw = decode_serpapi_candidate_url(url)
    if not raw:
        return ""

    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    host = (parsed.netloc or "").lower()
    if "webcache.googleusercontent.com" in host:
        return ""

    event_id = extract_facebook_event_id(raw)
    if event_id:
        return f"https://www.facebook.com/events/{event_id}/"

    cleaned_host = host.replace("m.facebook.com", "www.facebook.com")
    cleaned_path = clean_ws(parsed.path or "")
    cleaned_path = cleaned_path if cleaned_path.startswith("/") else f"/{cleaned_path}" if cleaned_path else ""
    if not cleaned_host:
        return ""
    return f"https://{cleaned_host}{cleaned_path}"
    url = clean_ws(url)
    if not url:
        return ""

    parsed = urlparse(url if "://" in url else f"https://{url}")
    host = (parsed.netloc or "").lower().replace("m.facebook.com", "www.facebook.com")
    path = clean_ws(parsed.path or "")

    if "facebook.com" in host:
        event_id = extract_facebook_event_id(f"https://{host}{path}")
        if event_id:
            return f"https://www.facebook.com/events/{event_id}/"

    normalized_path = path if path.startswith("/") else f"/{path}" if path else ""
    return f"https://{host}{normalized_path}" if host else clean_ws(url)


def classify_facebook_pages_url(page_url: str) -> str:
    raw = clean_ws(page_url).lower()
    if "facebook.com/groups/" in raw:
        return "group"
    if "facebook.com/" in raw:
        return "page"
    return "non_facebook"


def normalize_facebook_target_url(url: str) -> str:
    raw = decode_serpapi_candidate_url(url)
    if not raw:
        return ""
    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    host = (parsed.netloc or "").lower().replace("m.facebook.com", "www.facebook.com")
    path = clean_ws(parsed.path or "").strip("/")
    if not host:
        return ""
    return f"https://{host}/{path}/" if path else f"https://{host}/"
    if "facebook.com/" not in raw:
        return "non_facebook"
    if "/groups/" in raw:
        return "group"
    return "page"


def extract_facebook_group_key(group_url: str) -> str:
    raw = clean_ws(group_url)
    if not raw:
        return ""

    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    path = clean_ws(parsed.path).strip("/")
    segments = [clean_ws(seg) for seg in path.split("/") if clean_ws(seg)]
    for idx, segment in enumerate(segments):
        if segment.lower() == "groups" and idx + 1 < len(segments):
            key = clean_ws(segments[idx + 1])
            return re.sub(r"[^A-Za-z0-9._-]", "", key)
            return clean_ws(segments[idx + 1])
    return ""


def parse_facebook_event_from_html(event_url: str, html: str) -> Optional[dict]:
    """
    Best-effort extraction when Graph API can’t be used.
    Facebook pages are JS-heavy; this parser handles common embedded payload patterns.
    Returns dict with: title, start_dt, end_dt, location, url.
    """
    if not html:
        return None

    lower_html = html.lower()
    if any(x in lower_html for x in ["log in to facebook", "login_form", "consent", "accept all cookies"]):
        return None

    def _epoch_to_dt(ts_raw: str) -> Optional[datetime]:
        try:
            ts = int(ts_raw)
            if ts > 10_000_000_000:
                ts //= 1000
            return datetime.fromtimestamp(ts, tz=tz.gettz("America/New_York"))
        except Exception:
            return None

    title = ""
    m = re.search(r'<meta[^>]+property="og:title"[^>]+content="([^"]+)"', html, re.IGNORECASE)
    if m:
        title = clean_ws(m.group(1))
    if not title:
        m = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if m:
            title = clean_ws(re.sub(r"\s*\|\s*Facebook\s*$", "", m.group(1), flags=re.IGNORECASE))

    location = ""
    for patt in [
        r'"event_place"\s*:\s*\{[^{}]*?"name"\s*:\s*"([^"]+)"',
        r'"place"\s*:\s*\{[^{}]*?"name"\s*:\s*"([^"]+)"',
        r'<meta[^>]+property="og:description"[^>]+content="([^"]+)"',
    ]:
        m = re.search(patt, html, re.IGNORECASE | re.DOTALL)
        if m:
            location = clean_ws(m.group(1))[:300]
            if location:
                break

    start_dt = None
    end_dt = None
    for patt in [r'"start_timestamp"\s*:\s*(\d{9,13})', r'"event_start_time"\s*:\s*(\d{9,13})']:
        m = re.search(patt, html)
        if m:
            start_dt = _epoch_to_dt(m.group(1))
            if start_dt:
                break

    for patt in [r'"end_timestamp"\s*:\s*(\d{9,13})', r'"event_end_time"\s*:\s*(\d{9,13})']:
        m = re.search(patt, html)
        if m:
            end_dt = _epoch_to_dt(m.group(1))
            if end_dt:
                break

    if not start_dt:
        # ISO date fallback inside embedded JSON payloads
        for patt in [r'"start_time"\s*:\s*"([^"]+)"', r'"event_start_time"\s*:\s*"([^"]+)"']:
            m = re.search(patt, html)
            if m:
                start_dt = parse_dt(m.group(1))
                if start_dt:
                    break

    if not title or not start_dt:
        return None

    return {
        "title": title,
        "start_dt": start_dt,
        "end_dt": end_dt or (start_dt + timedelta(hours=2)),
        "location": location,
        "url": event_url,
    }

def fetch_facebook_event_via_graph(event_id: str) -> Optional[dict]:
    """
    Uses Graph API to fetch a public event by ID (best case).
    Returns normalized raw event dict with title/start_dt/end_dt/location/url.
    """
    if not facebook_graph_usable():
        return None

    url = f"https://graph.facebook.com/v18.0/{event_id}"
    params = {
        "access_token": get_facebook_access_token(),
        "fields": "name,start_time,end_time,place,timezone,description",
    }

    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        return None
    item = r.json()

    title = clean_ws(item.get("name", ""))
    start_dt = parse_dt(item.get("start_time"))
    end_dt = parse_dt(item.get("end_time")) if item.get("end_time") else None

    place = item.get("place") or {}
    location = clean_ws(place.get("name", ""))

    if place.get("location"):
        loc = place["location"]
        parts = [loc.get("street"), loc.get("city"), loc.get("state"), loc.get("zip")]
        location = clean_ws(" ".join(p for p in parts if p)) or location

    if not title or not start_dt:
        return None

    return {
        "title": title,
        "start_dt": start_dt,
        "end_dt": end_dt or (start_dt + timedelta(hours=2)),
        "location": location,
        "url": f"https://www.facebook.com/events/{event_id}",
    }


def collect_web_search_facebook_events_serpapi(source: dict, url_cache: Dict[str, dict], diagnostics: Optional[dict] = None) -> List[dict]:
    """
    SerpAPI discovers facebook.com/events/<id> URLs.
    Parse metadata from SerpAPI payload (no facebook.com fetch).
    Optional Graph enrichment can be enabled via ENABLE_FACEBOOK_GRAPH_ENRICH=1.
    """
    diagnostics = diagnostics if diagnostics is not None else {}
    diagnostics.setdefault("raw_candidates", 0)
    diagnostics.setdefault("parse_failures", 0)

    query = source.get("query", "")
    max_results = int(source.get("max_results", 50))
    source_name = source.get("name", "facebook:serpapi")

    rows: List[dict] = []
    if clean_ws(query):
        links, payload_rows = serpapi_search(query, max_results=max_results, return_payload=True)
        FACEBOOK_COVERAGE.setdefault("serp_queries", []).append({"query": query, "result_count": len(links)})
        log(f"ℹ️ SerpAPI FB query results: count={len(links)} query={query}")
        for link, row in zip(links, payload_rows):
            rows.append({"url": link, "result": row, "query": query})
    elif SERPAPI_API_KEY:
        cfg = load_yaml(CONFIG_PATH)
        for q in build_serpapi_discovery_queries(cfg, for_facebook=True, limit=10, organizer_terms=build_organizer_seed_terms(load_facebook_targets())):
            links, payload_rows = serpapi_search(q, max_results=min(max_results, 25), return_payload=True)
            FACEBOOK_COVERAGE.setdefault("serp_queries", []).append({"query": q, "result_count": len(links)})
            log(f"ℹ️ SerpAPI FB query results: count={len(links)} query={q}")
            for link, row in zip(links, payload_rows):
                rows.append({"url": link, "result": row, "query": q})
            time.sleep(0.2)
    else:
        diagnostics["reason"] = "no_results_from_search"
        log(f"ℹ️ SerpAPI disabled; skipping {source_name}.")
        return []

    if not rows:
        diagnostics["reason"] = "no_results_from_search"
        return []

    out: List[dict] = []
    now = datetime.now(tz=tz.gettz("America/New_York"))
    graph_state = {"token_expired": False}

    fb_rows = []
    seen = set()
    for row in rows:
        result_item = row.get("result", {}) if isinstance(row, dict) else {}
        candidate_urls = extract_facebook_event_urls_from_serpapi_result(result_item)
        if not candidate_urls:
            fallback = normalize_facebook_event_url(clean_ws(row.get("url", "")))
            candidate_urls = [fallback] if fallback else []
        for nu in candidate_urls:
            if nu in seen:
                continue
            seen.add(nu)
            item = dict(row)
            item["url"] = nu
            fb_rows.append(item)

    if not fb_rows:
        diagnostics["reason"] = "no_results_from_search"
        return []

    for row in fb_rows:
        event_url = row["url"]
        diagnostics["raw_candidates"] += 1
        cached = url_cache.get(event_url)
        if cached:
            try:
                last = datetime.fromisoformat(cached.get("fetched_at_iso"))
                if (now - last) < timedelta(hours=24) and cached.get("event"):
                    e = cached["event"]
                    out.append({
                        "title": e.get("title", ""),
                        "start_dt": datetime.fromisoformat(e["start_iso"]),
                        "end_dt": datetime.fromisoformat(e["end_iso"]),
                        "location": e.get("location", ""),
                        "url": e.get("url", event_url),
                        "source": source_name,
                        "facebook_event_id": e.get("facebook_event_id", ""),
                    })
                    continue
            except Exception:
                pass

        ev = parse_facebook_serpapi_result(row.get("result", {}), source_name=source_name)
        if not ev:
            parsed_page, fail_reason = parse_facebook_event_page(event_url, log)
            if parsed_page:
                ev = {
                    "title": parsed_page.get("title", ""),
                    "start_dt": parsed_page.get("start_dt"),
                    "end_dt": parsed_page.get("end_dt"),
                    "location": clean_ws(f"{parsed_page.get('location', '')} {parsed_page.get('address', '')}"),
                    "url": parsed_page.get("canonical_url") or parsed_page.get("url") or event_url,
                    "source": source_name,
                    "host": parsed_page.get("host", ""),
                }
            else:
                diagnostics["parse_failures"] += 1
                FACEBOOK_COVERAGE["urls_failed"] = int(FACEBOOK_COVERAGE.get("urls_failed", 0) or 0) + 1
                FACEBOOK_COVERAGE["failure_reasons"][fail_reason] += 1
                url_cache[event_url] = {"fetched_at_iso": now.isoformat(), "event": None, "failure_reason": fail_reason}
                continue

        ev = maybe_enrich_facebook_event_via_graph(ev, graph_state=graph_state)
        out.append(ev)
        FACEBOOK_COVERAGE["urls_parsed"] = int(FACEBOOK_COVERAGE.get("urls_parsed", 0) or 0) + 1
        url_cache[event_url] = {
            "fetched_at_iso": now.isoformat(),
            "event": {
                "title": ev["title"],
                "start_iso": ev["start_dt"].isoformat(),
                "end_iso": ev["end_dt"].isoformat(),
                "location": ev.get("location", ""),
                "url": ev.get("url", event_url),
                "facebook_event_id": ev.get("facebook_event_id", ""),
            },
        }

    if diagnostics.get("reason") is None and not out:
        diagnostics["reason"] = "parsing_schema_changed"

    FACEBOOK_COVERAGE["urls_discovered"] = int(FACEBOOK_COVERAGE.get("urls_discovered", 0) or 0) + len(fb_rows)
    return out


def drive_list_kwargs() -> dict:
    if APEX_SHARED_DRIVE_ID:
        return {
            "includeItemsFromAllDrives": True,
            "supportsAllDrives": True,
            "driveId": APEX_SHARED_DRIVE_ID,
            "corpora": "drive",
        }
    return {
        "includeItemsFromAllDrives": True,
        "supportsAllDrives": True,
    }


def resolve_parent_folder_id() -> str:
    if APEX_PARENT_FOLDER_URL:
        match = re.search(r"/folders/([a-zA-Z0-9_-]+)", APEX_PARENT_FOLDER_URL)
        if match:
            return match.group(1)
    return APEX_PARENT_FOLDER_ID


# -------------------------
# Models
# -------------------------
@dataclass
class EventItem:
    title: str
    start_iso: str
    end_iso: str
    location: str
    city_state: str
    url: str
    source: str
    category: str  # "local" or "rally"
    miles_from_cincy: Optional[float]
    lat: Optional[float]
    lon: Optional[float]
    last_seen_iso: str


class FacebookGraphTokenExpiredError(RuntimeError):
    """Raised when the Facebook Graph access token is expired/invalid."""


class SourceDisabledError(RuntimeError):
    """Raised when a source is intentionally disabled for the remainder of a run."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


def record_source_diagnostics(source_name: str, **kwargs) -> None:
    current = SOURCE_DIAGNOSTICS.get(source_name, {}).copy()
    current.update({k: v for k, v in kwargs.items() if v is not None})
    SOURCE_DIAGNOSTICS[source_name] = current


# -------------------------
# Helpers
# -------------------------
def now_et_iso() -> str:
    return datetime.now(tz=tz.gettz("America/New_York")).isoformat()


def parse_dt(text: str) -> Optional[datetime]:
    if not text:
        return None
    dt = dateparser.parse(
        text,
        settings={
            "RETURN_AS_TIMEZONE_AWARE": True,
            "TIMEZONE": "America/New_York",
            "TO_TIMEZONE": "America/New_York",
        },
    )
    return dt


def haversine_miles(lat1, lon1, lat2, lon2) -> float:
    R = 3958.7613
    import math

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def extract_facebook_page_identifier(page_url: str) -> str:
    """Extract a canonical Facebook page identifier (username or numeric ID) from a URL-ish value."""
    raw = clean_ws(page_url)
    if not raw:
        return ""

    if raw.isdigit():
        return raw

    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    host = (parsed.netloc or "").lower().replace("m.facebook.com", "www.facebook.com")
    path = clean_ws(parsed.path).strip("/")

    if "facebook.com" in host:
        if path.lower() == "profile.php":
            profile_id = clean_ws((parse_qs(parsed.query).get("id") or [""])[0])
            return profile_id if profile_id.isdigit() else ""

        segments = [clean_ws(seg) for seg in path.split("/") if clean_ws(seg)]
        if segments and segments[0].lower() == "pg" and len(segments) > 1:
            ident = segments[1]
        elif segments:
            ident = segments[0]
        else:
            ident = ""

        if ident.lower() in {"events", "pages", "profile.php"}:
            return ""
        return ident

    cleaned = re.sub(r"\?.*$", "", raw).strip("/")
    return cleaned if re.fullmatch(r"[A-Za-z0-9.\-_%]{3,}", cleaned) else ""


def parse_facebook_pages_from_env() -> List[dict]:
    raw = clean_ws(os.getenv("FACEBOOK_PAGE_IDS", ""))
    if not raw:
        return []

    out: List[dict] = []
    seen_urls = set()
    for token in raw.split(","):
        page_url = normalize_facebook_target_url(token)
        if not page_url or page_url in seen_urls:
            continue
        seen_urls.add(page_url)

        page_type = classify_facebook_pages_url(page_url)
        identifier = extract_facebook_page_identifier(page_url) if page_type in {"page", "group"} else ""
        if page_type in {"page", "group"} and not identifier:
            log(f"⚠️ Ignoring FACEBOOK_PAGE_IDS entry; unable to parse facebook URL: {token}")
            continue

        out.append({
            "page_url": page_url,
            "page_identifier": identifier,
            "page_type": page_type,
            "page_type": classify_facebook_pages_url(page_url),
            "enabled": True,
            "label": "",
            "notes": "",
            "origin": "env",
        })
    if out:
        log(f"✅ Loaded Facebook targets from FACEBOOK_PAGE_IDS: {len(out)}")
    return out


def load_facebook_pages_from_sheet() -> List[dict]:
    """Load URLs from the Pages tab and classify as page/group/non_facebook."""
    sheet_id = normalize_facebook_pages_sheet_id(clean_ws(os.getenv("APEX_FACEBOOK_PAGES_SHEET_ID", "")))
    if not sheet_id:
        log("⚠️ Missing APEX_FACEBOOK_PAGES_SHEET_ID; Pages sheet source disabled.")
        return []

    tab_name = clean_ws(os.getenv("APEX_FACEBOOK_PAGES_TAB", "")) or DEFAULT_FACEBOOK_PAGES_TAB

    creds = get_google_credentials()
    if not creds:
        log("⚠️ Cannot load Facebook pages sheet: missing Google credentials.")
        return []

    sheets = build("sheets", "v4", credentials=creds)

    try:
        resp = sheets.spreadsheets().values().get(
            spreadsheetId=sheet_id,
            range=f"{tab_name}!A1:D2000",
        ).execute()
    except HttpError as ex:
        status = getattr(ex, "status_code", None) or getattr(getattr(ex, "resp", None), "status", None)
        if status == 404:
            log(
                "⚠️ Facebook Pages sheet not found (HTTP 404). "
                "Confirm APEX_FACEBOOK_PAGES_SHEET_ID is the spreadsheet ID and that the service account has access."
            )
        else:
            log(f"⚠️ Failed reading {tab_name}!A1:D2000 from APEX_FACEBOOK_PAGES_SHEET_ID={sheet_id}: {ex}")
        return []

    rows = resp.get("values", [])
    if not rows:
        log(f"⚠️ Facebook Pages tab '{tab_name}' is missing or empty; continuing with no configured pages.")
        return []

    headers = [clean_ws(c).lower() for c in rows[0]]
    header_idx = {h: i for i, h in enumerate(headers) if h}
    if "page_url" not in header_idx:
        log(f"⚠️ {tab_name}!A1 header row missing required column 'page_url'; found headers={headers}")
        return []

    out: List[dict] = []
    seen_urls = set()
    malformed_facebook_rows = 0
    non_facebook_rows = 0
    duplicate_rows = 0

    for idx, r in enumerate(rows[1:], start=2):
        page_url_raw = clean_ws(r[header_idx["page_url"]] if len(r) > header_idx["page_url"] else "")
        if not page_url_raw:
            continue

        page_url = normalize_facebook_target_url(page_url_raw)
        if not page_url:
            continue
        if page_url in seen_urls:
            duplicate_rows += 1
            continue
        seen_urls.add(page_url)

        enabled_raw = ""
        if "enabled" in header_idx and len(r) > header_idx["enabled"]:
            enabled_raw = clean_ws(r[header_idx["enabled"]]).lower()
        enabled = enabled_raw not in {"0", "false", "no", "n", "off", "disabled"}

        label = clean_ws(r[header_idx["label"]] if "label" in header_idx and len(r) > header_idx["label"] else "")
        notes = clean_ws(r[header_idx["notes"]] if "notes" in header_idx and len(r) > header_idx["notes"] else "")

        page_type = classify_facebook_pages_url(page_url)
        identifier = ""
        organizer_domain = ""

        if page_type in {"page", "group"}:
            identifier = extract_facebook_page_identifier(page_url)
            if not identifier:
                malformed_facebook_rows += 1
                log(f"⚠️ {tab_name}!{idx} ignored; malformed facebook URL: {page_url_raw}")
                continue
        else:
            non_facebook_rows += 1
            parsed = urlparse(page_url)
            organizer_domain = clean_ws(parsed.netloc.replace("www.", ""))

        out.append({
            "page_url": page_url,
            "page_identifier": identifier,
            "page_type": page_type,
            "enabled": enabled,
            "label": label,
            "notes": notes,
            "organizer_domain": organizer_domain,
            "origin": "sheet",
        })

    log(
        f"✅ Facebook Pages sheet loaded: rows={len(out)} tab={tab_name} "
        f"(non_facebook={non_facebook_rows} deduped={duplicate_rows} malformed={malformed_facebook_rows})"
    )
    return out


def load_facebook_pages(force_reload: bool = False) -> List[dict]:
    global FACEBOOK_TARGETS_CACHE

    if FACEBOOK_TARGETS_CACHE is not None and not force_reload:
        return FACEBOOK_TARGETS_CACHE

    pages = load_facebook_pages_from_sheet()
    if not pages:
        env_pages = parse_facebook_pages_from_env()
        if env_pages:
            log("ℹ️ Using FACEBOOK_PAGE_IDS fallback because sheet source yielded no pages.")
        pages = env_pages

    FACEBOOK_TARGETS_CACHE = pages
    return pages


def load_facebook_targets(force_reload: bool = False) -> Dict[str, List[dict]]:
    """Load classified Facebook targets.

    Args:
        force_reload: When True, bypass in-memory cache and re-read source-of-truth inputs.
    """
    pages = load_facebook_pages(force_reload=force_reload)
    grouped = {"page": [], "group": [], "non_facebook": []}
    for row in pages:
        page_type = row.get("page_type") or classify_facebook_pages_url(row.get("page_url", ""))
        row["page_type"] = page_type
        grouped.setdefault(page_type, []).append(row)
    return grouped


def normalize_facebook_page_event(item: dict, page_name: str) -> Optional[dict]:
    title = clean_ws(item.get("name", ""))
    start_dt = parse_dt(item.get("start_time"))
    end_dt = parse_dt(item.get("end_time")) if item.get("end_time") else None

    place = item.get("place") or {}
    location = clean_ws(place.get("name", ""))
    if isinstance(place, dict) and place.get("location"):
        loc = place["location"] or {}
        parts = [loc.get("street"), loc.get("city"), loc.get("state"), loc.get("zip")]
        location = clean_ws(" ".join(x for x in parts if x)) or location

    if not title or not start_dt:
        return None

    event_id = item.get("id")
    location_bits = []
    city = ""
    state = ""
    if isinstance(place, dict):
        ploc = place.get("location") or {}
        if isinstance(ploc, dict):
            city = clean_ws(ploc.get("city", ""))
            state = clean_ws(ploc.get("state", ""))
            if city:
                location_bits.append(city)
            if state:
                location_bits.append(state)
    city_state = ", ".join([x for x in [city, state] if x])
    return {
        "title": title,
        "start_dt": start_dt,
        "end_dt": end_dt or (start_dt + timedelta(hours=2)),
        "location": location,
        "city_state": city_state or guess_city_state(location),
        "url": f"https://www.facebook.com/events/{event_id}" if event_id else "",
        "source": f"Facebook: {page_name}",
        "event_type": "facebook_page",
        "cost": "",
    }


def is_facebook_group_url(page_url: str) -> bool:
    raw = clean_ws(page_url).lower()
    return "/groups/" in raw


def parse_facebook_graph_error(resp: requests.Response) -> dict:
    try:
        payload = resp.json() or {}
    except Exception:
        return {}
    err = payload.get("error") if isinstance(payload, dict) else {}
    return err if isinstance(err, dict) else {}


def classify_facebook_graph_token_failure(resp: requests.Response) -> str:
    err = parse_facebook_graph_error(resp)
    code = str(err.get("code", ""))
    subcode = str(err.get("error_subcode", ""))
    message = clean_ws(str(err.get("message", ""))).lower()
    if code == "190" or subcode == "463" or "session has expired" in message:
        return "expired"
    if "access token" in message and "invalid" in message:
        return "malformed"
    if resp.status_code in (400, 401, 403):
        return "permissions"
    return f"http_{resp.status_code}"


def is_facebook_token_expired_error(resp: requests.Response) -> bool:
    err = parse_facebook_graph_error(resp)
    code = str(err.get("code", ""))
    subcode = str(err.get("error_subcode", ""))
    message = clean_ws(str(err.get("message", ""))).lower()
    return code == "190" or subcode == "463" or "session has expired" in message


def validate_facebook_graph_token() -> Dict[str, str]:
    status = get_token_manager().ensure_valid_user_token(refresh_days_threshold=7)
    return {"valid": status.get("valid", "no"), "reason": status.get("reason", "unknown")}


def facebook_graph_usable() -> bool:
    token = get_facebook_access_token()
    if not token:
        return False
    valid = FACEBOOK_GRAPH_RUNTIME.get("valid")
    if valid is None:
        return True
    return bool(valid)


def resolve_facebook_page(page_identifier: str, graph_state: Optional[dict] = None) -> Optional[dict]:
    """Resolve page username/ID to Graph page metadata containing numeric id + display name."""
    if not facebook_graph_usable():
        return None

    if graph_state and graph_state.get("token_expired"):
        raise SourceDisabledError("token_expired", "Facebook Graph source disabled after token expiry.")

    if page_identifier.isdigit():
        return {"page_id": page_identifier, "page_name": page_identifier, "page_link": ""}

    url = f"https://graph.facebook.com/{FACEBOOK_GRAPH_VERSION}/{page_identifier}"
    params = {
        "access_token": get_facebook_access_token(),
        "fields": "id,name,link",
    }

    try:
        r = requests.get(url, params=params, headers=DEFAULT_HTTP_HEADERS, timeout=30)
    except Exception as ex:
        log(f"⚠️ Failed resolving Facebook page '{page_identifier}': {ex}")
        return None

    if r.status_code != 200:
        detail = (r.text or "")[:300]
        if is_facebook_token_expired_error(r):
            if graph_state is not None:
                graph_state["token_expired"] = True
                if not graph_state.get("token_message_logged"):
                    log("⚠️ FACEBOOK_ACCESS_TOKEN expired; refresh token and update GitHub secret.")
                    graph_state["token_message_logged"] = True
            raise FacebookGraphTokenExpiredError("FACEBOOK_ACCESS_TOKEN expired")
        if r.status_code in (400, 401, 403):
            log(
                f"⚠️ Facebook page resolve denied for '{page_identifier}'. "
                f"Check token validity/permissions. HTTP {r.status_code}: {detail}"
            )
        else:
            log(f"⚠️ Facebook page resolve failed for '{page_identifier}': HTTP {r.status_code}: {detail}")
        return None

    data = r.json() or {}
    resolved_id = clean_ws(str(data.get("id", "")))
    if not resolved_id:
        log(f"⚠️ Facebook page '{page_identifier}' resolved without an id; skipping.")
        return None

    return {
        "page_id": resolved_id,
        "page_name": clean_ws(data.get("name", "")) or page_identifier,
        "page_link": clean_ws(data.get("link", "")),
    }


def collect_facebook_events_from_pages(pages: List[dict], diagnostics: Optional[dict] = None) -> List[dict]:
    """Graph API pull for page URLs/identifiers via /{page}/events (no HTML scraping)."""
    diagnostics = diagnostics if diagnostics is not None else {}
    diagnostics.setdefault("raw_candidates", 0)
    diagnostics.setdefault("blocked_http", 0)
    diagnostics.setdefault("group_urls_skipped", 0)

    if not get_facebook_access_token():
        diagnostics["reason"] = "disabled_missing_token"
        log("⚠️ Skipping Facebook page events: missing FACEBOOK_ACCESS_TOKEN.")
        return []

    if not facebook_graph_usable():
        diagnostics["reason"] = "disabled_invalid_token"
        reason = clean_ws(str(FACEBOOK_GRAPH_RUNTIME.get("reason", "invalid")))
        log(f"⚠️ Skipping Facebook page events: FACEBOOK_ACCESS_TOKEN invalid ({reason}).")
        return []

    if not pages:
        diagnostics["reason"] = "disabled_no_pages_configured"
        log("ℹ️ No Facebook pages configured; skipping page-events collector.")
        return []

    out: List[dict] = []
    enabled_pages = [p for p in pages if p.get("enabled", True)]
    page_errors: List[Tuple[str, str]] = []
    queried_pages = 0
    total_page_events = 0
    graph_state = {"token_expired": False, "token_message_logged": False}
    log("ℹ️ Facebook pages configured; running page-events collector.")

    for p in enabled_pages:
        page_identifier = clean_ws(p.get("page_identifier") or "")
        page_url = clean_ws(p.get("page_url") or page_identifier)
        page_label = clean_ws(p.get("label") or page_identifier or page_url)
        if not page_identifier:
            continue

        page_type = clean_ws(p.get("page_type") or classify_facebook_pages_url(page_url))
        if page_type == "group" or is_facebook_group_url(page_url):
            diagnostics["group_urls_skipped"] += 1
            log(f"ℹ️ skipping group URL; Graph Page Events requires Page ID + permissions: {page_url}")
            continue
        if page_type == "non_facebook":
            log(f"ℹ️ skipping non-facebook URL in Pages sheet for Graph collector: {page_url}")
            continue

        try:
            resolved = resolve_facebook_page(page_identifier, graph_state=graph_state)
        except FacebookGraphTokenExpiredError:
            diagnostics["reason"] = "disabled_token_expired"
            raise SourceDisabledError("disabled_token_expired", "Facebook Graph disabled due to expired token.")

        if not resolved:
            msg = "could not resolve via Graph API"
            log(f"⚠️ Skipping page; {msg}: {page_url}")
            page_errors.append((page_url, msg))
            continue

        page_id = resolved["page_id"]
        page_name = resolved["page_name"]

        queried_pages += 1
        log(f"--- Facebook pull start: {page_name} ({page_id}) from {page_url}")
        page_access_token, token_reason = get_token_manager().get_page_access_token(page_id)
        if page_access_token:
            log(f"ℹ️ Using page access token for {page_name} ({page_id}).")
        else:
            page_access_token = get_facebook_access_token()
            log(f"⚠️ Page token unavailable for {page_name} ({page_id}); fallback to user token ({token_reason}).")

        url = f"https://graph.facebook.com/{FACEBOOK_GRAPH_VERSION}/{page_id}/events"
        params = {
            "access_token": page_access_token,
            "fields": "id,name,description,start_time,end_time,place,event_times,timezone,is_online,ticket_uri,cover",
            "limit": 100,
            "since": int(time.time()),
        }

        page_event_count = 0
        while url:
            try:
                r = requests.get(url, params=params, headers=DEFAULT_HTTP_HEADERS, timeout=30)
            except Exception as ex:
                err = f"request error: {ex}"
                log(f"⚠️ Facebook request failed for {page_name} ({page_id}): {ex}")
                page_errors.append((page_url, err))
                break

            if r.status_code != 200:
                detail = (r.text or "")[:300]
                err = f"HTTP {r.status_code}: {detail}"
                if is_facebook_token_expired_error(r):
                    graph_state["token_expired"] = True
                    if not graph_state.get("token_message_logged"):
                        log("⚠️ FACEBOOK_ACCESS_TOKEN expired; refresh token and update GitHub secret.")
                        graph_state["token_message_logged"] = True
                    diagnostics["reason"] = "disabled_token_expired"
                    raise SourceDisabledError("disabled_token_expired", "Facebook Graph disabled due to expired token.")
                if r.status_code in (400, 401, 403):
                    diagnostics["blocked_http"] += 1
                    log(
                        f"⚠️ Facebook page events query failed for label='{page_label}' url='{page_url}'. "
                        "Likely permissions/endpoint restrictions. "
                        f"HTTP {r.status_code}: {detail}. "
                        "Next actions: verify FACEBOOK_ACCESS_TOKEN, app permissions, and page visibility."
                    )
                else:
                    log(f"⚠️ FB non-200 for {page_name} ({page_id}): {r.status_code} :: {detail}")
                page_errors.append((page_url, err))
                break

            payload = r.json() or {}
            data_rows = payload.get("data", []) or []
            diagnostics["raw_candidates"] += len(data_rows)
            for item in data_rows:
                normalized = normalize_facebook_page_event(item, page_name)
                if not normalized:
                    continue
                out.append(normalized)
                page_event_count += 1
                total_page_events += 1

            paging = payload.get("paging", {}) or {}
            url = paging.get("next")
            params = None
            time.sleep(0.2)

        if page_event_count == 0:
            log(
                f"⚠️ Facebook page returned 0 events for label='{page_label}' url='{page_url}'. "
                "This may be normal if events edge is restricted or no upcoming events exist."
            )
        log(f"--- Facebook pull done: {page_name} ({page_id}) events={page_event_count}")

    log("📘 Facebook page collector summary:")
    log(f"   Facebook pages configured: {len(pages)}")
    log(f"   Facebook pages enabled: {len(enabled_pages)}")
    log(f"   Pages successfully queried: {queried_pages}")
    log(f"   Total Facebook page events returned: {total_page_events}")
    if page_errors:
        log("   Pages with errors:")
        for page_url, err in page_errors:
            log(f"    - {page_url} :: {err}")
    else:
        log("   Pages with errors: none")

    if diagnostics.get("reason") is None and total_page_events == 0:
        diagnostics["reason"] = "no_results_from_search"

    FACEBOOK_COVERAGE["page_events"] = int(FACEBOOK_COVERAGE.get("page_events", 0) or 0) + total_page_events
    return out


def collect_facebook_from_pages_sheet() -> List[dict]:
    pages = load_facebook_pages()
    return collect_facebook_events_from_pages(pages)


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def clean_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def guess_city_state(location: str) -> str:
    m = re.search(r"([A-Za-z .'-]+),\s*([A-Z]{2})\b", location or "")
    if m:
        return f"{m.group(1).strip()}, {m.group(2)}"
    return ""


def categorize(title: str, location: str, cfg: dict) -> str:
    t = (title + " " + location).lower()
    for kw in cfg["categorization"]["rally_keywords"]:
        if kw.lower() in t:
            return "rally"
    return "local"


def is_automotive_event(
    title: str,
    location: str = "",
    cfg: Optional[dict] = None,
    *_unused,
    **_unused_kw,
) -> bool:
    """Best-effort keyword gate with backwards-compatible signature."""
    text = clean_ws(f"{title} {location}").lower()
    if not text:
        return False

    if not cfg:
        return True

    categorization = cfg.get("categorization", {})
    keywords = [
        *(categorization.get("local_keywords", []) or []),
        *(categorization.get("rally_keywords", []) or []),
    ]
    if not keywords:
        return True

    return any(clean_ws(k).lower() in text for k in keywords if clean_ws(k))


def is_automotive_event_safe(title: str, location: str, cfg: dict) -> bool:
    """Deterministic automotive keyword check used by main() filtering."""
    text = clean_ws(f"{title} {location}").lower()
    if not text:
        return False

    categorization = (cfg or {}).get("categorization", {})
    keywords = [
        *(categorization.get("local_keywords", []) or []),
        *(categorization.get("rally_keywords", []) or []),
    ]
    if not keywords:
        return True

    return any(clean_ws(k).lower() in text for k in keywords if clean_ws(k))


def filter_existing_automotive_events(existing: List[EventItem], cfg: dict) -> List[EventItem]:
    """Filter persisted events to automotive-only without relying on call-signature-sensitive paths."""
    filtered = [e for e in existing if is_automotive_event_safe(e.title, e.location, cfg)]
    dropped = len(existing) - len(filtered)
    if dropped:
        log(f"🧹 Filtered out {dropped} non-automotive persisted events before merge.")
    return filtered
def evaluate_automotive_focus_event(title: str, location: str, source: str, url: str, cfg: dict) -> Tuple[bool, str]:
    """Returns (is_allowed, reason) for automotive filtering transparency."""
    filters = (cfg or {}).get("filters", {})
    text = clean_ws(f"{title} {location} {source} {url}").lower()
    if not text:
        return False, "empty_text"

    focus_keywords = [clean_ws(k).lower() for k in (filters.get("automotive_focus_keywords", []) or []) if clean_ws(k)]
    exclude_keywords = [clean_ws(k).lower() for k in (filters.get("non_automotive_exclude_keywords", []) or []) if clean_ws(k)]

    default_trusted = [
        "facebook.com/events",
        "eventbrite.com",
        "motorsportreg.com",
        "trackrabbit.com",
        "scca.com",
        "carsandcoffeeevents.com",
        "pca.org",
    ]
    trusted_platforms = [
        clean_ws(k).lower() for k in (filters.get("trusted_event_platforms", default_trusted) or []) if clean_ws(k)
    ]

    hard_exclusions = [
        "5k", "10k", "half marathon", "marathon", "music track", "spotify track", "job fair", "hiring event",
        "church service", "yoga", "kids camp", "food pantry",
    ]

    for x in hard_exclusions:
        if x in text:
            return False, f"hard_exclusion:{x}"

    for x in exclude_keywords:
        if x in text:
            return False, f"exclude_keyword:{x}"

    for k in focus_keywords:
        if k in text:
            return True, f"focus_keyword:{k}"

    for k in trusted_platforms:
        if k in text:
            return True, f"trusted_platform:{k}"

    return False, "no_automotive_keyword_match"


def is_automotive_focus_event(title: str, location: str, source: str, url: str, cfg: dict) -> bool:
    ok, _reason = evaluate_automotive_focus_event(title, location, source, url, cfg)
    return ok


def log(msg: str) -> None:
    print(msg, flush=True)


def log_exception_context(context: str, ex: Exception) -> str:
    tb = traceback.format_exc()
    log(f"⚠️ {context}: {ex}")
    log(tb.rstrip())
    return tb


# -------------------------
# Geocoding (Nominatim) + cache
# -------------------------
def geocode(place: str, cache: Dict[str, dict]) -> Optional[Tuple[float, float]]:
    place = clean_ws(place)
    if not place:
        return None

    if place in cache:
        v = cache[place]
        if v and "lat" in v and "lon" in v:
            return float(v["lat"]), float(v["lon"])
        return None

    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": place, "format": "json", "limit": 1}
    headers = {"User-Agent": "cincy-car-events-bot/1.0 (github actions)"}

    try:
        r = requests.get(url, params=params, headers=headers, timeout=20)
        r.raise_for_status()
        data = r.json()
        time.sleep(1.1)
        if data:
            lat, lon = data[0]["lat"], data[0]["lon"]
            cache[place] = {"lat": lat, "lon": lon}
            return float(lat), float(lon)
        cache[place] = None
        return None
    except Exception as ex:
        cache[place] = None
        log(f"⚠️ Geocode failed for '{place}': {ex}")
        return None


def miles_from_home(lat: float, lon: float, home_lat: float, home_lon: float) -> float:
    return haversine_miles(home_lat, home_lon, lat, lon)


# -------------------------
# HTTP fetching
# -------------------------
def fetch_html(url: str) -> BeautifulSoup:
    r = requests.get(url, headers=DEFAULT_HTTP_HEADERS, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


def fetch_text(url: str) -> str:
    """
    Fetch page text with basic retries and a real browser-ish UA.
    This helps when sites rate limit or return transient 403/429/5xx.
    """
    last_err = None
    for attempt in range(1, 4):
        try:
            r = requests.get(url, headers=DEFAULT_HTTP_HEADERS, timeout=35)
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(1.0 * attempt)
                continue
            r.raise_for_status()
            return r.text
        except Exception as ex:
            last_err = ex
            time.sleep(1.0 * attempt)

    raise last_err


# -------------------------
# Existing collectors
# -------------------------
def collect_carsandcoffeeevents_ohio(source: dict) -> List[dict]:
    soup = fetch_html(source["url"])
    events = []

    blocks = soup.select(".tribe-events-calendar-list__event-row, article.tribe-events-calendar-list__event")
    if not blocks:
        blocks = soup.select(".tribe-events-calendar-list__event")

    for b in blocks:
        title_el = b.select_one(".tribe-events-calendar-list__event-title a, .tribe-event-url, a.tribe-events-calendar-list__event-title-link")
        title = clean_ws(title_el.get_text()) if title_el else ""
        url = title_el["href"].strip() if title_el and title_el.has_attr("href") else source["url"]

        time_el = b.select_one("time")
        start_txt = clean_ws(time_el.get("datetime") or time_el.get_text()) if time_el else ""
        start_dt = parse_dt(start_txt) if start_txt else None

        loc_el = b.select_one(".tribe-events-calendar-list__event-venue-title, .tribe-events-calendar-list__event-venue")
        addr_el = b.select_one(".tribe-events-calendar-list__event-venue-address, .tribe-events-venue-details")
        location = clean_ws((loc_el.get_text() if loc_el else "") + " " + (addr_el.get_text() if addr_el else ""))

        if not title or not start_dt:
            continue

        events.append(
            {
                "title": title,
                "start_dt": start_dt,
                "end_dt": start_dt + timedelta(hours=2),
                "location": location,
                "url": url,
                "source": source["name"],
            }
        )

    return events


def collect_wordpress_events_series(source: dict) -> List[dict]:
    soup = fetch_html(source["url"])
    events = []

    rows = soup.select(".tribe-common-g-row.tribe-events-calendar-list__event-row, article.tribe-events-calendar-list__event")
    if not rows:
        rows = soup.select("article")

    for r in rows:
        title_el = r.select_one(".tribe-events-calendar-list__event-title a, a.tribe-events-calendar-list__event-title-link")
        if not title_el:
            continue
        title = clean_ws(title_el.get_text())
        url = title_el["href"].strip() if title_el.has_attr("href") else source["url"]

        time_el = r.select_one("time")
        start_txt = clean_ws(time_el.get("datetime") or time_el.get_text()) if time_el else ""
        start_dt = parse_dt(start_txt) if start_txt else None

        loc_el = r.select_one(".tribe-events-calendar-list__event-venue-title, .tribe-events-calendar-list__event-venue")
        addr_el = r.select_one(".tribe-events-calendar-list__event-venue-address, .tribe-events-venue-details")
        location = clean_ws((loc_el.get_text() if loc_el else "") + " " + (addr_el.get_text() if addr_el else ""))

        if not title or not start_dt:
            continue

        events.append(
            {
                "title": title,
                "start_dt": start_dt,
                "end_dt": start_dt + timedelta(hours=2),
                "location": location,
                "url": url,
                "source": source["name"],
            }
        )
    return events


def collect_ics(source: dict) -> List[dict]:
    headers = {"User-Agent": "cincy-car-events-bot/1.0 (github actions)"}
    r = requests.get(source["url"], headers=headers, timeout=30)
    r.raise_for_status()

    cal = Calendar.from_ical(r.content)
    out = []

    for component in cal.walk():
        if component.name != "VEVENT":
            continue

        title = clean_ws(str(component.get("summary", "")))
        loc = clean_ws(str(component.get("location", "")))
        url = clean_ws(str(component.get("url", ""))) or source["url"]

        dtstart = component.get("dtstart")
        dtend = component.get("dtend")

        if not title or not dtstart:
            continue

        start_dt = dtstart.dt
        end_dt = dtend.dt if dtend else None

        if isinstance(start_dt, datetime) and start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc).astimezone(tz.gettz("America/New_York"))
        elif isinstance(start_dt, datetime):
            start_dt = start_dt.astimezone(tz.gettz("America/New_York"))
        else:
            start_dt = datetime.combine(start_dt, datetime.min.time()).replace(tzinfo=tz.gettz("America/New_York"))

        if end_dt:
            if isinstance(end_dt, datetime) and end_dt.tzinfo is None:
                end_dt = end_dt.replace(tzinfo=timezone.utc).astimezone(tz.gettz("America/New_York"))
            elif isinstance(end_dt, datetime):
                end_dt = end_dt.astimezone(tz.gettz("America/New_York"))
        else:
            end_dt = start_dt + timedelta(hours=2)

        out.append(
            {
                "title": title,
                "start_dt": start_dt,
                "end_dt": end_dt,
                "location": loc,
                "url": url,
                "source": source["name"],
            }
        )

    return out


def collect_facebook_page_events(source: dict, diagnostics: Optional[dict] = None) -> List[dict]:
    targets = source.get("_facebook_targets") or load_facebook_targets()
    pages = targets.get("page", [])
    events = collect_facebook_events_from_pages(pages, diagnostics=diagnostics)
    source_name = source.get("name", "Facebook Page Events")
    for e in events:
        e["source"] = source_name
    return events


# -------------------------
# NEW: Search the web (SerpAPI) + parse schema.org Event
# -------------------------
def serpapi_debug_slug(value: str) -> str:
    raw = clean_ws(value).lower()
    raw = re.sub(r"[^a-z0-9]+", "-", raw).strip("-")
    return raw[:80] or "query"


def save_serpapi_debug_artifact(slug: str, content: str) -> Optional[str]:
    if not RUN_ARTIFACT_DIR:
        return None
    try:
        os.makedirs(RUN_ARTIFACT_DIR, exist_ok=True)
        path = os.path.join(RUN_ARTIFACT_DIR, f"serpapi_debug_{slug}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path
    except Exception as ex:
        log(f"⚠️ Failed writing SerpAPI debug artifact for slug={slug}: {ex}")
        return None


def log_serpapi_request_meta(
    *,
    query_name: str,
    q: str,
    engine: str,
    location: str,
    htichips: str,
    start: int,
    payload: Optional[dict] = None,
) -> None:
    payload = payload or {}
    meta = payload.get("search_metadata") if isinstance(payload, dict) else {}
    status = clean_ws(str((meta or {}).get("status", "")))
    error_value = clean_ws(str(payload.get("serpapi_error") or payload.get("error") or "")) if isinstance(payload, dict) else ""
    log(
        "ℹ️ SerpAPI request: "
        f"query_name={query_name} q={q} location={location} engine={engine} "
        f"htichips={htichips} start={start} status={status or 'unknown'} "
        f"error={error_value or 'none'}"
    )


def serpapi_request_with_retry(
    params: dict,
    *,
    query_name: str,
    q: str,
    engine: str,
    location: str,
    htichips: str,
    start: int,
    output_mode: str = "json",
) -> Optional[requests.Response]:
    url = "https://serpapi.com/search"
    req_params = dict(params)
    if output_mode == "html":
        req_params["output"] = "html"

    for attempt in range(3):
        try:
            resp = requests.get(url, params=req_params, headers=DEFAULT_HTTP_HEADERS, timeout=45)
            if resp.status_code in (429, 500, 502, 503, 504):
                time.sleep((2 ** attempt) * 0.8)
                continue
            return resp
        except Exception as ex:
            if attempt == 2:
                log(
                    f"⚠️ SerpAPI request failed permanently: query_name={query_name} "
                    f"start={start} error={ex}"
                )
                return None
            time.sleep((2 ** attempt) * 0.8)
    return None


def serpapi_search(
    query: str,
    max_results: int = 20,
    page_size: int = 20,
    return_payload: bool = False,
    max_pages: int = 1,
    *,
    engine: str = "google",
    query_name: str = "serpapi",
    location: Optional[str] = None,
    gl: Optional[str] = None,
    hl: Optional[str] = None,
    htichips: Optional[str] = None,
):
    """Return SerpAPI links (and payload rows) with bounded pagination and retry/backoff."""
    if not SERPAPI_API_KEY:
        log("⚠️ Skipping SerpAPI search: missing SERPAPI_API_KEY.")
        return ([], []) if return_payload else []

    page_size = max(10, min(page_size, 100))
    target = max(1, max_results)
    max_pages = max(1, max_pages)

    links: List[str] = []
    payload_rows: List[dict] = []
    start = 0
    page_count = 0

    req_location = clean_ws(location or SERPAPI_LOCATION)
    req_gl = clean_ws(gl or SERPAPI_GL or "us")
    req_hl = clean_ws(hl or SERPAPI_HL or "en")
    req_htichips = clean_ws(htichips or "")

    while len(links) < target and page_count < max_pages:
        params = {
            "engine": engine,
            "q": query,
            "api_key": SERPAPI_API_KEY,
            "num": min(page_size, target - len(links)),
            "start": start,
            "hl": req_hl,
            "gl": req_gl,
        }
        if req_location:
            params["location"] = req_location
        if req_htichips:
            params["htichips"] = req_htichips

        response = serpapi_request_with_retry(
            params,
            query_name=query_name,
            q=query,
            engine=engine,
            location=req_location,
            htichips=req_htichips,
            start=start,
            output_mode="json",
        )
        if response is None:
            break

        if response.status_code != 200:
            log(
                f"⚠️ SerpAPI non-200: query_name={query_name} q={query} "
                f"engine={engine} status={response.status_code} detail={(response.text or '')[:220]}"
            )
            break

        data = {}
        try:
            data = response.json() or {}
        except Exception as ex:
            log(f"⚠️ SerpAPI JSON decode failed for query_name={query_name} q={query}: {ex}")
            data = {}

        log_serpapi_request_meta(
            query_name=query_name,
            q=query,
            engine=engine,
            location=req_location,
            htichips=req_htichips,
            start=start,
            payload=data,
        )

        if engine == "google_events":
            rows = data.get("events_results") or []
        else:
            rows = data.get("organic_results") or []

        if not rows:
            # One-time debug capture for zero-result responses.
            debug_resp = serpapi_request_with_retry(
                params,
                query_name=query_name,
                q=query,
                engine=engine,
                location=req_location,
                htichips=req_htichips,
                start=start,
                output_mode="html",
            )
            debug_text = ""
            if debug_resp is not None:
                raw_html_url = clean_ws(debug_resp.url)
                debug_text = (
                    f"query_name={query_name}\nq={query}\nengine={engine}\nlocation={req_location}\n"
                    f"htichips={req_htichips}\nstart={start}\nstatus={debug_resp.status_code}\n"
                    f"raw_html_file_url={raw_html_url}\n\n"
                    f"body_head={(debug_resp.text or '')[:5000]}"
                )
            else:
                debug_text = (
                    f"query_name={query_name}\nq={query}\nengine={engine}\nlocation={req_location}\n"
                    f"htichips={req_htichips}\nstart={start}\nraw_html_file_url=unavailable"
                )
            debug_path = save_serpapi_debug_artifact(serpapi_debug_slug(f"{query_name}-{query}"), debug_text)
            if debug_path:
                log(f"ℹ️ SerpAPI zero-result debug saved: {debug_path}")
            break

        for item in rows:
            if engine == "google_events":
                link = clean_ws(item.get("link") or item.get("event_link") or item.get("website") or "")
            else:
                link = clean_ws(item.get("link") or "")
            links.append(link)
            payload_rows.append(item)

        page_count += 1
        if len(rows) < params["num"]:
            break
        start += 10
        time.sleep(0.2)

    dedup_links = []
    dedup_rows = []
    seen = set()
    for link, item in zip(links, payload_rows):
        dedupe_key = link or clean_ws(str(item.get("title") or item.get("name") or ""))
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        dedup_links.append(link)
        dedup_rows.append(item)
        if len(dedup_rows) >= target:
            break

    if return_payload:
        return dedup_links, dedup_rows
    return dedup_links


def collect_serpapi_candidate_urls(result_item: dict) -> List[str]:
    """Extract candidate URLs from known SerpAPI item fields."""
    if not isinstance(result_item, dict):
        return []

    out: List[str] = []

    def _append_url(value):
        if isinstance(value, str):
            u = clean_ws(value)
            if u:
                out.append(u)
        elif isinstance(value, dict):
            for k in ("link", "redirect_link", "source", "url"):
                _append_url(value.get(k))
        elif isinstance(value, list):
            for v in value:
                _append_url(v)

    for key in ("link", "redirect_link", "source", "url"):
        _append_url(result_item.get(key))

    for key in ("inline_images", "sitelinks", "rich_snippet", "about_this_result"):
        _append_url(result_item.get(key))

    deduped = []
    seen = set()
    for u in out:
        nu = decode_serpapi_candidate_url(u)
        if not nu or nu in seen:
            continue
        seen.add(nu)
        deduped.append(nu)
    return deduped


def extract_facebook_event_urls_from_serpapi_result(result_item: dict) -> List[str]:
    urls = []
    seen = set()
    for candidate in collect_serpapi_candidate_urls(result_item):
        normalized = normalize_facebook_event_url(candidate)
        if not normalized:
            continue
        if not re.fullmatch(r"https://www\.facebook\.com/events/\d+/", normalized):
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        urls.append(normalized)
    return urls


def build_organizer_seed_terms(facebook_targets: Optional[Dict[str, List[dict]]] = None) -> List[str]:
    targets = facebook_targets or load_facebook_targets()
    seeds: List[str] = []
    for row in targets.get("non_facebook", []):
        if not row.get("enabled", True):
            continue
        label = clean_ws(row.get("label", ""))
        domain = clean_ws(row.get("organizer_domain", ""))
        url = clean_ws(row.get("page_url", ""))
        if not domain and url:
            domain = clean_ws(urlparse(url).netloc.replace("www.", ""))
        if label:
            seeds.append(label)
        if domain:
            seeds.append(domain)
    return list(dict.fromkeys([s for s in seeds if s]))


def build_serpapi_discovery_queries(cfg: dict, for_facebook: bool = False, limit: int = 16, organizer_terms: Optional[List[str]] = None) -> List[str]:
    """Build rotating car-focused SerpAPI queries across geos, time windows, and event types."""
    now = datetime.now(tz=tz.gettz("America/New_York"))
    y = now.year
    month_name = now.strftime("%B")
    next_month_name = (now + timedelta(days=32)).strftime("%B")

    geos = [
        "Cincinnati OH", "Northern Kentucky", "NKY", "Mason OH", "West Chester OH", "Loveland OH",
        "Dayton OH", "Columbus OH", "Louisville KY", "Indianapolis IN", "Lexington KY", "Springfield OH"
    ]
    event_types = [
        '"cars and coffee"', '"cars & coffee"', '"car meet"', '"car meetup"', '"cruise-in"', '"cruise night"',
        '"car show"', 'autocross', '"track day"', 'HPDE', 'rally', '"driving tour"', '"road rally"', 'concourse',
        '"supercar"', '"exotic car"', '"test and tune"', '"dyno day"'
    ]
    keywords = ["JDM", "Euro", "muscle", "tuner", '"Porsche Club"', '"BMW CCA"', '"Corvette club"', '"Mustang club"']
    times = [str(y), "this weekend", f"{month_name} {y}", f"{next_month_name} {y}"]

    platform_sites = [
        "eventbrite.com", "motorsportreg.com", "trackrabbit.com", "scca.com", "meetup.com", "facebook.com/events"
    ]

    out = []
    idx = 0
    while len(out) < limit:
        geo = geos[idx % len(geos)]
        et = event_types[idx % len(event_types)]
        kw = keywords[idx % len(keywords)]
        tv = times[idx % len(times)]
        site = platform_sites[idx % len(platform_sites)]
        base = f'site:{site} ({geo}) ({et} OR {kw}) ({tv})'
        if for_facebook:
            base = f'site:facebook.com/events ({geo}) ({et} OR {kw}) ({tv})'
        out.append(base)
        idx += 1

    configured = cfg.get("discovery", {}).get("facebook_event_queries" if for_facebook else "web_discovery_queries", []) or []
    configured = [clean_ws(q) for q in configured if clean_ws(q)]
    organizer_terms = organizer_terms or []
    organizer_queries: List[str] = []
    if for_facebook:
        organizer_queries.extend([
            "site:facebook.com/events (cincinnati OR \"Cincinnati, OH\") (cars OR \"car show\" OR \"cars and coffee\" OR rally OR meet)",
            "site:facebook.com/events (Ohio OR \"Northern Kentucky\" OR NKY) (cars OR \"car show\" OR \"cars and coffee\" OR rally)",
            "site:facebook.com/events (regional rally OR driving tour) (ohio OR kentucky OR indiana OR tennessee OR michigan)",
        ])
        for term in organizer_terms[:40]:
            organizer_queries.append(f"({term}) site:facebook.com/events")

    merged = configured + organizer_queries + out
    deduped = list(dict.fromkeys([clean_ws(q) for q in merged if clean_ws(q)]))
    return deduped[:limit]


def parse_event_platform_fallback(page_url: str, html: str) -> List[dict]:
    """Lightweight fallback parser for pages where schema.org Event is missing/incomplete."""
    soup = BeautifulSoup(html, "html.parser")
    title = clean_ws((soup.title.get_text(" ") if soup.title else "") or "")
    text = clean_ws(soup.get_text(" "))
    if not title and not text:
        return []

    if not any(k in (title + " " + text).lower() for k in ["car", "cars and coffee", "autocross", "track day", "rally"]):
        return []

    dt = None
    m = re.search(r"(\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}[^.]{0,60})", text, re.IGNORECASE)
    if m:
        dt = parse_dt(m.group(1))
    if not dt:
        return []

    loc = ""
    for patt in [r"\b(Cincinnati|Northern Kentucky|NKY|Mason|West Chester|Loveland|Dayton|Columbus|Louisville|Indianapolis)\b"]:
        m2 = re.search(patt, text, re.IGNORECASE)
        if m2:
            loc = m2.group(1)
            break

    return [{
        "title": title[:180] or "Car Event",
        "start_dt": dt,
        "end_dt": dt + timedelta(hours=2),
        "location": loc,
        "url": page_url,
    }]
def verify_serpapi_or_raise() -> None:
    """
    Fail fast if SERPAPI_API_KEY is missing or invalid.
    Runs a lightweight validation query to confirm key usability.
    """
    if not SERPAPI_API_KEY:
        raise RuntimeError("SERPAPI_API_KEY is required but missing.")

    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google",
        "q": "cincinnati cars and coffee",
        "api_key": SERPAPI_API_KEY,
        "num": 3,
        "hl": "en",
        "gl": "us",
    }

    r = requests.get(url, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(f"SERPAPI_API_KEY validation failed: HTTP {r.status_code} :: {(r.text or '')[:250]}")

    data = r.json()
    err = data.get("error")
    if err:
        raise RuntimeError(f"SERPAPI_API_KEY validation failed: {err}")

    organic = data.get("organic_results") or []
    if not isinstance(organic, list):
        raise RuntimeError("SERPAPI validation failed: unexpected response payload.")

    log(f"✅ SERPAPI_API_KEY verified; validation returned {len(organic)} organic results.")
def parse_date_from_text(text: str) -> Optional[datetime]:
    raw = clean_ws(text)
    if not raw:
        return None
    m = re.search(
        r"((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:,\s*\d{4})?)",
        raw,
        re.IGNORECASE,
    )
    if m:
        return parse_dt(m.group(1))
    return parse_dt(raw)


def parse_facebook_serpapi_result(item: dict, source_name: str = "facebook:discovered") -> Optional[dict]:
    link = normalize_facebook_event_url(clean_ws(item.get("link") or ""))
    if "facebook.com/events/" not in link:
        return None

    title = clean_ws(item.get("title") or "")
    snippet = clean_ws(item.get("snippet") or "")
    text_blob = clean_ws(f"{title} {snippet}")
    start_dt = parse_date_from_text(text_blob)
    if not start_dt:
        return None

    location = ""
    loc_match = re.search(r"(?:at|in)\s+([^|·\-]{3,80})", snippet, re.IGNORECASE)
    if loc_match:
        location = clean_ws(loc_match.group(1))

    event_id = extract_facebook_event_id(link)
    return {
        "title": title or "Facebook Event",
        "start_dt": start_dt,
        "end_dt": start_dt + timedelta(hours=2),
        "location": location,
        "url": link,
        "source": source_name,
        "facebook_event_id": event_id or "",
    }


def maybe_enrich_facebook_event_via_graph(ev: dict, graph_state: Optional[dict] = None) -> dict:
    enrich_enabled = clean_ws(os.getenv("ENABLE_FACEBOOK_GRAPH_ENRICH", "")).lower() in ("1", "true", "yes")
    event_id = clean_ws(ev.get("facebook_event_id", ""))
    if not enrich_enabled or not event_id or not facebook_graph_usable():
        return ev
    if graph_state and graph_state.get("token_expired"):
        return ev
    try:
        g = fetch_facebook_event_via_graph(event_id)
    except Exception:
        return ev
    if not g:
        return ev
    ev.update({
        "title": g.get("title", ev.get("title", "")),
        "start_dt": g.get("start_dt", ev.get("start_dt")),
        "end_dt": g.get("end_dt", ev.get("end_dt")),
        "location": g.get("location", ev.get("location", "")),
        "url": g.get("url", ev.get("url", "")),
    })
    return ev


def parse_bool_env(name: str, default: bool = False) -> bool:
    raw = clean_ws(os.getenv(name, ""))
    if not raw:
        return default
    return raw.lower() not in {"0", "false", "no", "off", "n"}


def parse_int_env(name: str, default: int) -> int:
    raw = clean_ws(os.getenv(name, ""))
    if not raw:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def enrich_facebook_group_identity(group_key: str, max_results: int = 8) -> dict:
    """Find better group identity hints from search results (name and slug)."""
    query = f'site:facebook.com/groups "{group_key}"'
    group_name = ""
    group_slug = ""

    links, payload_rows = serpapi_search(query, max_results=max_results, return_payload=True, max_pages=1)
    rows = []
    for link, row in zip(links, payload_rows):
        rows.append({"url": link, "result": row})

    for row in rows:
        for candidate in collect_serpapi_candidate_urls(row.get("result", {})) + [row.get("url", "")]:
            parsed = urlparse(candidate if "://" in candidate else f"https://{candidate}")
            host = (parsed.netloc or "").lower()
            if "facebook.com" not in host:
                continue
            path = clean_ws(parsed.path or "")
            m = re.search(r"/groups/([^/?#]+)/?", path, re.IGNORECASE)
            if m:
                token = clean_ws(m.group(1))
                if token and token != group_key:
                    group_slug = token
                    break
        title = clean_ws((row.get("result") or {}).get("title", ""))
        if title and "facebook" in title.lower():
            title = clean_ws(re.split(r"\s*[|·-]\s*Facebook", title, maxsplit=1, flags=re.IGNORECASE)[0])
        if title and len(title) > 2:
            group_name = title
        if group_slug and group_name:
            break

    return {
        "group_name": group_name,
        "group_slug": group_slug,
        "group_url_canonical": f"https://www.facebook.com/groups/{group_key}/",
    }


def build_facebook_group_serpapi_queries(group_key: str, group_name: str = "", group_slug: str = "") -> List[str]:
    terms = [group_key]
    if group_slug and group_slug not in terms:
        terms.append(group_slug)

    queries = [f'site:facebook.com/events "{group_key}"']
    if group_name:
        queries.append(f'site:facebook.com/events "{group_name}"')

    for t in terms:
        queries.append(f'site:facebook.com "{t}" ("facebook.com/events/" OR "event.php?eid=")')
    if group_name:
        queries.append(f'site:facebook.com "{group_name}" ("facebook.com/events/" OR "event.php?eid=")')

    upcoming_clause = " OR ".join([f'"{t}"' for t in terms + ([group_name] if group_name else [])])
    queries.append(f'site:facebook.com ({upcoming_clause}) "Upcoming events"')

    deduped = []
    seen = set()
    for q in queries:
        if q in seen:
            continue
        seen.add(q)
        deduped.append(q)
    return deduped[:5]


def collect_facebook_group_event_urls_serpapi(group_url: str, group_key: str, cfg: dict) -> Tuple[List[dict], Optional[str], dict]:
    max_results = max(5, parse_int_env("FACEBOOK_GROUPS_SERP_MAX_RESULTS", 30))
    max_pages = max(1, parse_int_env("FACEBOOK_GROUPS_SERP_MAX_PAGES", 2))
    enable_enrichment = parse_bool_env("FACEBOOK_GROUPS_ENABLE_ENRICHMENT", True)

    enrich = {"group_name": "", "group_slug": "", "group_url_canonical": f"https://www.facebook.com/groups/{group_key}/"}
    if enable_enrichment:
        try:
            enrich = enrich_facebook_group_identity(group_key)
        except Exception as ex:
            log(f"⚠️ Group enrichment failed for {group_key}: {ex}")

    queries = build_facebook_group_serpapi_queries(
        group_key,
        group_name=enrich.get("group_name", ""),
        group_slug=enrich.get("group_slug", ""),
    )

    found_rows: List[dict] = []
    for q in queries:
        links, payload_rows = serpapi_search(q, max_results=max_results, return_payload=True, max_pages=max_pages)
def build_facebook_group_serpapi_queries(group_key: str) -> List[str]:
    return [
        f'site:facebook.com/events "groups/{group_key}"',
        f'site:facebook.com/events "{group_key}"',
        f'site:facebook.com "{group_key}" "event"',
    ]


def collect_facebook_group_event_urls_serpapi(group_url: str, group_key: str, cfg: dict) -> Tuple[List[dict], Optional[str], dict]:
    max_results = int(cfg.get("discovery", {}).get("facebook_group_serpapi_max_results", 20) or 20)
    queries = build_facebook_group_serpapi_queries(group_key)
    enrich = {"group_name": clean_ws(group_key.replace("-", " ")).title() if group_key else "", "serp_urls": 0}

    found_rows: List[dict] = []
    for q in queries:
        links, payload_rows = serpapi_search(q, max_results=max_results, return_payload=True)
        for link, row in zip(links, payload_rows):
            found_rows.append({"url": link, "result": row, "query": q, "group_url": group_url, "group_key": group_key})
        time.sleep(0.2)

    serp_urls_seen = set()
    event_rows: List[dict] = []
    event_seen = set()

    for row in found_rows:
        for candidate in collect_serpapi_candidate_urls(row.get("result", {})) + [row.get("url", "")]:
            decoded = decode_serpapi_candidate_url(candidate)
            if not decoded:
                continue
            serp_urls_seen.add(decoded)

        candidate_urls = extract_facebook_event_urls_from_serpapi_result(row.get("result", {}))
        if not candidate_urls:
            fallback = normalize_facebook_event_url(row.get("url", ""))
            candidate_urls = [fallback] if fallback else []

        for normalized in candidate_urls:
            if normalized in event_seen:
                continue
            event_seen.add(normalized)
            item = dict(row)
            item["url"] = normalized
            event_rows.append(item)

    enrich["serp_urls"] = len(serp_urls_seen)
    return event_rows, None, enrich


def collect_facebook_group_events_serpapi(source: dict, cfg: dict, url_cache: Dict[str, dict], diagnostics: Optional[dict] = None) -> List[dict]:
    diagnostics = diagnostics if diagnostics is not None else {}
    diagnostics.setdefault("raw_candidates", 0)
    diagnostics.setdefault("parse_failures", 0)

    if not SERPAPI_API_KEY:
        diagnostics["reason"] = "disabled_missing_serpapi_key"
        log("ℹ️ SerpAPI disabled (missing SERPAPI_API_KEY); skipping Facebook Group Events collector.")
        return []

    targets = source.get("_facebook_targets") or load_facebook_targets()
    groups = [g for g in targets.get("group", []) if g.get("enabled", True)]

    groups_max = parse_int_env("FACEBOOK_GROUPS_MAX", 0)
    if groups_max > 0:
        groups = groups[:groups_max]
    targets = load_facebook_targets()
    groups = [g for g in targets.get("group", []) if g.get("enabled", True)]
    limit_raw = clean_ws(os.getenv("FACEBOOK_GROUP_SERPAPI_DEBUG_LIMIT", ""))
    limit = int(limit_raw) if limit_raw.isdigit() else 0
    if limit > 0:
        groups = groups[:limit]

    if not groups:
        diagnostics["reason"] = "disabled_no_groups_configured"
        log("ℹ️ No Facebook groups configured; skipping Facebook Group Events collector.")
        return []

    out: List[dict] = []
    seen_urls_global = set()
    groups_with_errors: List[str] = []
    groups_queried = 0
    total_serp_urls = 0
    total_event_urls_discovered = 0
    now = datetime.now(tz=tz.gettz("America/New_York"))
    graph_state = {"token_expired": False}

    for g in groups:
        group_url = clean_ws(g.get("page_url", ""))
        group_key = extract_facebook_group_key(group_url)
        if not group_key:
            groups_with_errors.append(group_url)
            log(f"⚠️ Group discovery skipped; unable to extract group key from URL: {group_url}")
            continue

        serp_urls_count = 0
        event_urls_count = 0
        parsed_count = 0
        group_name = ""

        try:
            event_rows, _, enrich = collect_facebook_group_event_urls_serpapi(group_url, group_key, cfg)
            groups_queried += 1
            serp_urls_count = int(enrich.get("serp_urls", 0))
            group_name = clean_ws(enrich.get("group_name", ""))
            total_serp_urls += serp_urls_count

            for row in event_rows:
                u = clean_ws(row.get("url", ""))
                if not u or u in seen_urls_global:
                    continue
                seen_urls_global.add(u)
                event_urls_count += 1
                diagnostics["raw_candidates"] += 1

                cached = url_cache.get(u)
                if cached:
                    try:
                        last = datetime.fromisoformat(cached.get("fetched_at_iso"))
                        if (now - last) < timedelta(hours=24) and cached.get("event"):
                            e = cached["event"]
                            out.append(
                                {
                                    "title": e.get("title", ""),
                                    "start_dt": datetime.fromisoformat(e["start_iso"]),
                                    "end_dt": datetime.fromisoformat(e["end_iso"]),
                                    "location": e.get("location", ""),
                                    "url": e.get("url", u),
                                    "source": "facebook_group_serpapi",
                                    "facebook_event_id": e.get("facebook_event_id", ""),
                                    "source_group_url": group_url,
                                    "source_group_key": group_key,
                                    "source_group_name": group_name,
                                }
                            )
                            parsed_count += 1
                            continue
                    except Exception:
                        pass

                ev = parse_facebook_serpapi_result(row.get("result", {}), source_name="facebook_group_serpapi")
                if not ev:
                    diagnostics["parse_failures"] += 1
                    url_cache[u] = {"fetched_at_iso": now.isoformat(), "event": None}
                    continue

                ev = maybe_enrich_facebook_event_via_graph(ev, graph_state=graph_state)
                ev["source"] = "facebook_group_serpapi"
                ev["source_group_url"] = group_url
                ev["source_group_key"] = group_key
                ev["source_group_name"] = group_name
                out.append(ev)
                parsed_count += 1
                url_cache[u] = {
                    "fetched_at_iso": now.isoformat(),
                    "event": {
                        "title": ev["title"],
                        "start_iso": ev["start_dt"].isoformat(),
                        "end_iso": ev["end_dt"].isoformat(),
                        "location": ev.get("location", ""),
                        "url": ev.get("url", u),
                        "source": "facebook_group_serpapi",
                        "facebook_event_id": ev.get("facebook_event_id", ""),
                    },
                }
            total_event_urls_discovered += event_urls_count
        except Exception as ex:
            groups_with_errors.append(group_key)
            log_exception_context(f"Group discovery failed for {group_key}", ex)

        log(f"🔎 Group discovery: {group_key} -> serp_urls={serp_urls_count} event_urls={event_urls_count} parsed={parsed_count}")

    log("📘 Facebook group collector summary:")
    log(f"   groups configured: {len(targets.get('group', []))}")
    log(f"   groups queried: {groups_queried}")
    log(f"   total serp urls: {total_serp_urls}")
    log(f"   total event urls discovered: {total_event_urls_discovered}")
    log(f"   total events parsed: {len(out)}")
    if groups_with_errors:
        log(f"   groups with errors: {groups_with_errors}")
    else:
        log("   groups with errors: none")

    if diagnostics.get("reason") is None and not out:
        diagnostics["reason"] = "no_results_from_search"

    return out


def collect_facebook_events_serpapi_discovery(cfg: dict, url_cache: Dict[str, dict], diagnostics: Optional[dict] = None) -> List[dict]:
    """
    Discover FB event URLs via SerpAPI and parse metadata from SerpAPI payload.
    Optional Graph enrichment is controlled by ENABLE_FACEBOOK_GRAPH_ENRICH=1.
    """
    diagnostics = diagnostics if diagnostics is not None else {}
    diagnostics.setdefault("raw_candidates", 0)
    diagnostics.setdefault("parse_failures", 0)

    event_rows = collect_facebook_event_urls_serpapi(cfg)
    if not event_rows:
        diagnostics["reason"] = "no_results_from_search"
        return []

    out: List[dict] = []
    now = datetime.now(tz=tz.gettz("America/New_York"))
    graph_state = {"token_expired": False}

    for row in event_rows:
        u = clean_ws(row.get("url", ""))
        if not u:
            continue
        diagnostics["raw_candidates"] += 1

        cached = url_cache.get(u)
        if cached:
            try:
                last = datetime.fromisoformat(cached.get("fetched_at_iso"))
                if (now - last) < timedelta(hours=24) and cached.get("event"):
                    e = cached["event"]
                    out.append(
                        {
                            "title": e.get("title", ""),
                            "start_dt": datetime.fromisoformat(e["start_iso"]),
                            "end_dt": datetime.fromisoformat(e["end_iso"]),
                            "location": e.get("location", ""),
                            "url": e.get("url", u),
                            "source": e.get("source", "facebook:discovered"),
                            "facebook_event_id": e.get("facebook_event_id", ""),
                        }
                    )
                    continue
            except Exception:
                pass

        ev = parse_facebook_serpapi_result(row.get("result", {}), source_name="facebook:discovered")
        if not ev:
            parsed_page, fail_reason = parse_facebook_event_page(u, log)
            if parsed_page:
                ev = {
                    "title": parsed_page.get("title", ""),
                    "start_dt": parsed_page.get("start_dt"),
                    "end_dt": parsed_page.get("end_dt"),
                    "location": clean_ws(f"{parsed_page.get('location', '')} {parsed_page.get('address', '')}"),
                    "url": parsed_page.get("canonical_url") or parsed_page.get("url") or u,
                    "source": "facebook:discovered",
                    "host": parsed_page.get("host", ""),
                }
            else:
                diagnostics["parse_failures"] += 1
                FACEBOOK_COVERAGE["urls_failed"] = int(FACEBOOK_COVERAGE.get("urls_failed", 0) or 0) + 1
                FACEBOOK_COVERAGE["failure_reasons"][fail_reason] += 1
                url_cache[u] = {"fetched_at_iso": now.isoformat(), "event": None, "failure_reason": fail_reason}
                continue

        ev = maybe_enrich_facebook_event_via_graph(ev, graph_state=graph_state)
        FACEBOOK_COVERAGE["urls_parsed"] = int(FACEBOOK_COVERAGE.get("urls_parsed", 0) or 0) + 1
        out.append(ev)
        url_cache[u] = {
            "fetched_at_iso": now.isoformat(),
            "event": {
                "title": ev["title"],
                "start_iso": ev["start_dt"].isoformat(),
                "end_iso": ev["end_dt"].isoformat(),
                "location": ev.get("location", ""),
                "url": ev.get("url", u),
                "source": ev.get("source", "facebook:discovered"),
                "facebook_event_id": ev.get("facebook_event_id", ""),
            },
        }

    if diagnostics.get("reason") is None and not out:
        diagnostics["reason"] = "parsing_schema_changed"

    log(f"   Discovered+parsed {len(out)} Facebook events from SerpAPI URLs.")
    return out


def collect_facebook_event_urls_serpapi(cfg: dict) -> List[dict]:
    """Discover public FB event URLs via dynamic SerpAPI queries and keep raw rows."""
    if not SERPAPI_API_KEY:
        log("ℹ️ SerpAPI disabled (missing SERPAPI_API_KEY); skipping FB URL discovery.")
        return []

    organizer_terms = build_organizer_seed_terms(load_facebook_targets())
    queries = build_serpapi_discovery_queries(cfg, for_facebook=True, limit=24, organizer_terms=organizer_terms)
    found_rows: List[dict] = []

    for q in queries:
        links = []
        payload_rows = []
        for attempt in range(3):
            try:
                links, payload_rows = serpapi_search(q, max_results=25, return_payload=True)
                break
            except Exception as ex:
                if attempt == 2:
                    log(f"⚠️ SerpAPI discovery query failed: {q} :: {ex}")
                time.sleep((2 ** attempt) * 0.8)
        FACEBOOK_COVERAGE.setdefault("serp_queries", []).append({"query": q, "result_count": len(links)})
        log(f"ℹ️ SerpAPI FB query results: count={len(links)} query={q}")
        for link, row in zip(links, payload_rows):
            found_rows.append({"url": link, "result": row, "query": q})
        time.sleep(0.2)

    event_rows = []
    seen = set()
    for r in found_rows:
        result_item = r.get("result", {}) if isinstance(r, dict) else {}
        candidate_urls = extract_facebook_event_urls_from_serpapi_result(result_item)
        if not candidate_urls:
            fallback = normalize_facebook_event_url(r.get("url", ""))
            candidate_urls = [fallback] if fallback else []
        for normalized in candidate_urls:
            if normalized in seen:
                continue
            seen.add(normalized)
            item = dict(r)
            item["url"] = normalized
            event_rows.append(item)

    FACEBOOK_COVERAGE["urls_discovered"] = int(FACEBOOK_COVERAGE.get("urls_discovered", 0) or 0) + len(event_rows)
    log(f"   SerpAPI found {len(event_rows)} Facebook event URLs (pre-parse).")
    return event_rows


def parse_schema_org_events_from_html(page_url: str, html: str) -> List[dict]:
    """Extract schema.org Event objects from JSON-LD, including nested @graph/list payloads."""
    soup = BeautifulSoup(html, "html.parser")
    canonical_tag = soup.select_one('link[rel="canonical"]')
    canonical_url = clean_ws(canonical_tag.get("href", "")) if canonical_tag else page_url
    scripts = soup.select('script[type="application/ld+json"]')
    out: List[dict] = []

    def iter_dict_nodes(node):
        if isinstance(node, dict):
            yield node
            for v in node.values():
                yield from iter_dict_nodes(v)
        elif isinstance(node, list):
            for item in node:
                yield from iter_dict_nodes(item)

    def location_from_obj(loc) -> str:
        if isinstance(loc, list):
            return clean_ws(" | ".join(location_from_obj(x) for x in loc if location_from_obj(x)))
        if isinstance(loc, dict):
            loc_name = clean_ws(loc.get("name") or "")
            addr = loc.get("address") or {}
            if isinstance(addr, dict):
                parts = [addr.get("streetAddress"), addr.get("addressLocality"), addr.get("addressRegion"), addr.get("postalCode")]
                addr_str = clean_ws(" ".join(p for p in parts if p))
            else:
                addr_str = clean_ws(str(addr))
            return clean_ws(" ".join(x for x in [loc_name, addr_str] if x))
        return clean_ws(str(loc))

    for s in scripts:
        txt = s.get_text(strip=True)
        if not txt:
            continue
        try:
            data = json.loads(txt)
        except Exception:
            continue

        for obj in iter_dict_nodes(data):
            t = obj.get("@type")
            types = [str(x).lower() for x in t] if isinstance(t, list) else [str(t).lower()]
            if "event" not in types:
                continue

            name = clean_ws(obj.get("name") or obj.get("summary") or "")
            start_dt = parse_dt(clean_ws(obj.get("startDate") or ""))
            end_raw = clean_ws(obj.get("endDate") or "")
            end_dt = parse_dt(end_raw) if end_raw else None
            if not name or not start_dt:
                continue

            out.append({
                "title": name,
                "start_dt": start_dt,
                "end_dt": end_dt or (start_dt + timedelta(hours=2)),
                "location": location_from_obj(obj.get("location") or {}),
                "url": clean_ws(obj.get("url") or canonical_url or page_url),
            })

    if not out:
        out = parse_event_platform_fallback(canonical_url or page_url, html)

    seen = set()
    deduped = []
    for e in out:
        k = (e["title"].lower().strip(), e["start_dt"].isoformat()[:16], clean_ws(e.get("url", "")))
        if k in seen:
            continue
        seen.add(k)
        deduped.append(e)
    return deduped

def parse_google_events_date_best_effort(date_text: str) -> Tuple[Optional[datetime], bool]:
    raw = clean_ws(date_text)
    if not raw:
        return None, True

    parsed = parse_dt(raw)
    if not parsed:
        return None, True

    estimated = bool(re.search(r"\b(today|tomorrow|this|next|weekend|week|month|am|pm|\d:\d)\b", raw.lower()) is None)
    if parsed.hour == 0 and parsed.minute == 0:
        parsed = parsed.replace(hour=9, minute=0, second=0, microsecond=0)
        estimated = True
    return parsed, estimated


def collect_serpapi_google_events(source: dict, diagnostics: Optional[dict] = None) -> List[dict]:
    diagnostics = diagnostics if diagnostics is not None else {}
    diagnostics.setdefault("raw_candidates", 0)
    diagnostics.setdefault("parse_failures", 0)

    if not SERPAPI_API_KEY:
        diagnostics["reason"] = "no_results_from_search"
        log("ℹ️ SerpAPI disabled; skipping google_events collector.")
        return []

    location = clean_ws(os.getenv("SERPAPI_LOCATION", SERPAPI_LOCATION or "Cincinnati, OH")) or "Cincinnati, OH"
    gl = clean_ws(os.getenv("SERPAPI_GL", SERPAPI_GL or "us")) or "us"
    hl = clean_ws(os.getenv("SERPAPI_HL", SERPAPI_HL or "en")) or "en"
    htichips = clean_ws(os.getenv("SERPAPI_EVENTS_DATE_FILTER", SERPAPI_EVENTS_DATE_FILTER or "date:month")) or "date:month"

    query_phrases = [
        "car show in Cincinnati OH",
        "cars and coffee in Cincinnati OH",
        "cruise in in Cincinnati OH",
        "autocross in Cincinnati OH",
        "rally in Ohio",
    ]

    out: List[dict] = []
    seen = set()

    for query in query_phrases:
        links, rows = serpapi_search(
            query,
            max_results=50,
            page_size=10,
            return_payload=True,
            max_pages=6,
            engine="google_events",
            query_name="serpapi_google_events",
            location=location,
            gl=gl,
            hl=hl,
            htichips=htichips,
        )

        diagnostics["raw_candidates"] += len(rows)

        for row in rows:
            title = clean_ws(row.get("title") or row.get("name") or "")
            date_text = clean_ws(row.get("date") or row.get("start_date") or row.get("when") or "")
            start_dt, time_is_estimated = parse_google_events_date_best_effort(date_text)
            if not title or not start_dt:
                diagnostics["parse_failures"] += 1
                continue

            end_dt = start_dt + timedelta(hours=2)
            venue = clean_ws(row.get("venue") or row.get("event_location") or "")
            address = clean_ws(row.get("address") or row.get("location") or "")
            location_text = clean_ws(" ".join([venue, address]))
            url = clean_ws(row.get("link") or row.get("event_link") or row.get("website") or "")
            ticket_info = row.get("ticket_info")
            thumbnail = clean_ws(row.get("thumbnail") or "")

            dedupe_key = (
                title.lower(),
                start_dt.isoformat()[:16],
                location_text.lower(),
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)

            out.append(
                {
                    "title": title,
                    "start_dt": start_dt,
                    "end_dt": end_dt,
                    "location": location_text,
                    "url": url,
                    "source": source.get("name", "Google Events (SerpAPI) [serpapi_google_events]"),
                    "time_is_estimated": time_is_estimated,
                    "ticket_info": ticket_info,
                    "thumbnail": thumbnail,
                }
            )

    if diagnostics.get("reason") is None and not out:
        diagnostics["reason"] = "no_results_from_search"

    return out


def collect_web_search_serpapi(source: dict, url_cache: Dict[str, dict], diagnostics: Optional[dict] = None) -> List[dict]:
    """SerpAPI discovery -> URL dedupe -> schema/fallback parse with caching."""
    diagnostics = diagnostics if diagnostics is not None else {}
    diagnostics.setdefault("raw_candidates", 0)
    diagnostics.setdefault("parse_failures", 0)
    diagnostics.setdefault("blocked_http", 0)

    max_results = int(source.get("max_results", 20))
    source_name = source.get("name", "web_search_serpapi")

    queries = [clean_ws(source.get("query", ""))] if clean_ws(source.get("query", "")) else []
    if not queries:
        cfg = load_yaml(CONFIG_PATH)
        queries = build_serpapi_discovery_queries(cfg, for_facebook=False, limit=12)

    if not SERPAPI_API_KEY:
        diagnostics["reason"] = "no_results_from_search"
        log(f"ℹ️ SerpAPI disabled; skipping {source_name}.")
        return []

    links: List[str] = []
    for q in queries[:20]:
        links.extend(serpapi_search(q, max_results=max_results))
        time.sleep(0.2)

    links = list(dict.fromkeys(clean_ws(u) for u in links if clean_ws(u)))
    diagnostics["raw_candidates"] = len(links)
    if not links:
        diagnostics["reason"] = "no_results_from_search"
        return []

    out: List[dict] = []
    now = datetime.now(tz=tz.gettz("America/New_York"))

    for u in links:
        cached = url_cache.get(u)
        if cached:
            try:
                last = datetime.fromisoformat(cached.get("fetched_at_iso"))
                if (now - last) < timedelta(hours=24) and cached.get("events"):
                    for e in cached["events"]:
                        out.append({
                            "title": e.get("title", ""),
                            "start_dt": datetime.fromisoformat(e["start_iso"]),
                            "end_dt": datetime.fromisoformat(e["end_iso"]),
                            "location": e.get("location", ""),
                            "url": e.get("url", u),
                            "source": source_name,
                        })
                    continue
            except Exception:
                pass

        try:
            if "eventbrite.com" in u and ("/d/" in u or "/e/" not in u):
                url_cache[u] = {"fetched_at_iso": now.isoformat(), "events": []}
                continue
            html = fetch_text(u)
            events = parse_schema_org_events_from_html(u, html)
            if not events:
                diagnostics["parse_failures"] += 1
            packed = []
            for e in events:
                e2 = {
                    "title": e["title"], "start_dt": e["start_dt"], "end_dt": e["end_dt"],
                    "location": e.get("location", ""), "url": e.get("url", u), "source": source_name,
                }
                out.append(e2)
                packed.append({
                    "title": e2["title"], "start_iso": e2["start_dt"].isoformat(), "end_iso": e2["end_dt"].isoformat(),
                    "location": e2["location"], "url": e2["url"],
                })
            url_cache[u] = {"fetched_at_iso": now.isoformat(), "events": packed}
            time.sleep(0.35)
        except requests.HTTPError as ex:
            code = getattr(getattr(ex, "response", None), "status_code", None)
            if code in (400, 403):
                diagnostics["blocked_http"] += 1
            diagnostics["parse_failures"] += 1
            log(f"⚠️ Web page parse failed: {u} :: {ex}")
            url_cache[u] = {"fetched_at_iso": now.isoformat(), "events": []}
        except Exception as ex:
            diagnostics["parse_failures"] += 1
            log(f"⚠️ Web page parse failed: {u} :: {ex}")
            url_cache[u] = {"fetched_at_iso": now.isoformat(), "events": []}

    if diagnostics.get("reason") is None and not out:
        if diagnostics.get("blocked_http", 0) > 0:
            diagnostics["reason"] = "blocked_http_403/400"
        elif diagnostics.get("parse_failures", 0) > 0:
            diagnostics["reason"] = "parsing_schema_changed"
        else:
            diagnostics["reason"] = "no_results_from_search"

    return out


def event_signature(ev: EventItem) -> str:
    title = re.sub(r"\s+", " ", clean_ws(ev.title).lower()).strip()
    when = clean_ws(ev.start_iso)[:16]
    place = re.sub(r"\s+", " ", clean_ws(ev.city_state or ev.location).lower()).strip()
    return f"{title}||{when}||{place}"


def log_counter_top(title: str, counter: Counter, top_n: int = 8) -> None:
    if not counter:
        return
    top = ", ".join(f"{k}={v}" for k, v in counter.most_common(top_n))
    log(f"📊 {title}: {top}")


def log_source_health_summary(source_run_stats: List[dict]) -> None:
    """Human-readable summary of what worked vs what did not."""
    working = [s for s in source_run_stats if s.get("status") == "ok" and int(s.get("collected", 0)) > 0]
    no_results = [s for s in source_run_stats if s.get("status") == "ok" and int(s.get("collected", 0)) == 0]
    skipped = [s for s in source_run_stats if s.get("status") == "skipped"]
    disabled = [s for s in source_run_stats if s.get("status") == "disabled"]
    failed = [s for s in source_run_stats if s.get("status") == "failed"]

    if working:
        msg = ", ".join(f"{s['name']}={s['collected']}" for s in working)
        log(f"🟢 Working (returned events): {msg}")
    else:
        log("🟢 Working (returned events): none")

    if no_results:
        msg = ", ".join(
            f"{s['name']}[{s.get('reason', 'no_results_from_search')}]"
            for s in no_results
        )
        log(f"🟡 Working but returned 0 events: {msg}")

    if failed:
        msg = ", ".join(f"{s['name']} ({s.get('error', 'unknown error')})" for s in failed)
        log(f"🔴 Failed sources: {msg}")

    if disabled:
        msg = ", ".join(f"{s['name']} ({s.get('reason', 'disabled')})" for s in disabled)
        log(f"⚫ Disabled sources: {msg}")

    if skipped:
        msg = ", ".join(s["name"] for s in skipped)
        log(f"⚪ Skipped sources: {msg}")
# -------------------------
# Normalize + filter + store
# -------------------------


def derive_zero_yield_reason(source_name: str, diagnostics: dict, source_drop_reasons: Counter) -> str:
    if diagnostics.get("reason"):
        return diagnostics["reason"]
    if diagnostics.get("blocked_http", 0) > 0:
        return "blocked_http_403/400"
    if source_drop_reasons.get("outside_window_past", 0) + source_drop_reasons.get("outside_window_future", 0) > 0:
        return "filtered_out_by_window"
    if source_drop_reasons.get("location_too_far_local", 0) + source_drop_reasons.get("location_too_far_rally", 0) > 0:
        return "filtered_out_by_distance"
    if any(k.startswith("non_automotive:") for k in source_drop_reasons):
        return "filtered_out_by_keywords"
    if diagnostics.get("parse_failures", 0) > 0:
        return "parsing_schema_changed"
    return "no_results_from_search"


def to_event_items(
    raw_events: List[dict],
    cfg: dict,
    geocache: Dict[str, dict],
    metrics: Optional[dict] = None,
    source_filter_stats: Optional[Dict[str, Counter]] = None,
) -> List[EventItem]:
    home_lat = cfg["home"]["lat"]
    home_lon = cfg["home"]["lon"]

    lookahead_days = int(cfg["filters"]["lookahead_days"])
    drop_past_days = int(cfg["filters"]["drop_past_days"])
    now_et = datetime.now(tz=tz.gettz("America/New_York"))
    window_start = now_et - timedelta(days=drop_past_days)
    window_end = None if lookahead_days < 0 else (now_et + timedelta(days=lookahead_days))

    local_max = float(cfg["filters"]["local_max_miles"])
    rally_max = float(cfg["filters"]["rally_max_miles"])

    out: List[EventItem] = []
    drop_reasons: Counter = Counter()
    non_auto_examples: List[str] = []

    for e in raw_events:
        title = clean_ws(e.get("title", ""))
        location = clean_ws(e.get("location", ""))
        url = clean_ws(e.get("url", ""))
        source = clean_ws(e.get("source", ""))
        start_dt: Optional[datetime] = e.get("start_dt")
        end_dt: Optional[datetime] = e.get("end_dt")

        source_counter = source_filter_stats.setdefault(source or "(unknown)", Counter()) if source_filter_stats is not None else None

        if not title:
            drop_reasons["missing_title"] += 1
            if source_counter is not None:
                source_counter["missing_title"] += 1
            continue
        if not start_dt:
            drop_reasons["missing_start_dt"] += 1
            if source_counter is not None:
                source_counter["missing_start_dt"] += 1
            continue
        if not end_dt:
            end_dt = start_dt + timedelta(hours=2)

        allowed, auto_reason = evaluate_automotive_focus_event(title, location, source, url, cfg)
        if not allowed:
            drop_reasons[f"non_automotive:{auto_reason}"] += 1
            if source_counter is not None:
                source_counter[f"non_automotive:{auto_reason}"] += 1
            if len(non_auto_examples) < 5:
                non_auto_examples.append(f"{title[:80]} ({auto_reason})")
            continue

        if start_dt < window_start:
            drop_reasons["outside_window_past"] += 1
            if source_counter is not None:
                source_counter["outside_window_past"] += 1
            continue
        if window_end is not None and start_dt > window_end:
            drop_reasons["outside_window_future"] += 1
            if source_counter is not None:
                source_counter["outside_window_future"] += 1
            continue

        city_state = guess_city_state(location)
        query = city_state or location
        latlon = geocode(query, geocache) if query else None
        lat = lon = None
        miles = None
        if latlon:
            lat, lon = latlon
            miles = miles_from_home(lat, lon, home_lat, home_lon)

        cat = categorize(title, location, cfg)

        if miles is not None:
            if cat == "local" and miles > local_max:
                drop_reasons["location_too_far_local"] += 1
                if source_counter is not None:
                    source_counter["location_too_far_local"] += 1
                continue
            if cat == "rally" and miles > rally_max:
                drop_reasons["location_too_far_rally"] += 1
                if source_counter is not None:
                    source_counter["location_too_far_rally"] += 1
                continue

        out.append(
            EventItem(
                title=title,
                start_iso=start_dt.isoformat(),
                end_iso=end_dt.isoformat(),
                location=location,
                city_state=city_state,
                url=url,
                source=source,
                category=cat,
                miles_from_cincy=round(miles, 1) if miles is not None else None,
                lat=lat,
                lon=lon,
                last_seen_iso=now_et_iso(),
            )
        )

    if metrics is not None:
        metrics["normalize_drop_reasons"] = dict(drop_reasons)
        metrics["non_automotive_examples"] = non_auto_examples
    log_counter_top("Normalize/filter drop reasons", drop_reasons)
    if non_auto_examples:
        log(f"📊 Non-automotive removed examples: {non_auto_examples}")
    return out


def dedupe_merge(
    existing: List[EventItem],
    incoming: List[EventItem],
    metrics: Optional[dict] = None,
) -> List[EventItem]:
    merged: Dict[str, EventItem] = {event_signature(e): e for e in existing}
    dedupe_reasons: Counter = Counter()

    for ev in incoming:
        k = event_signature(ev)
        if k in merged:
            cur = merged[k]
            dedupe_reasons["duplicate_signature_match"] += 1
            cur.last_seen_iso = ev.last_seen_iso
            if (cur.miles_from_cincy is None) and (ev.miles_from_cincy is not None):
                cur.miles_from_cincy = ev.miles_from_cincy
                cur.lat = ev.lat
                cur.lon = ev.lon
            if not cur.location and ev.location:
                cur.location = ev.location
            if not cur.city_state and ev.city_state:
                cur.city_state = ev.city_state
            if ev.source and ev.source not in cur.source:
                cur.source = clean_ws(f"{cur.source}; {ev.source}")
            merged[k] = cur
        else:
            merged[k] = ev

    if metrics is not None:
        metrics["dedupe_drop_reasons"] = dict(dedupe_reasons)
    log_counter_top("Dedupe merge reasons", dedupe_reasons)

    def sort_key(ev: EventItem):
        try:
            return datetime.fromisoformat(ev.start_iso)
        except Exception:
            return datetime.max.replace(tzinfo=tz.gettz("America/New_York"))

    return sorted(merged.values(), key=sort_key)


def write_csv(events: List[EventItem], path: str) -> None:
    fields = [
        "title",
        "start_iso",
        "end_iso",
        "category",
        "miles_from_cincy",
        "location",
        "city_state",
        "url",
        "source",
        "lat",
        "lon",
        "last_seen_iso",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for ev in events:
            w.writerow(asdict(ev))


# -------------------------
# Google Sheets export
# -------------------------
def get_google_credentials() -> Optional[service_account.Credentials]:
    service_account_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")
    service_account_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    scopes = [
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/spreadsheets",
    ]
    if service_account_path:
        return service_account.Credentials.from_service_account_file(service_account_path, scopes=scopes)
    if service_account_json:
        info = json.loads(service_account_json)
        return service_account.Credentials.from_service_account_info(info, scopes=scopes)
    return None


def find_or_create_subfolder(drive, parent_id: str, folder_name: str) -> str:
    query = (
        "mimeType='application/vnd.google-apps.folder' "
        f"and name='{folder_name}' "
        f"and '{parent_id}' in parents "
        "and trashed=false"
    )
    response = drive.files().list(q=query, fields="files(id, name)", **drive_list_kwargs()).execute()
    files = response.get("files", [])
    if files:
        return files[0]["id"]

    metadata = {"name": folder_name, "mimeType": "application/vnd.google-apps.folder", "parents": [parent_id]}
    created = drive.files().create(body=metadata, fields="id", supportsAllDrives=True).execute()
    return created["id"]


def find_or_create_spreadsheet(drive, parent_id: str, name: str) -> str:
    query = (
        "mimeType='application/vnd.google-apps.spreadsheet' "
        f"and name='{name}' "
        f"and '{parent_id}' in parents "
        "and trashed=false"
    )
    response = drive.files().list(q=query, fields="files(id, name)", **drive_list_kwargs()).execute()
    files = response.get("files", [])
    if files:
        return files[0]["id"]

    metadata = {"name": name, "mimeType": "application/vnd.google-apps.spreadsheet", "parents": [parent_id]}
    created = drive.files().create(body=metadata, fields="id", supportsAllDrives=True).execute()
    return created["id"]


def verify_parent_access(drive, parent_id: str) -> bool:
    try:
        meta = drive.files().get(
            fileId=parent_id,
            fields="id,name,driveId,owners(emailAddress),permissions(emailAddress,role)",
            supportsAllDrives=True,
        ).execute()
        name = meta.get("name", "unknown")
        drive_id = meta.get("driveId") or "My Drive"
        log(f"   Access verified for folder: {name} (drive: {drive_id})")
        return True
    except Exception as ex:
        log(f"❌ Unable to access parent folder {parent_id}: {ex}")
        log("   Ensure the service account has Editor access to the APEX folder.")
        return False


def ensure_sheet_tab(sheets, spreadsheet_id: str, title: str) -> None:
    sheet_info = sheets.spreadsheets().get(spreadsheetId=spreadsheet_id, includeGridData=False).execute()
    titles = {sheet["properties"]["title"] for sheet in sheet_info.get("sheets", [])}
    if title in titles:
        return
    requests_body = {"requests": [{"addSheet": {"properties": {"title": title}}}]}
    sheets.spreadsheets().batchUpdate(spreadsheetId=spreadsheet_id, body=requests_body).execute()

def update_apex_spreadsheet(events: List[EventItem]) -> None:
    dry_run = clean_ws(os.getenv("COLLECTOR_DRY_RUN", "")).lower() in ("1", "true", "yes", "y")
    if dry_run:
        log("ℹ️ COLLECTOR_DRY_RUN enabled: skipping Google Sheets write.")
        return

    creds = get_google_credentials()
    if not creds:
        log("⚠️ Skipping Google Sheets update: missing GOOGLE_SERVICE_ACCOUNT_FILE or GOOGLE_SERVICE_ACCOUNT_JSON.")
        return

    spreadsheet_id = os.getenv("APEX_SPREADSHEET_ID")
    if not spreadsheet_id:
        log("⚠️ Skipping Google Sheets update: missing APEX_SPREADSHEET_ID.")
        return

    sheets = build("sheets", "v4", credentials=creds)

    log(f"   Using spreadsheet ID: {spreadsheet_id}")
    log(f"   Sheet URL: https://docs.google.com/spreadsheets/d/{spreadsheet_id}")

    # Ensure tab exists
    ensure_sheet_tab(sheets, spreadsheet_id, "Events")

    headers = [
        "title",
        "start_iso",
        "end_iso",
        "category",
        "miles_from_cincy",
        "location",
        "city_state",
        "url",
        "source",
        "lat",
        "lon",
        "last_seen_iso",
    ]

    values = [headers]
    for ev in events:
        values.append(
            [
                ev.title,
                ev.start_iso,
                ev.end_iso,
                ev.category,
                ev.miles_from_cincy,
                ev.location,
                ev.city_state,
                ev.url,
                ev.source,
                ev.lat,
                ev.lon,
                ev.last_seen_iso,
            ]
        )

    # 1) Clear the sheet range first (prevents leftovers if list shrinks)
    sheets.spreadsheets().values().clear(
        spreadsheetId=spreadsheet_id,
        range="Events!A1:L",
        body={}
    ).execute()

    # 2) Write data
    sheets.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range="Events!A1",
        valueInputOption="RAW",
        body={"values": values},
    ).execute()

    # 3) Write a visible update stamp (column M is outside your table)
    stamp = f"Updated by bot: {now_et_iso()} | rows={len(values)-1}"
    sheets.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range="Events!M1",
        valueInputOption="RAW",
        body={"values": [[stamp]]},
    ).execute()

    # 4) Read back first few rows to prove it wrote
    preview = sheets.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id,
        range="Events!A1:C5",
    ).execute()
    got = preview.get("values", [])
    log(f"   Wrote {len(values)-1} events to Events!A1")
    log(f"   Preview A1:C5 = {got}")
    log(f"   Stamp written to Events!M1 = {stamp}")

# -------------------------
# Main
# -------------------------
def main():
    import time as _time

    t0 = _time.time()

    log("🚀 Collector starting…")
    log(f"   now_et_iso={now_et_iso()}")
    serpapi_enabled = bool(SERPAPI_API_KEY)
    log(f"   SERPAPI_API_KEY set? {'YES' if serpapi_enabled else 'NO'}")
    if not serpapi_enabled:
        log("ℹ️ SerpAPI disabled; continuing with configured collectors.")
    token_present = bool(get_facebook_access_token())
    log("   FACEBOOK_ACCESS_TOKEN env var read: FACEBOOK_ACCESS_TOKEN")
    log(f"   FACEBOOK_ACCESS_TOKEN non-empty? {'YES' if token_present else 'NO'}")
    log(f"   APEX_FACEBOOK_PAGES_SHEET_ID set? {'YES' if bool(os.getenv('APEX_FACEBOOK_PAGES_SHEET_ID')) else 'NO'}")
    log(f"   APEX_SPREADSHEET_ID set? {'YES' if bool(os.getenv('APEX_SPREADSHEET_ID')) else 'NO'}")

    cfg = load_yaml(CONFIG_PATH)
    sources = cfg.get("sources", [])
    log(f"   Config sources loaded: {len(sources)}")

    global RUN_ARTIFACT_DIR
    run_timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    RUN_ARTIFACT_DIR = os.path.join(RUNS_DIR, run_timestamp)
    os.makedirs(RUN_ARTIFACT_DIR, exist_ok=True)

    facebook_targets = load_facebook_targets(force_reload=True)

    geocache = load_json(GEOCODE_CACHE_PATH, {})
    url_cache = load_json(URL_CACHE_PATH, {})

    fb_token_status = get_token_manager().ensure_valid_user_token(refresh_days_threshold=7)
    FACEBOOK_COVERAGE["token"] = fb_token_status
    FACEBOOK_GRAPH_RUNTIME["checked"] = True
    FACEBOOK_GRAPH_RUNTIME["valid"] = fb_token_status.get("valid") == "yes"
    FACEBOOK_GRAPH_RUNTIME["reason"] = fb_token_status.get("reason", "unknown")
    if FACEBOOK_GRAPH_RUNTIME["valid"]:
        log(f"ℹ️ FACEBOOK_GRAPH token_valid: yes (expires_at_et={fb_token_status.get('expires_at_et', 'unknown')})")
    else:
        log(f"⚠️ FACEBOOK_GRAPH token_valid: no ({FACEBOOK_GRAPH_RUNTIME['reason']})")
        log(f"⚠️ Facebook Graph refresh/error detail: {fb_token_status.get('refresh_error') or fb_token_status.get('reason')}")
        log("ℹ️ Facebook Graph sources may skip; continuing with non-Graph collectors.")

    existing_raw = load_json(EVENTS_JSON_PATH, {"events": []})
    existing = [EventItem(**e) for e in existing_raw.get("events", [])]
    existing = filter_existing_automotive_events(existing, cfg)
    existing_before_focus_filter = len(existing)
    existing = [e for e in existing if is_automotive_event_safe(e.title, e.location, cfg)]
    dropped_existing_non_automotive = existing_before_focus_filter - len(existing)
    if dropped_existing_non_automotive:
        log(f"🧹 Removed non-automotive legacy events from existing set: {dropped_existing_non_automotive}")

    raw_events: List[dict] = []
    source_run_stats: List[dict] = []
    source_failures: List[dict] = []
    pipeline_metrics: Dict[str, dict] = {}
    source_filter_stats: Dict[str, Counter] = {}

    serpapi_source_types = {"web_search_serpapi", "web_search_facebook_events_serpapi", "facebook_group_events_serpapi", "serpapi_google_events"}

    for s in sources:
        stype = s.get("type")
        sname = s.get("name", "(unnamed source)")

        if stype in serpapi_source_types and not serpapi_enabled:
            log(f"⚠️ Skipping SerpAPI source (missing key): {sname} [{stype}]")
            source_run_stats.append({"name": sname, "type": stype, "status": "skipped", "collected": 0})
            continue

        source_with_context = dict(s)
        source_with_context["_facebook_targets"] = facebook_targets

        before_count = len(raw_events)
        try:
            diagnostics: dict = {}
            if stype == "html_carsandcoffeeevents_ohio":
                raw_events.extend(collect_carsandcoffeeevents_ohio(source_with_context))
            elif stype == "html_wordpress_events_list":
                raw_events.extend(collect_wordpress_events_series(source_with_context))
            elif stype == "ics":
                raw_events.extend(collect_ics(source_with_context))
            elif stype == "facebook_page_events":
                raw_events.extend(collect_facebook_page_events(source_with_context, diagnostics=diagnostics))
            elif stype == "web_search_serpapi":
                raw_events.extend(collect_web_search_serpapi(source_with_context, url_cache, diagnostics=diagnostics))
            elif stype == "web_search_facebook_events_serpapi":
                raw_events.extend(collect_web_search_facebook_events_serpapi(source_with_context, url_cache, diagnostics=diagnostics))
            elif stype == "facebook_group_events_serpapi":
                raw_events.extend(collect_facebook_group_events_serpapi(source_with_context, cfg, url_cache, diagnostics=diagnostics))
            elif stype == "serpapi_google_events":
                raw_events.extend(collect_serpapi_google_events(source_with_context, diagnostics=diagnostics))
            else:
                log(f"Skipping unknown source type: {stype} ({sname})")

            SOURCE_DIAGNOSTICS[sname] = diagnostics
            collected = len(raw_events) - before_count
            stat = {"name": sname, "type": stype, "status": "ok", "collected": collected}
            diag = SOURCE_DIAGNOSTICS.get(sname, {})
            if collected == 0:
                stat["reason"] = diag.get("reason", "no_results_from_search")
            source_run_stats.append(stat)
            log(f"🔎 Source complete: {sname} [{stype}] -> {collected} events")
        except SourceDisabledError as ex:
            source_run_stats.append({"name": sname, "type": stype, "status": "disabled", "collected": 0, "reason": ex.reason, "error": str(ex)})
            log(f"⚪ Source disabled: {sname} [{stype}] :: {ex}")
        except Exception as ex:
            source_run_stats.append({"name": sname, "type": stype, "status": "failed", "collected": 0, "error": str(ex)})
            source_failures.append({"name": sname, "type": stype, "error": str(ex), "traceback": log_exception_context(f"Source failed: {sname} [{stype}]", ex)})

    enable_fb_discovery = clean_ws(os.getenv("ENABLE_FACEBOOK_SERP_DISCOVERY", "1")).lower() not in ("0", "false", "no")
    if serpapi_enabled and enable_fb_discovery:
        try:
            fb_discovery_diag: dict = {}
            discovered = collect_facebook_events_serpapi_discovery(cfg, url_cache, diagnostics=fb_discovery_diag)
            SOURCE_DIAGNOSTICS["SerpAPI FB discovery (optional)"] = fb_discovery_diag
            raw_events.extend(discovered)
            log(f"🔎 Source complete: SerpAPI FB discovery (optional) -> {len(discovered)} events")
        except Exception as ex:
            source_failures.append({
                "name": "SerpAPI FB discovery (optional)",
                "type": "web_search_facebook_events_serpapi_discovery",
                "error": str(ex),
                "traceback": log_exception_context("Facebook SerpAPI discovery failed", ex),
            })
    elif serpapi_enabled:
        log("ℹ️ Facebook event URL discovery disabled via ENABLE_FACEBOOK_SERP_DISCOVERY.")
    else:
        log("⚠️ SERPAPI_API_KEY missing; skipping broad web discovery collectors.")

    log(f"📦 Raw events collected (pre-filter): {len(raw_events)}")
    ok_sources = [x for x in source_run_stats if x.get("status") == "ok"]
    failed_sources = [x for x in source_run_stats if x.get("status") == "failed"]
    log(f"🔎 Source summary: ok={len(ok_sources)} failed={len(failed_sources)}")

    incoming = to_event_items(raw_events, cfg, geocache, metrics=pipeline_metrics, source_filter_stats=source_filter_stats)
    log(f"✅ Incoming after filters: {len(incoming)}")

    for stat in source_run_stats:
        if stat.get("status") == "ok" and int(stat.get("collected", 0)) == 0:
            source_name = stat.get("name", "")
            diag = SOURCE_DIAGNOSTICS.get(source_name, {})
            stat["reason"] = derive_zero_yield_reason(source_name, diag, source_filter_stats.get(source_name, Counter()))

    merged = dedupe_merge(existing, incoming, metrics=pipeline_metrics)
    merged_before_focus_filter = len(merged)
    merged = [ev for ev in merged if is_automotive_event_safe(ev.title, ev.location, cfg)]
    dropped_merged_non_automotive = merged_before_focus_filter - len(merged)
    if dropped_merged_non_automotive:
        log(f"🧹 Removed non-automotive events after merge: {dropped_merged_non_automotive}")
    log(f"✅ Merged total events: {len(merged)}")

    save_json(GEOCODE_CACHE_PATH, geocache)
    save_json(URL_CACHE_PATH, url_cache)

    payload = {
        "generated_at_iso": now_et_iso(),
        "count": len(merged),
        "events": [asdict(e) for e in merged],
    }
    save_json(EVENTS_JSON_PATH, payload)
    write_csv(merged, EVENTS_CSV_PATH)

    update_apex_spreadsheet(merged)
    spreadsheet_id = os.getenv("APEX_SPREADSHEET_ID")
    if spreadsheet_id:
        log(f"📄 Sheets URL: https://docs.google.com/spreadsheets/d/{spreadsheet_id}")

    run_dir = RUN_ARTIFACT_DIR or os.path.join(RUNS_DIR, datetime.now().strftime('%Y-%m-%d_%H%M%S'))
    os.makedirs(run_dir, exist_ok=True)

    run_report = {
        "generated_at_iso": now_et_iso(),
        "raw_events": len(raw_events),
        "incoming_after_filters": len(incoming),
        "merged_total": len(merged),
        "source_stats": source_run_stats,
        "source_failures": source_failures,
        "serpapi_enabled": serpapi_enabled,
        "dry_run": clean_ws(os.getenv("COLLECTOR_DRY_RUN", "")).lower() in ("1", "true", "yes", "y"),
        "pipeline_metrics": pipeline_metrics,
        "facebook_coverage": {
            **{k: v for k, v in FACEBOOK_COVERAGE.items() if k != "failure_reasons"},
            "failure_reasons": dict(FACEBOOK_COVERAGE.get("failure_reasons", {})),
        },
    }
    run_path = os.path.join(run_dir, "run.json")
    save_json(run_path, run_report)
    failures_path = ""
    if source_failures:
        failures_path = os.path.join(run_dir, "failures.json")
        save_json(failures_path, {"generated_at_iso": now_et_iso(), "count": len(source_failures), "failures": source_failures})

    skipped_sources = [x for x in source_run_stats if x.get("status") == "skipped"]
    log(f"📄 Sheets rows written (excluding header): {len(merged)}")
    log(f"🔎 Source summary: ok={len(ok_sources)} skipped={len(skipped_sources)} failed={len(failed_sources)}")
    log_source_health_summary(source_run_stats)
    failure_top = []
    if isinstance(FACEBOOK_COVERAGE.get("failure_reasons"), Counter):
        failure_top = FACEBOOK_COVERAGE["failure_reasons"].most_common(10)
    elif isinstance(FACEBOOK_COVERAGE.get("failure_reasons"), dict):
        failure_top = Counter(FACEBOOK_COVERAGE["failure_reasons"]).most_common(10)

    log("📘 Facebook coverage report:")
    token_cov = FACEBOOK_COVERAGE.get("token", {}) or {}
    log(
        "   token_status="
        f"{token_cov.get('valid', 'unknown')} "
        f"expires_at_et={token_cov.get('expires_at_et', 'unknown')} "
        f"refresh_attempted={token_cov.get('refresh_attempted', False)} "
        f"refresh_succeeded={token_cov.get('refresh_succeeded', False)}"
    )
    log(f"   page_events_pulled={FACEBOOK_COVERAGE.get('page_events', 0)}")
    log(f"   serp_queries_executed={len(FACEBOOK_COVERAGE.get('serp_queries', []))}")
    for item in (FACEBOOK_COVERAGE.get("serp_queries", []) or [])[:30]:
        log(f"      - count={item.get('result_count', 0)} query={item.get('query', '')}")
    log(
        f"   event_urls discovered={FACEBOOK_COVERAGE.get('urls_discovered', 0)} "
        f"parsed={FACEBOOK_COVERAGE.get('urls_parsed', 0)} failed={FACEBOOK_COVERAGE.get('urls_failed', 0)}"
    )
    if failure_top:
        log("   top_failure_reasons:")
        for reason, cnt in failure_top:
            log(f"      - {reason}: {cnt}")
    log(f"✅ Done. Incoming: {len(incoming)} | Total: {len(merged)}")
    log(f"   Wrote: {EVENTS_JSON_PATH}")
    log(f"   Wrote: {EVENTS_CSV_PATH}")
    log(f"   Wrote: {GEOCODE_CACHE_PATH}")
    log(f"   Wrote: {URL_CACHE_PATH}")
    log(f"   Wrote: {run_path}")
    if failures_path:
        log(f"   Wrote: {failures_path}")
    log(
        f"📊 Final summary: sources_attempted={len(sources)} sources_failed={len(source_failures)} "
        f"events_collected={len(raw_events)}"
    )
    log(
        f"📁 Output summary: events_json={EVENTS_JSON_PATH} events_csv={EVENTS_CSV_PATH} "
        f"run_report={run_path}{' failures=' + failures_path if failures_path else ''}"
    )
    log(f"⏱️ Total runtime seconds: {round(_time.time() - t0, 1)}")



if __name__ == "__main__":
    try:
        main()
    except Exception as ex:
        log(f"❌ Fatal: {ex}")
        raise
