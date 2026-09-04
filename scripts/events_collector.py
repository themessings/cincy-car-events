#!/usr/bin/env python3
import csv
import html
import json
import os
import re
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, unquote, urlencode, urljoin, urlparse, urlunparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import dateparser
import requests
import yaml
from bs4 import BeautifulSoup
from dateutil import tz
from dateutil import parser as dateutil_parser
from dateutil.rrule import rrulestr
from icalendar import Calendar
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    from PIL import Image, ImageOps
except Exception:
    Image = ImageOps = None

from scripts.facebook_event_parser import parse_facebook_event_html, parse_facebook_event_page
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
URL_ENRICH_CACHE: Dict[str, Optional[dict]] = {}
# Real Eastern time (EST/EDT). The old fixed -5:00 offset stamped summer events
# an hour off internally, which broke duplicate detection across sources.
EST_TZ = tz.gettz("America/New_York")
SCREENSHOT_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif"}


def _normalize_title_for_export_dedupe(value: str) -> str:
    text = clean_ws(value).lower()
    if not text:
        return ""
    text = text.replace("&", " and ")
    text = re.sub(r"\band\b", " and ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"[\s\.,;:!\?-]+$", "", text)
    return text


def _normalize_location_for_export_dedupe(value: str) -> str:
    text = clean_ws(value).lower()
    if not text:
        return ""
    text = re.sub(r"\b(united\s+states|u\.?s\.?a\.?|u\.?s\.?)\b", "", text)
    text = re.sub(r"\s+", " ", text)
    parts = [clean_ws(p) for p in re.split(r"\s*,\s*", text) if clean_ws(p)]
    deduped_parts: List[str] = []
    seen = set()
    for part in parts:
        normalized_part = re.sub(r"[^a-z0-9 ]+", "", part).strip()
        if not normalized_part or normalized_part in seen:
            continue
        seen.add(normalized_part)
        deduped_parts.append(part)
    return ", ".join(deduped_parts)


_STREET_SUFFIX_CANON = {
    "street": "st", "avenue": "ave", "road": "rd", "drive": "dr",
    "boulevard": "blvd", "lane": "ln", "highway": "hwy", "parkway": "pkwy",
    "circle": "cir", "court": "ct", "place": "pl", "terrace": "ter",
    "trail": "trl", "square": "sq", "turnpike": "tpke",
}
_STREET_DIRECTION_CANON = {
    "north": "n", "south": "s", "east": "e", "west": "w",
    "northeast": "ne", "northwest": "nw", "southeast": "se", "southwest": "sw",
}
_STREET_WORD_CANON = {**_STREET_SUFFIX_CANON, **_STREET_DIRECTION_CANON}


def _canonicalize_street_address_for_dedupe(value: str) -> str:
    """Address match for dedupe, tolerant of formatting noise ('Main Street' vs
    'Main St', 'Suite 4' vs 'Ste 4') that would otherwise hide an obvious
    same-address duplicate from a plain string comparison."""
    text = _normalize_location_for_export_dedupe(value)
    if not text:
        return ""
    text = re.sub(r"[.,#]", " ", text)
    text = re.sub(r"\b(suite|ste|unit|apt|#)\s*\w*\b", " ", text)
    words = text.split()
    words = [_STREET_WORD_CANON.get(w, w) for w in words]
    return re.sub(r"\s+", " ", " ".join(words)).strip()


def _normalize_link_for_export_dedupe(value: str) -> str:
    raw = clean_ws(value)
    if not raw:
        return ""
    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    host = (parsed.netloc or "").lower()
    query_items = parse_qs(parsed.query, keep_blank_values=True)
    filtered_query = []
    for key in sorted(query_items.keys()):
        key_l = key.lower()
        if key_l.startswith("utm_") or key_l in {"fbclid", "gclid"}:
            continue
        for val in query_items[key]:
            filtered_query.append((key, val))
    query = urlencode(filtered_query, doseq=True)
    return urlunparse((parsed.scheme or "https", host, parsed.path, "", query, ""))


def _parse_export_row_start_dt(row: dict, start_iso_key: str, date_key: str, start_time_key: str) -> Optional[datetime]:
    raw_iso = clean_ws(str(row.get(start_iso_key, ""))) if start_iso_key else ""
    dt_value: Optional[datetime] = None
    if raw_iso:
        try:
            dt_value = dateutil_parser.isoparse(raw_iso)
        except Exception:
            try:
                dt_value = datetime.fromisoformat(raw_iso)
            except Exception:
                try:
                    dt_value = dateutil_parser.parse(raw_iso)
                except Exception:
                    dt_value = None

    if dt_value is None and date_key:
        date_raw = clean_ws(str(row.get(date_key, "")))
        time_raw = clean_ws(str(row.get(start_time_key, ""))) if start_time_key else ""
        if date_raw:
            try:
                dt_value = dateutil_parser.parse(f"{date_raw} {time_raw}".strip())
            except Exception:
                dt_value = None

    if dt_value and dt_value.tzinfo is None:
        dt_value = dt_value.replace(tzinfo=tz.gettz("America/New_York"))
    return dt_value


def dedupe_export_rows(rows: List[dict]) -> List[dict]:
    """Deduplicate export rows using canonical title/time/location/link rules.

    Inline test examples (expected kept row first in each cluster):
    1) Same title + same start_iso + same location variant => keep row with non-empty link.
    2) Same title + start times 7 minutes apart + same link (utm differs) => keep row with date/start/end populated.
    3) Same title + exact start_iso + duplicated address fragments => keep row with cleaner, longer location.
    4) Same title + same start_iso + matching normalized link + one row has end_iso => keep row with end_iso.
    5) Perfect score tie between duplicates => keep earliest occurrence for deterministic output.
    """
    if not rows:
        return []

    alias_map = {
        "title": ["title", "event_title", "name", "Event Name", "event name"],
        "start_iso": ["start_iso", "iso", "start", "startdatetime"],
        "end_iso": ["end_iso", "end", "enddatetime"],
        "date": ["date", "event_date", "Date"],
        "start_time": ["start_time", "time", "starttime", "Start Time", "start time"],
        "end_time": ["end_time", "endtime", "End Time", "end time"],
        "location": ["location", "venue", "address", "Location"],
        "link": ["link", "url", "event_url", "eventlink", "Event URL", "event url"],
        "categ": ["categ", "category", "type"],
    }
    row_keys = set().union(*(row.keys() for row in rows if isinstance(row, dict)))

    def pick_key(cands: List[str]) -> str:
        for c in cands:
            if c in row_keys:
                return c
        return ""

    k_title = pick_key(alias_map["title"])
    k_start_iso = pick_key(alias_map["start_iso"])
    k_end_iso = pick_key(alias_map["end_iso"])
    k_date = pick_key(alias_map["date"])
    k_start_time = pick_key(alias_map["start_time"])
    k_end_time = pick_key(alias_map["end_time"])
    k_location = pick_key(alias_map["location"])
    k_link = pick_key(alias_map["link"])

    enriched = []
    for index, row in enumerate(rows):
        title_norm = _normalize_title_for_export_dedupe(str(row.get(k_title, ""))) if k_title else ""
        location_norm = _normalize_location_for_export_dedupe(str(row.get(k_location, ""))) if k_location else ""
        link_norm = _normalize_link_for_export_dedupe(str(row.get(k_link, ""))) if k_link else ""
        start_dt = _parse_export_row_start_dt(row, k_start_iso, k_date, k_start_time)
        start_iso_norm = start_dt.isoformat() if start_dt else ""

        score = (
            1 if link_norm else 0,
            1 if all(clean_ws(str(row.get(k, ""))) for k in (k_date, k_start_time, k_end_time) if k) and all([k_date, k_start_time, k_end_time]) else 0,
            min(len(location_norm), 300),
            1 if (k_end_iso and clean_ws(str(row.get(k_end_iso, "")))) else 0,
            -index,
        )
        enriched.append(
            {
                "index": index,
                "row": row,
                "title_norm": title_norm,
                "location_norm": location_norm,
                "link_norm": link_norm,
                "start_dt": start_dt,
                "start_iso_norm": start_iso_norm,
                "score": score,
            }
        )

    _PLACEHOLDER_LOCATION_RE = re.compile(r"will be sent|ticket holders|\btba\b|\btbd\b|to be announced|to be determined")

    def _locations_compatible(a: str, b: str) -> bool:
        """Same event described two ways? Venue-only text from a calendar and a
        full street address from the web must still cluster; genuinely different
        venues (no shared text or street/zip numbers) must not."""
        if not a or not b:
            return True
        if _PLACEHOLDER_LOCATION_RE.search(a) or _PLACEHOLDER_LOCATION_RE.search(b):
            return True
        if a == b or a in b or b in a:
            return True
        nums_a = set(re.findall(r"\d{3,5}", a))
        nums_b = set(re.findall(r"\d{3,5}", b))
        return bool(nums_a & nums_b)

    def _title_is_specific(title_norm: str) -> bool:
        # Generic names ("car show", "cars and coffee") can belong to different
        # events on the same day; distinctive names cannot.
        return len(title_norm) >= 16 and len(title_norm.split()) >= 3

    def _items_match(item: dict, candidate: dict) -> bool:
        if not item["title_norm"] or item["title_norm"] != candidate["title_norm"]:
            return False
        starts_close = False
        if item["start_iso_norm"] and item["start_iso_norm"] == candidate["start_iso_norm"]:
            starts_close = True
        elif item["start_dt"] and candidate["start_dt"]:
            diff_minutes = abs((item["start_dt"] - candidate["start_dt"]).total_seconds()) / 60
            starts_close = diff_minutes <= 70
        if not starts_close:
            return False
        if _locations_compatible(item["location_norm"], candidate["location_norm"]):
            return True
        if item["link_norm"] and item["link_norm"] == candidate["link_norm"]:
            return True
        # Distinctive title at the same day+time: same event even when the two
        # sources describe the venue differently.
        return _title_is_specific(item["title_norm"])

    groups: List[List[dict]] = []
    for item in enriched:
        # Collect every group this item matches, then merge them — location
        # variants of one event can otherwise land in separate groups depending
        # on row order (A~C and B~C but not A~B).
        matching = [g for g in groups if any(_items_match(item, cand) for cand in g)]
        if not matching:
            groups.append([item])
        else:
            merged = matching[0]
            for g in matching[1:]:
                merged.extend(g)
                groups.remove(g)
            merged.append(item)

    deduped: List[dict] = []
    removed: List[dict] = []
    for group in groups:
        keep = max(group, key=lambda g: g["score"])
        deduped.append(keep["row"])
        for item in group:
            if item is not keep:
                removed.append(item["row"])

    log(
        "🧹 Export dedupe summary: "
        f"in={len(rows)} unique_out={len(deduped)} duplicates_removed={len(removed)}"
    )
    if removed:
        sample = []
        for row in removed[:5]:
            sample.append(
                f"{clean_ws(str(row.get(k_title, '')))} | {clean_ws(str(row.get(k_date, '')))} | {clean_ws(str(row.get(k_start_time, '')))}"
            )
        log("🧹 Removed duplicate sample: " + " || ".join(sample))

    return deduped


def get_token_manager() -> TokenManager:
    global TOKEN_MANAGER
    if TOKEN_MANAGER is None:
        TOKEN_MANAGER = TokenManager(log)
    return TOKEN_MANAGER


def get_facebook_access_token() -> str:
    return clean_ws(os.getenv("FACEBOOK_ACCESS_TOKEN") or FACEBOOK_ACCESS_TOKEN or "")


def extract_google_spreadsheet_id(value: str) -> str:
    raw = re.sub(r"\s+", " ", (value or "")).strip()
    if not raw:
        return ""
    if "docs.google.com/spreadsheets" in raw:
        m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", raw)
        if m:
            return m.group(1)
    return raw


APEX_IMPORT_SPREADSHEET_ID = extract_google_spreadsheet_id(
    os.getenv("APEX_IMPORT_SPREADSHEET_ID", "1xlIu0QIhyNSB1ptnLM6QvSKKYFwFcuVZBl26NMERczk")
)


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
        "fields": (
            "name,start_time,end_time,place,timezone,description,ticket_uri,"
            "attending_count,interested_count,maybe_count"
        ),
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
    full_address = ""

    if place.get("location"):
        loc = place["location"]
        parts = [loc.get("street"), loc.get("city"), loc.get("state"), loc.get("zip")]
        full_address = ", ".join(clean_ws(str(p)) for p in parts if clean_ws(str(p)))
        location = simplify_location(location, full_address) or location

    if not title or not start_dt:
        return None

    def _as_int(value) -> Optional[int]:
        try:
            return int(value)
        except Exception:
            return None

    return {
        "title": title,
        "start_dt": start_dt,
        "end_dt": end_dt or (start_dt + timedelta(hours=2)),
        "location": location,
        "address": full_address,
        "url": f"https://www.facebook.com/events/{event_id}",
        "description": clean_ws(str(item.get("description", "") or ""))[:2000],
        "ticket_uri": clean_ws(str(item.get("ticket_uri", "") or "")),
        "attending_count": _as_int(item.get("attending_count")),
        "interested_count": _as_int(item.get("interested_count")),
        "maybe_count": _as_int(item.get("maybe_count")),
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
                    "location": simplify_location(parsed_page.get('location', ''), parsed_page.get('address', '')),
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
    callout: str = ""  # e.g. "Registration required" / "Tickets required"
    closest_city: str = ""  # nearest major city, e.g. "Cincinnati, OH"
    address: str = ""  # full street address (looked up when the source lacks one)
    attending_count: Optional[int] = None  # Facebook "going" count when known
    interested_count: Optional[int] = None  # Facebook "interested" count when known
    popularity: Optional[int] = None  # 0-100 heuristic score (higher = bigger)
    size: str = ""  # tier label: Major / Regional / Local / Small
    description: str = ""  # short blurb captured during enrichment (not exported)
    date_verified: str = ""  # "" / "verified" / "corrected" / "unconfirmed"


def event_item_from_stored_row(row: dict) -> Optional[EventItem]:
    """Best-effort adapter for legacy events.json rows.

    Supports both the current EventItem JSON schema and older export-oriented
    rows that only contained date/start_time/end_time/link fields.
    """

    def pick(*keys: str) -> str:
        for key in keys:
            if key in row:
                return clean_ws(str(row.get(key, "")))
        return ""

    def parse_float(raw: str) -> Optional[float]:
        text = clean_ws(raw)
        if not text:
            return None
        try:
            return float(text)
        except Exception:
            return None

    start_iso = pick("start_iso", "iso", "start", "start_datetime", "startdatetime")
    end_iso = pick("end_iso", "end", "end_datetime", "enddatetime")

    start_dt = parse_iso_datetime_safe(start_iso)
    if start_dt is None:
        start_dt = parse_dt(f"{pick('date', 'event_date', 'Date')} {pick('start_time', 'time', 'starttime', 'Start Time')}")
    if start_dt is None:
        start_dt = parse_dt(pick("date", "event_date", "Date"))
    if start_dt is None:
        return None

    end_dt = parse_iso_datetime_safe(end_iso)
    if end_dt is None:
        end_dt = parse_dt(f"{pick('date', 'event_date', 'Date')} {pick('end_time', 'endtime', 'End Time')}")
    if end_dt is None:
        end_dt = start_dt

    location = pick("location", "venue", "address", "Location", "Address")
    city_state = pick("city_state") or guess_city_state(location)

    return EventItem(
        title=pick("title", "event_title", "name", "Event Name") or "(untitled event)",
        start_iso=start_dt.isoformat(),
        end_iso=end_dt.isoformat(),
        location=location,
        city_state=city_state,
        url=pick("url", "link", "event_url", "eventlink", "Event URL"),
        source=pick("source", "Source") or "legacy_import",
        category=pick("category", "categ", "type") or "local",
        miles_from_cincy=parse_float(pick("miles_from_cincy", "miles_from_c", "miles", "distance", "mileage")),
        lat=parse_float(pick("lat", "latitude")),
        lon=parse_float(pick("lon", "lng", "longitude")),
        last_seen_iso=pick("last_seen_iso", "last_seen") or now_et_iso(),
        callout=pick("callout", "Callout"),
        closest_city=pick("closest_city", "Closest City"),
    )


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

    raw = clean_ws(text)
    dt: Optional[datetime] = None
    try:
        dt = dateutil_parser.parse(raw)
    except Exception:
        dt = None

    if dt is None:
        dt = dateparser.parse(
            raw,
            settings={
                "RETURN_AS_TIMEZONE_AWARE": True,
                "TIMEZONE": "America/New_York",
                "TO_TIMEZONE": "America/New_York",
            },
        )

    if dt is None:
        return None

    et_tz = tz.gettz("America/New_York")
    if dt.tzinfo is None:
        return dt.replace(tzinfo=et_tz)
    return dt.astimezone(et_tz)


def _parse_time_token_for_screenshots(token: str) -> Optional[Tuple[int, int]]:
    raw = clean_ws(token).lower().replace(".", "").replace(" ", "")
    m = re.match(r"^(\d{1,2})(?::(\d{2}))?(am|pm)?$", raw)
    if not m:
        return None
    hour = int(m.group(1))
    minute = int(m.group(2) or 0)
    meridiem = m.group(3)
    if meridiem == "pm" and hour < 12:
        hour += 12
    if meridiem == "am" and hour == 12:
        hour = 0
    if hour > 23 or minute > 59:
        return None
    return hour, minute


def _extract_time_range_for_screenshots(text: str) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
    compact = text.replace("\u2013", "-").replace("\u2014", "-")
    range_match = re.search(
        r"(\d{1,2}(?::\d{2})?\s?(?:am|pm)?)\s*(?:-|to|until)\s*(\d{1,2}(?::\d{2})?\s?(?:am|pm)?)",
        compact,
        flags=re.IGNORECASE,
    )
    if range_match:
        return (
            _parse_time_token_for_screenshots(range_match.group(1)),
            _parse_time_token_for_screenshots(range_match.group(2)),
        )
    single = re.search(r"\b(\d{1,2}(?::\d{2})?\s?(?:am|pm))\b", compact, flags=re.IGNORECASE)
    if single:
        return _parse_time_token_for_screenshots(single.group(1)), None
    return None, None


def _extract_events_from_screenshot_text(text: str, filename: str, source_name: str) -> List[dict]:
    cleaned_text = text or ""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", cleaned_text) if clean_ws(b)]
    if not blocks and clean_ws(cleaned_text):
        blocks = [cleaned_text]

    out: List[dict] = []
    for block in blocks:
        lines = [clean_ws(line) for line in block.splitlines() if clean_ws(line)]
        if not lines:
            continue
        parsed_dt = parse_dt(block)
        if not parsed_dt:
            continue
        start_tuple, end_tuple = _extract_time_range_for_screenshots(block)
        if start_tuple:
            parsed_dt = parsed_dt.replace(hour=start_tuple[0], minute=start_tuple[1], second=0, microsecond=0)
        else:
            parsed_dt = parsed_dt.replace(hour=9, minute=0, second=0, microsecond=0)

        end_dt = parsed_dt + timedelta(hours=2)
        if end_tuple:
            end_dt = parsed_dt.replace(hour=end_tuple[0], minute=end_tuple[1], second=0, microsecond=0)
            if end_dt <= parsed_dt:
                end_dt += timedelta(days=1)

        title = lines[0]
        location_line = ""
        for line in lines[1:6]:
            if any(tok in line.lower() for tok in [" ave", " st", " rd", " blvd", " drive", " dr", " way", ", oh", " cincinnati"]):
                location_line = line
                break

        out.append(
            {
                "title": title,
                "start_dt": parsed_dt,
                "end_dt": end_dt,
                "location": location_line,
                "url": "",
                "source": f"screenshot:{filename}",
                "source_stream": source_name,
            }
        )
    return out


def collect_screenshot_folder_events(source: dict, diagnostics: Optional[dict] = None) -> List[dict]:
    diagnostics = diagnostics if diagnostics is not None else {}
    source_name = source.get("name", "Screenshot Folder OCR")
    folder_path = clean_ws(source.get("path") or os.getenv("SCREENSHOTS_DIR") or "/screenshots")
    if not os.path.isdir(folder_path):
        log(f"ℹ️ Screenshot folder not found: {folder_path}; skipping screenshot OCR source.")
        diagnostics["reason"] = "missing_folder"
        diagnostics["screenshots_detected"] = 0
        return []

    filenames = [
        name for name in sorted(os.listdir(folder_path))
        if os.path.isfile(os.path.join(folder_path, name)) and os.path.splitext(name)[1].lower() in SCREENSHOT_EXTS
    ]
    log(f"📸 Screenshot OCR: detected={len(filenames)} folder={folder_path}")
    diagnostics["screenshots_detected"] = len(filenames)

    if not filenames:
        diagnostics["reason"] = "no_results_from_search"
        return []

    if pytesseract is None or Image is None or ImageOps is None:
        log("⚠️ Screenshot OCR dependencies missing (pytesseract/Pillow); screenshots cannot be processed.")
        diagnostics["reason"] = "ocr_dependencies_missing"
        diagnostics["screenshots_processed"] = 0
        return []

    out: List[dict] = []
    processed = 0
    for filename in filenames:
        path = os.path.join(folder_path, filename)
        try:
            image = Image.open(path)
            image = ImageOps.exif_transpose(image).convert("RGB")
            ocr_text = pytesseract.image_to_string(image, config="--oem 3 --psm 6")
            processed += 1
            log(f"🧾 OCR raw output for {filename}:\n{ocr_text if ocr_text else '[empty]'}")

            parsed = _extract_events_from_screenshot_text(ocr_text, filename, source_name)
            log(f"📸 Screenshot parsed events: file={filename} count={len(parsed)}")
            if not parsed:
                log(f"WARNING: No events extracted from {filename}")
            out.extend(parsed)
        except Exception as ex:
            log(f"⚠️ Screenshot OCR failed for {filename}: {ex}")

    diagnostics["screenshots_processed"] = processed
    diagnostics["parsed_events"] = len(out)
    if not out:
        diagnostics["reason"] = "parse_failed"
    return out




def parse_iso_datetime_safe(raw_value: str, et_tz=None) -> Optional[datetime]:
    raw = clean_ws(raw_value or "")
    if not raw:
        return None

    et_tz = et_tz or EST_TZ
    dt: Optional[datetime] = None
    try:
        dt = datetime.fromisoformat(raw)
    except Exception:
        try:
            dt = dateutil_parser.parse(raw)
        except Exception:
            return None

    if dt.tzinfo is None:
        return dt.replace(tzinfo=et_tz)
    return dt.astimezone(et_tz)


def prune_past_events(events: List[dict], now: datetime) -> List[dict]:
    mode = clean_ws(os.getenv("EVENT_PRUNE_MODE", "end_before_now")).lower() or "end_before_now"
    if mode not in {"end_before_now", "start_before_now"}:
        log(f"⚠️ Unknown EVENT_PRUNE_MODE='{mode}', defaulting to end_before_now")
        mode = "end_before_now"

    et_tz = EST_TZ
    if now.tzinfo is None:
        now_et = now.replace(tzinfo=et_tz)
    else:
        now_et = now.astimezone(et_tz)

    kept: List[dict] = []
    removed: List[dict] = []

    for ev in events:
        start_dt = parse_iso_datetime_safe(ev.get("start_iso", ""), et_tz=et_tz)
        end_dt = parse_iso_datetime_safe(ev.get("end_iso", ""), et_tz=et_tz) or start_dt

        cmp_dt = start_dt if mode == "start_before_now" else end_dt
        if cmp_dt is None:
            kept.append(ev)
            continue

        if cmp_dt < now_et:
            removed.append(ev)
            continue

        kept.append(ev)

    log(
        f"🧹 Pruned past events: removed={len(removed)} kept={len(kept)} now_et_iso={now_et.isoformat()}"
    )
    if removed:
        sample_titles = [clean_ws(ev.get("title", "")) or "(untitled)" for ev in removed[:3]]
        log(f"   Removed examples: {sample_titles}")

    return kept


def prune_cache_by_age(cache_obj, now: datetime, days: int = 180, label: str = "cache"):
    if not isinstance(cache_obj, dict):
        return cache_obj

    et_tz = EST_TZ
    now_et = now if now.tzinfo else now.replace(tzinfo=et_tz)
    now_et = now_et.astimezone(et_tz)
    cutoff = now_et - timedelta(days=days)

    kept = {}
    removed = 0
    missing_ts = 0
    for key, value in cache_obj.items():
        if not isinstance(value, dict):
            kept[key] = value
            continue
        fetched = parse_iso_datetime_safe(value.get("fetched_at_iso", ""), et_tz=et_tz)
        if fetched is None:
            missing_ts += 1
            kept[key] = value
            continue
        if fetched < cutoff:
            removed += 1
            continue
        kept[key] = value

    if removed:
        log(f"🧹 Pruned {label}: removed={removed} kept={len(kept)} cutoff_et_iso={cutoff.isoformat()}")
    elif missing_ts:
        log(f"ℹ️ Skipped age prune for {label} entries without fetched_at_iso: {missing_ts}")

    return kept


def haversine_miles(lat1, lon1, lat2, lon2) -> float:
    R = 3958.7613
    import math

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# Major cities used for the "Closest City" export column (name, lat, lon).
MAJOR_CITIES: List[Tuple[str, float, float]] = [
    ("Cincinnati, OH", 39.1031, -84.5120),
    ("Dayton, OH", 39.7589, -84.1916),
    ("Columbus, OH", 39.9612, -82.9988),
    ("Cleveland, OH", 41.4993, -81.6944),
    ("Toledo, OH", 41.6528, -83.5379),
    ("Louisville, KY", 38.2527, -85.7585),
    ("Lexington, KY", 38.0406, -84.5037),
    ("Indianapolis, IN", 39.7684, -86.1581),
    ("Fort Wayne, IN", 41.0793, -85.1394),
    ("Nashville, TN", 36.1627, -86.7816),
    ("Knoxville, TN", 35.9606, -83.9207),
    ("Pittsburgh, PA", 40.4406, -79.9959),
    ("Charleston, WV", 38.3498, -81.6326),
    ("Detroit, MI", 42.3314, -83.0458),
    ("Chicago, IL", 41.8781, -87.6298),
    ("St. Louis, MO", 38.6270, -90.1994),
]


def closest_major_city(lat: Optional[float], lon: Optional[float]) -> str:
    if lat is None or lon is None:
        return ""
    try:
        name, _, _ = min(MAJOR_CITIES, key=lambda c: haversine_miles(lat, lon, c[1], c[2]))
    except Exception:
        return ""
    return name


REGISTRATION_KEYWORDS = (
    "registration", "register", "rsvp", "sign up", "sign-up", "signup",
    "pre-register", "preregister", "entry fee", "entry form",
)
TICKET_KEYWORDS = (
    "ticket", "admission", "box office", "gate fee", "admittance",
)
REGISTRATION_NEGATIONS = (
    "no registration", "registration not required", "registration is not required",
    "without registration", "no rsvp",
)
TICKET_NEGATIONS = (
    "no ticket", "free admission", "free entry", "free to attend",
    "free event", "no admission",
)


def detect_registration_callout(*texts: str, ticket_url: str = "") -> str:
    """Best-effort detection for the export "Callout" column: does the event
    appear to require registration and/or tickets? Returns "" when no signal."""
    blob = " ".join(clean_ws(str(t)).lower() for t in texts if t)

    reg_negated = any(p in blob for p in REGISTRATION_NEGATIONS)
    tix_negated = any(p in blob for p in TICKET_NEGATIONS)

    has_reg = (not reg_negated) and any(k in blob for k in REGISTRATION_KEYWORDS)
    has_tix = bool(clean_ws(ticket_url)) or ((not tix_negated) and any(k in blob for k in TICKET_KEYWORDS))

    if has_reg and has_tix:
        return "Registration & tickets required"
    if has_reg:
        return "Registration required"
    if has_tix:
        return "Tickets required"
    return ""


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
        normalized_address = ", ".join(clean_ws(str(p)) for p in parts if clean_ws(str(p)))
        location = simplify_location(location, normalized_address) or location

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
        "description": clean_ws(str(item.get("description", "") or ""))[:2000],
        "ticket_uri": clean_ws(str(item.get("ticket_uri", "") or "")),
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


def _stringify_location_like(value) -> str:
    """Convert heterogeneous location payloads into a readable one-line string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return clean_ws(value)
    if isinstance(value, (int, float, bool)):
        return clean_ws(str(value))
    if isinstance(value, list):
        parts = [_stringify_location_like(v) for v in value]
        return clean_ws(" | ".join(p for p in parts if p))
    if isinstance(value, dict):
        direct_keys = ["name", "title", "venue", "event_location", "location", "address"]
        bits = []
        for k in direct_keys:
            v = value.get(k)
            txt = _stringify_location_like(v)
            if txt:
                bits.append(txt)

        if not bits:
            addr = value.get("address")
            if isinstance(addr, dict):
                addr_bits = [
                    addr.get("street"),
                    addr.get("streetAddress"),
                    addr.get("addressLocality"),
                    addr.get("city"),
                    addr.get("addressRegion"),
                    addr.get("state"),
                    addr.get("postalCode"),
                ]
                bits.extend(clean_ws(str(b)) for b in addr_bits if clean_ws(str(b)))

        if bits:
            deduped = []
            seen = set()
            for b in bits:
                key = b.lower()
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(b)
            return clean_ws(", ".join(deduped))

        try:
            return clean_ws(json.dumps(value, ensure_ascii=False, sort_keys=True))
        except Exception:
            return clean_ws(str(value))

    return clean_ws(str(value))


def simplify_location(location: str, address: str = "") -> str:
    """Return a simple, de-duplicated address string.

    - keeps venue name when present
    - normalizes address to comma-separated segments
    - removes duplicated trailing address text when location already contains it
    """
    raw_location = clean_ws(location)
    raw_address = clean_ws(address)

    if not raw_address:
        return raw_location

    address_csv = re.sub(r"\s+", " ", raw_address)
    address_csv = re.sub(r"\s*,\s*", ", ", address_csv)
    address_csv = address_csv.strip(" ,")

    address_plain = re.sub(r",", "", address_csv)
    address_plain = clean_ws(address_plain)

    if not raw_location:
        return address_csv

    lower_location = raw_location.lower()
    if address_csv.lower() in lower_location or address_plain.lower() in lower_location:
        return raw_location

    return clean_ws(f"{raw_location}, {address_csv}")


US_STATE_ABBR_RE = (
    r"(?:AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|"
    r"NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY|DC)"
)
US_ADDR_FULL_RE = re.compile(
    rf"(\d{{1,6}}\s+[^,]+,\s*[^,]+,\s*{US_STATE_ABBR_RE}\s*\d{{5}}(?:-\d{{4}})?)",
    re.IGNORECASE,
)
US_ADDR_WEAK_RE = re.compile(
    rf"(\d{{1,6}}\s+[^,]+,\s*[^,]+,\s*{US_STATE_ABBR_RE})(?!\s*\d)",
    re.IGNORECASE,
)


def _format_address_candidate(candidate: str, require_zip: bool = True) -> str:
    """Normalize an extracted candidate to: street, city, ST [ZIP]."""
    candidate_clean = re.sub(r"\s*,\s*", ", ", clean_ws(candidate)).strip(" ,")
    if require_zip:
        m = re.match(
            rf"^(\d{{1,6}}\s+[^,]+),\s*([^,]+),\s*({US_STATE_ABBR_RE})\s*(\d{{5}}(?:-\d{{4}})?)$",
            candidate_clean,
            flags=re.IGNORECASE,
        )
        if not m:
            return ""
        street, city, state, zip_code = m.groups()
        return f"{clean_ws(street)}, {clean_ws(city)}, {state.upper()} {zip_code}"

    m = re.match(
        rf"^(\d{{1,6}}\s+[^,]+),\s*([^,]+),\s*({US_STATE_ABBR_RE})$",
        candidate_clean,
        flags=re.IGNORECASE,
    )
    if not m:
        return ""
    street, city, state = m.groups()
    return f"{clean_ws(street)}, {clean_ws(city)}, {state.upper()}"


def _extract_venue_prefix(text: str, formatted_address: str) -> str:
    """Return the venue name preceding the extracted address, if any.

    Looks for the first occurrence of the address's street portion in the raw
    text; anything before it is treated as the venue name candidate.
    """
    street = clean_ws(formatted_address.split(",", 1)[0])
    if not street:
        return ""
    idx = text.lower().find(street.lower())
    if idx <= 0:
        return ""
    prefix = clean_ws(text[:idx]).strip(" ,-–—")
    if not prefix or len(prefix) > 80:
        return ""
    # Reject street-address fragments ("2812 Fishinger Rd") but keep venue names
    # that merely start with a number ("41 North Tavern").
    street_word = r"(?:rd|road|st|street|ave|avenue|dr|drive|blvd|boulevard|hwy|highway|pike|ln|lane|way|pkwy|parkway|ct|court|cir|circle|pl|place)"
    if re.search(rf"^\d{{1,6}}\s+.*\b{street_word}\.?$", prefix, flags=re.IGNORECASE):
        return ""
    return prefix


_STREET_ADDRESS_START_RE = re.compile(
    r"\d{1,6}\s+[A-Za-z0-9.\-' ]+?\b(?:street|st|avenue|ave|road|rd|drive|dr|boulevard|blvd|"
    r"lane|ln|way|pike|highway|hwy|court|ct|parkway|pkwy|circle|cir|trail|terrace|place|pl)\b",
    re.IGNORECASE,
)


def _reconcile_location_with_address(location: str, address: str) -> str:
    """Drop a street-address fragment embedded in Location once we have a
    confirmed Address, so the two columns can't state conflicting street-level
    facts (e.g. Location carrying a source page's stale ZIP that differs from
    the geocoded one). Keeps the venue-name prefix; leaves Location alone when
    it has no venue name of its own (it just *is* the address already)."""
    text = clean_ws(location)
    if not text or not clean_ws(address):
        return text
    m = _STREET_ADDRESS_START_RE.search(text)
    if not m:
        return text
    venue = clean_ws(text[: m.start()]).strip(" ,-–—")
    return venue or text


def clean_location(raw: str) -> str:
    """Extract one clean US postal address, keeping the venue name when present.

    Output format: "Venue Name - street, city, ST ZIP" (no country), or just the
    address when no venue name precedes it.

    Example mappings (unit-test style):
      - "Colin's Coffee 2812 Fishinger Rd, Upper Arlington, United States 2812 Fishinger Rd, Upper Arlington, OH 43221"
        -> "Colin's Coffee - 2812 Fishinger Rd, Upper Arlington, OH 43221"
      - "Starbucks (Westerville) 745 Springfield St, Cincinnati, OH 45215 USA"
        -> "Starbucks (Westerville) - 745 Springfield St, Cincinnati, OH 45215"
      - "745 Springfield St, Cincinnati, OH 45215 745 Springfield St, Cincinnati, OH 45215"
        -> "745 Springfield St, Cincinnati, OH 45215"
      - "Venue Name 123 Main St, Columbus, OH"
        -> "Venue Name - 123 Main St, Columbus, OH"
      - "Meet at the lot behind the mall"
        -> "Meet at the lot behind the mall"
    """
    text = clean_ws(raw)
    if not text:
        return ""

    text = re.sub(r"\b(?:United\s+States(?:\s+of\s+America)?|USA|US)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = clean_ws(text).strip(" ,")

    def with_venue(formatted: str) -> str:
        venue = _extract_venue_prefix(text, formatted)
        return f"{venue} - {formatted}" if venue else formatted

    seen = set()
    for match in US_ADDR_FULL_RE.finditer(text):
        formatted = _format_address_candidate(match.group(1), require_zip=True)
        if not formatted:
            continue
        key = formatted.lower()
        if key in seen:
            continue
        seen.add(key)
        return with_venue(formatted)

    seen.clear()
    for match in US_ADDR_WEAK_RE.finditer(text):
        formatted = _format_address_candidate(match.group(1), require_zip=False)
        if not formatted:
            continue
        key = formatted.lower()
        if key in seen:
            continue
        seen.add(key)
        return with_venue(formatted)

    return text


def normalize_location_for_output(location: str, city_state: str = "") -> str:
    """Normalize final location field for event exports.

    - collapses whitespace and duplicate commas
    - appends city/state when location is blank
    - appends city/state to venue-only location when it's missing
    - avoids repeating city/state when already included in location text
    """
    normalized_location = clean_location(location)
    normalized_location = re.sub(r"\s*,\s*", ", ", normalized_location).strip(" ,")

    normalized_city_state = clean_ws(city_state)
    normalized_city_state = re.sub(r"\s*,\s*", ", ", normalized_city_state).strip(" ,")

    if not normalized_location:
        return normalized_city_state
    if not normalized_city_state:
        return normalized_location

    location_lower = normalized_location.lower()
    city_state_lower = normalized_city_state.lower()
    city_only = clean_ws(normalized_city_state.split(",", 1)[0]).lower()

    if city_state_lower in location_lower:
        return normalized_location
    if city_only and re.search(rf"\b{re.escape(city_only)}\b", location_lower):
        return normalized_location

    return f"{normalized_location}, {normalized_city_state}"


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


# -------------------------
# Automotive classification (content-based, word-boundary matched)
# -------------------------
# The automotive signal is judged from the event's CONTENT (title / description /
# location) — never the source name. Historically the source name "Meetup Cincy
# Cars" (contains "cars") auto-passed every unrelated meetup, and naive substring
# matching let "car" match "health<car>e" / "s<car>ecrow" / "<car>eer". Both are
# fixed here with word-boundary regexes and a tiered keyword model.

# STRONG terms qualify an event on their own.
AUTO_STRONG_DEFAULTS = [
    "car show", "cars and coffee", "cars & coffee", "cars n coffee", "car meet",
    "car meetup", "auto show", "autocross", "auto cross", "motorsport", "motorsports",
    "concours", "cruise-in", "cruise in", "cruise night", "car cruise", "car club",
    "car rally", "road rally", "roadrally", "rally", "rallycross", "rally cross",
    "stage rally", "driving tour", "track day",
    "hpde", "time trial", "time attack", "lapping day", "drag racing", "drag strip",
    "dyno day", "test and tune", "test & tune", "poker run", "hot rod", "street rod",
    "muscle car", "classic car", "exotic car", "super car", "supercar", "hypercar",
    "sports car", "vintage car", "antique car", "import car", "tuner car",
    "car festival", "swap meet", "porsche club", "bmw cca", "corvette club",
    "mustang club", "vehicle show", "truck show", "gymkhana",
]
# CONTEXT tokens make an ambiguous WEAK keyword qualify.
AUTO_CONTEXT_DEFAULTS = [
    "car", "cars", "auto", "autos", "automobile", "automotive", "vehicle", "vehicles",
    "truck", "trucks", "motor", "engine", "horsepower", "coupe", "roadster", "jdm",
    "porsche", "corvette", "mustang", "camaro", "chevrolet", "chevy", "ford", "dodge",
    "mopar", "bmw", "audi", "mercedes", "ferrari", "lamborghini", "nissan", "toyota",
    "honda", "subaru", "mazda", "cadillac", "pontiac", "volkswagen", "jeep", "lexus",
    "acura",
]
# WEAK keywords qualify ONLY when a CONTEXT token co-occurs.
AUTO_WEAK_DEFAULTS = [
    "meet", "meetup", "meet-up", "meet up", "cruise", "cruisin", "show", "rally",
    "solo", "euro", "muscle", "tuner", "classic", "import", "exotic", "rod", "tour",
]
# HARD exclusions always drop the event and cannot be overridden by a car keyword.
AUTO_HARD_EXCLUSION_DEFAULTS = [
    "5k", "10k", "half marathon", "marathon", "fun run", "color run", "job fair",
    "hiring event", "career fair", "church service", "yoga", "kids camp",
    "food pantry", "music track", "spotify track", "medical marijuana",
    "speed dating", "book club", "blood drive", "dance class",
    # non-car "rally" senses (bare "rally" is otherwise a strong car keyword)
    "pep rally", "political rally", "prayer rally", "protest rally", "campaign rally",
]

_AUTO_PATTERN_CACHE: Dict[str, "re.Pattern"] = {}


def _auto_normalize_text(value: str) -> str:
    text = clean_ws(str(value or "")).lower()
    text = text.replace("&", " and ")
    return re.sub(r"\s+", " ", text).strip()


def _auto_keyword_pattern(keyword: str) -> "re.Pattern":
    key = clean_ws(str(keyword)).lower()
    cached = _AUTO_PATTERN_CACHE.get(key)
    if cached is not None:
        return cached
    norm = key.replace("&", " and ")
    norm = re.sub(r"[\s\-]+", " ", norm).strip()
    tokens = [re.escape(tok) for tok in norm.split(" ") if tok]
    if not tokens:
        pattern = re.compile(r"(?!x)x")  # never matches
    else:
        body = r"[\s\-]+".join(tokens)
        # Boundaries are alnum-aware so "car" does NOT match "scarecrow"/"career".
        pattern = re.compile(rf"(?<![a-z0-9]){body}(?![a-z0-9])")
    _AUTO_PATTERN_CACHE[key] = pattern
    return pattern


def _auto_keywords(cfg: Optional[dict], key: str, defaults: List[str]) -> List[str]:
    filters = (cfg or {}).get("filters", {}) if isinstance(cfg, dict) else {}
    values = filters.get(key)
    if not values:
        values = defaults
    return [clean_ws(str(k)).lower() for k in values if clean_ws(str(k))]


def _auto_first_match(text: str, keywords: List[str]) -> str:
    for kw in keywords:
        if _auto_keyword_pattern(kw).search(text):
            return kw
    return ""


def automotive_content_verdict(
    title: str, location: str = "", description: str = "", cfg: Optional[dict] = None
) -> Tuple[str, str]:
    """Classify an event from its content. Returns (verdict, reason) where
    verdict is 'keep' (clear automotive signal), 'drop' (excluded / clearly not),
    or 'unknown' (no signal either way — caller decides using source trust)."""
    text = _auto_normalize_text(f"{title} {location} {description}")
    if not text:
        return "drop", "empty_text"

    hard_hit = _auto_first_match(text, AUTO_HARD_EXCLUSION_DEFAULTS)
    if hard_hit:
        return "drop", f"hard_exclusion:{hard_hit}"

    strong_hit = _auto_first_match(text, _auto_keywords(cfg, "automotive_strong_keywords", AUTO_STRONG_DEFAULTS))
    if strong_hit:
        # A clear car term overrides soft (config) exclusions like "farmers market".
        return "keep", f"strong:{strong_hit}"

    excl_hit = _auto_first_match(text, _auto_keywords(cfg, "non_automotive_exclude_keywords", []))
    if excl_hit:
        return "drop", f"exclude_keyword:{excl_hit}"

    weak_hit = _auto_first_match(text, _auto_keywords(cfg, "automotive_weak_keywords", AUTO_WEAK_DEFAULTS))
    if weak_hit:
        context_hit = _auto_first_match(text, _auto_keywords(cfg, "automotive_context_keywords", AUTO_CONTEXT_DEFAULTS))
        if context_hit:
            return "keep", f"weak+context:{weak_hit}+{context_hit}"

    return "unknown", "no_automotive_signal"


def is_automotive_event(
    title: str,
    location: str = "",
    cfg: Optional[dict] = None,
    *_unused,
    **_unused_kw,
) -> bool:
    """Best-effort keyword gate with backwards-compatible signature.

    Lenient: only rejects events that are excluded or clearly non-automotive;
    keeps 'unknown' events (no signal either way)."""
    verdict, _ = automotive_content_verdict(title, location, cfg=cfg)
    return verdict != "drop"


def is_automotive_event_safe(title: str, location: str, cfg: dict) -> bool:
    """Strict automotive check used by main() filtering of existing/merged events.

    Requires a positive automotive signal from the event content; 'unknown'
    events are rejected here so previously-leaked non-car rows get cleaned out.
    Car-dedicated/bypass sources are exempted by the caller before this runs."""
    verdict, _ = automotive_content_verdict(title, location, cfg=cfg)
    return verdict == "keep"


def filter_existing_automotive_events(existing: List[EventItem], cfg: dict) -> List[EventItem]:
    """Filter persisted events to automotive-only without relying on call-signature-sensitive paths."""
    filtered = [e for e in existing if is_automotive_event_safe(e.title, e.location, cfg)]
    dropped = len(existing) - len(filtered)
    if dropped:
        log(f"🧹 Filtered out {dropped} non-automotive persisted events before merge.")
    return filtered
def source_is_car_dedicated(source: str, cfg: Optional[dict]) -> str:
    """Return the matched car-dedicated source substring, or "" if none.

    Car-dedicated sources (club calendars, sanctioning bodies, registration
    platforms) are trusted as automotive even when a title lacks car keywords."""
    source_l = clean_ws(str(source)).lower()
    if not source_l:
        return ""
    filters = (cfg or {}).get("filters", {}) if isinstance(cfg, dict) else {}
    for sub in (filters.get("car_dedicated_sources", []) or []):
        sub_l = clean_ws(str(sub)).lower()
        if sub_l and sub_l in source_l:
            return sub_l
    return ""


def evaluate_automotive_focus_event(
    title: str, location: str, source: str, url: str, cfg: dict, description: str = ""
) -> Tuple[bool, str]:
    """Returns (is_allowed, reason) for automotive filtering transparency.

    The automotive signal comes from event CONTENT (title/description/location),
    not the source name. 'unknown' events are admitted only when they come from a
    car-dedicated source or a trusted event platform URL."""
    verdict, reason = automotive_content_verdict(title, location, description, cfg=cfg)
    if verdict == "keep":
        return True, reason
    if verdict == "drop":
        return False, reason

    # verdict == "unknown": lean on source/platform trust, else drop.
    dedicated = source_is_car_dedicated(source, cfg)
    if dedicated:
        return True, f"car_dedicated_source:{dedicated}"

    filters = (cfg or {}).get("filters", {})
    url_l = clean_ws(str(url)).lower()
    trusted_platforms = [clean_ws(k).lower() for k in (filters.get("trusted_event_platforms", []) or []) if clean_ws(k)]
    for k in trusted_platforms:
        # Require the platform in the URL (not the source name) so a car-scoped
        # discovery query on eventbrite/facebook keeps its structured event pages
        # without blanket-passing every unrelated event from that domain.
        if k and k in url_l:
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
def reliable_geocode_query(location: str, city_state: str = "") -> Optional[str]:
    """Return a geocode query only when it can be trusted.

    Bare venue names ("Cox Park", "Austin Landing") geocode to random places
    worldwide — those produced bogus 770–9,375 mile distances that wrongly
    deleted nearby events. Only ZIPs, city+state, or state-bearing addresses
    are reliable enough to act on.
    """
    location = clean_ws(location)
    city_state = clean_ws(city_state)

    zip_match = re.search(r"\b(\d{5})(?:-\d{4})?\b", location)
    if zip_match:
        return f"{zip_match.group(1)}, USA"
    if re.search(r",\s*[A-Z]{2}\b", city_state):
        return city_state
    if re.search(r",\s*[A-Z]{2}\b", location):
        return location
    return None


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
    params = {"q": place, "format": "json", "limit": 1, "countrycodes": "us"}
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
        start_dt, end_dt = _extract_wordpress_event_datetimes(b, start_dt)

        loc_el = b.select_one(".tribe-events-calendar-list__event-venue-title, .tribe-events-calendar-list__event-venue")
        addr_el = b.select_one(".tribe-events-calendar-list__event-venue-address, .tribe-events-venue-details")
        location = clean_ws((loc_el.get_text() if loc_el else "") + " " + (addr_el.get_text() if addr_el else ""))

        if not title or not start_dt:
            continue

        events.append(
            {
                "title": title,
                "start_dt": start_dt,
                "end_dt": end_dt or (start_dt + timedelta(hours=2)),
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
        start_dt, end_dt = _extract_wordpress_event_datetimes(r, start_dt)

        loc_el = r.select_one(".tribe-events-calendar-list__event-venue-title, .tribe-events-calendar-list__event-venue")
        addr_el = r.select_one(".tribe-events-calendar-list__event-venue-address, .tribe-events-venue-details")
        location = clean_ws((loc_el.get_text() if loc_el else "") + " " + (addr_el.get_text() if addr_el else ""))

        if not title or not start_dt:
            continue

        events.append(
            {
                "title": title,
                "start_dt": start_dt,
                "end_dt": end_dt or (start_dt + timedelta(hours=2)),
                "location": location,
                "url": url,
                "source": source["name"],
            }
        )
    return events


def collect_tribe_events_api(source: dict, diagnostics: Optional[dict] = None) -> List[dict]:
    """Collect events from a The Events Calendar (Tribe) WordPress REST API.

    Replaces fragile HTML scraping of carsandcoffeeevents.com: the open REST
    endpoint returns structured venue data (address/city/state/zip), cost, and
    description, filtered server-side by state category.
    """
    diagnostics = diagnostics if diagnostics is not None else {}
    diagnostics.setdefault("raw_candidates", 0)
    diagnostics.setdefault("parse_failures", 0)

    base_url = clean_ws(source.get("url", "")).rstrip("/")
    if not base_url:
        diagnostics["reason"] = "missing_env"
        return []
    endpoint = f"{base_url}/wp-json/tribe/events/v1/events"

    categories = source.get("categories") or [None]
    lookahead_days = int(source.get("lookahead_days", 120))
    per_page = int(source.get("per_page", 50))
    max_pages = int(source.get("max_pages", 30))
    source_name = source.get("name", "Tribe Events API")
    bypass = bool(source.get("bypass_automotive_filter", False))

    et_tz = tz.gettz("America/New_York")  # real ET (handles EST/EDT), not the fixed EST_TZ offset
    now_et = datetime.now(tz=et_tz)
    start_date = now_et.strftime("%Y-%m-%d")
    end_date = (now_et + timedelta(days=lookahead_days)).strftime("%Y-%m-%d")

    out: List[dict] = []
    seen = set()

    for category in categories:
        page = 1
        while page <= max_pages:
            params = {
                "per_page": per_page,
                "page": page,
                "start_date": start_date,
                "end_date": end_date,
            }
            if category:
                params["categories"] = category
            try:
                r = requests.get(endpoint, params=params, headers=DEFAULT_HTTP_HEADERS, timeout=45)
                if r.status_code == 400 and page > 1:
                    break  # past the last page
                r.raise_for_status()
                payload = r.json() or {}
            except Exception as ex:
                log(f"⚠️ Tribe API request failed ({source_name} cat={category} page={page}): {ex}")
                diagnostics["parse_failures"] += 1
                break

            rows = payload.get("events", []) or []
            diagnostics["raw_candidates"] += len(rows)

            for item in rows:
                title = clean_ws(html.unescape(str(item.get("title", "") or "")))
                start_raw = clean_ws(str(item.get("start_date", "") or ""))
                end_raw = clean_ws(str(item.get("end_date", "") or ""))
                if not title or not start_raw:
                    diagnostics["parse_failures"] += 1
                    continue
                try:
                    start_dt = datetime.fromisoformat(start_raw).replace(tzinfo=et_tz)
                except Exception:
                    diagnostics["parse_failures"] += 1
                    continue
                try:
                    end_dt = datetime.fromisoformat(end_raw).replace(tzinfo=et_tz) if end_raw else None
                except Exception:
                    end_dt = None

                venue = item.get("venue") or {}
                if isinstance(venue, list):
                    venue = venue[0] if venue else {}
                venue_name = clean_ws(html.unescape(str(venue.get("venue", "") or "")))
                street = clean_ws(html.unescape(str(venue.get("address", "") or "")))
                city = clean_ws(html.unescape(str(venue.get("city", "") or "")))
                state = clean_ws(str(venue.get("stateprovince", "") or venue.get("state", "") or ""))
                zip_code = clean_ws(str(venue.get("zip", "") or ""))

                addr_parts = ", ".join(p for p in (street, city, f"{state} {zip_code}".strip()) if p)
                # Skip the venue prefix when it just repeats the street address.
                if venue_name and venue_name.lower() in (street.lower(), city.lower()):
                    venue_name = ""
                location = f"{venue_name} - {addr_parts}" if (venue_name and addr_parts) else (addr_parts or venue_name)
                city_state = f"{city}, {state}" if (city and state) else ""

                description_html = str(item.get("description", "") or "")
                description = clean_ws(BeautifulSoup(description_html, "html.parser").get_text(" ")) if description_html else ""
                cost = clean_ws(str(item.get("cost", "") or ""))
                if cost:
                    description = clean_ws(f"{description} Cost: {cost}")

                key = (title.lower(), start_dt.isoformat()[:16], city.lower())
                if key in seen:
                    continue
                seen.add(key)

                out.append({
                    "title": title,
                    "start_dt": start_dt,
                    "end_dt": end_dt or (start_dt + timedelta(hours=2)),
                    "location": location,
                    "city_state": city_state,
                    "geocode_query": f"{zip_code[:5]}, USA" if re.match(r"^\d{5}", zip_code) else (city_state or None),
                    "description": description[:2000],
                    "url": clean_ws(str(item.get("url", "") or item.get("website", "") or "")),
                    "source": source_name,
                    "bypass_automotive_filter": bypass,
                })

            total_pages = int(payload.get("total_pages", 1) or 1)
            if page >= total_pages:
                break
            page += 1
            time.sleep(0.3)

        log(f"   Tribe API {source_name} cat={category or 'all'}: running total {len(out)} events")

    if diagnostics.get("reason") is None and not out:
        diagnostics["reason"] = "no_results_from_search"
    return out


def _extract_wordpress_event_datetimes(row, fallback_start_dt: Optional[datetime]) -> Tuple[Optional[datetime], Optional[datetime]]:
    """
    Extract start/end datetimes from WordPress/The Events Calendar list rows.
    These rows often expose a <time datetime="YYYY-MM-DD"> value (date-only)
    while the visible text contains the true time range.
    """
    time_text = ""
    dt_block = row.select_one(".tribe-events-calendar-list__event-datetime") if row else None
    if dt_block:
        time_text = clean_ws(dt_block.get_text(" ", strip=True))

    if not time_text and row:
        time_el = row.select_one("time")
        if time_el:
            time_text = clean_ws(time_el.get_text(" ", strip=True))

    start_dt = fallback_start_dt
    end_dt = None
    if not start_dt or not time_text:
        return start_dt, end_dt

    date_prefix = start_dt.strftime("%Y-%m-%d")

    time_token = r"\d{1,2}(?::\d{2})?"
    meridiem_token = r"(?:\s*[AaPp]\.?[Mm]\.?)?"

    # Handle common variants:
    # - 8:00 am - 10:30 am EST
    # - 8:00am - 10:30 EST (missing second am/pm)
    # - 8 am to 10 am
    range_match = re.search(
        rf"({time_token}{meridiem_token})\s*(?:-|–|—|to)\s*({time_token}{meridiem_token})",
        time_text,
    )
    single_match = re.search(rf"\b({time_token}\s*[AaPp]\.?[Mm]\.?)\b", time_text)

    parsed_start = None
    parsed_end = None
    if range_match:
        start_raw = clean_ws(range_match.group(1))
        end_raw = clean_ws(range_match.group(2))

        # If one side is missing am/pm, inherit it from the other side.
        meridiem_pattern = r"[AaPp]\.?[Mm]\.?"
        start_has_meridiem = bool(re.search(meridiem_pattern, start_raw))
        end_has_meridiem = bool(re.search(meridiem_pattern, end_raw))
        if start_has_meridiem and not end_has_meridiem:
            inferred_meridiem = re.search(meridiem_pattern, start_raw).group(0)
            end_raw = clean_ws(f"{end_raw} {inferred_meridiem}")
        elif end_has_meridiem and not start_has_meridiem:
            inferred_meridiem = re.search(meridiem_pattern, end_raw).group(0)
            start_raw = clean_ws(f"{start_raw} {inferred_meridiem}")

        parsed_start = parse_dt(f"{date_prefix} {start_raw}")
        parsed_end = parse_dt(f"{date_prefix} {end_raw}")
    elif single_match:
        parsed_start = parse_dt(f"{date_prefix} {single_match.group(1)}")

    # Use parsed clock times when they are available; this prevents date-only
    # <time datetime="YYYY-MM-DD"> values from becoming midnight defaults.
    if parsed_start:
        start_dt = parsed_start
    if parsed_end:
        end_dt = parsed_end

    return start_dt, end_dt


def collect_ics(source: dict, diagnostics: Optional[dict] = None) -> List[dict]:
    diagnostics = diagnostics if diagnostics is not None else {}
    source_name = source.get("name", "ICS")
    source_url_raw = clean_ws(source.get("url", ""))
    source_url = source_url_raw
    if source_url.startswith("webcal://"):
        source_url = "https://" + source_url[len("webcal://") :]

    now_et = datetime.now(tz=tz.gettz("America/New_York"))
    future_only = clean_ws(str(source.get("future_only", ""))).lower() in {"1", "true", "yes", "y"}

    headers = {"User-Agent": "cincy-car-events-bot/1.0 (github actions)"}
    try:
        r = requests.get(source_url, headers=headers, timeout=45)
    except Exception as ex:
        log(f"⚠️ ICS download failed: source={source_name} url={source_url} error={ex}")
        diagnostics["reason"] = "parse_failed"
        raise

    log(
        f"ℹ️ ICS fetch: source={source_name} original_url_scheme={'webcal' if source_url_raw.startswith('webcal://') else 'http'} "
        f"resolved_url={source_url} http_status={r.status_code} bytes_downloaded={len(r.content or b'')}"
    )
    r.raise_for_status()

    cal = Calendar.from_ical(r.content)
    out = []
    total_vevents = 0
    skipped_past = 0
    expanded_recurring = 0

    def _normalize_ics_dt(value) -> Optional[datetime]:
        if not value:
            return None
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=tz.gettz("America/New_York"))
            return value.astimezone(tz.gettz("America/New_York"))
        return datetime.combine(value, datetime.min.time()).replace(tzinfo=tz.gettz("America/New_York"))

    def _collect_rdate_datetimes(component) -> List[datetime]:
        result = []
        rdate_prop = component.get("rdate")
        if not rdate_prop:
            return result
        props = rdate_prop if isinstance(rdate_prop, list) else [rdate_prop]
        for prop in props:
            for dt_val in getattr(prop, "dts", []):
                normalized = _normalize_ics_dt(getattr(dt_val, "dt", None))
                if normalized:
                    result.append(normalized)
        return result

    def _collect_exdate_datetimes(component) -> set:
        result = set()
        exdate_prop = component.get("exdate")
        if not exdate_prop:
            return result
        props = exdate_prop if isinstance(exdate_prop, list) else [exdate_prop]
        for prop in props:
            for dt_val in getattr(prop, "dts", []):
                normalized = _normalize_ics_dt(getattr(dt_val, "dt", None))
                if normalized:
                    result.add(normalized)
        return result

    for component in cal.walk():
        if component.name != "VEVENT":
            continue
        total_vevents += 1

        title = clean_ws(str(component.get("summary", "")))
        loc = clean_ws(str(component.get("location", "")))
        url = clean_ws(str(component.get("url", ""))) or source_url

        dtstart = component.get("dtstart")
        dtend = component.get("dtend")

        if not title or not dtstart:
            continue

        start_dt = _normalize_ics_dt(dtstart.dt)
        end_dt = _normalize_ics_dt(dtend.dt if dtend else None)
        if not start_dt:
            continue
        if not end_dt:
            end_dt = start_dt + timedelta(hours=2)

        duration = end_dt - start_dt
        if duration.total_seconds() <= 0:
            duration = timedelta(hours=2)

        event_starts: List[datetime] = [start_dt]
        rrule_prop = component.get("rrule")
        if rrule_prop:
            window_start = now_et - timedelta(days=1)
            window_end = now_et + timedelta(days=730)
            try:
                rule_text = rrule_prop.to_ical().decode("utf-8")
                rule = rrulestr(rule_text, dtstart=start_dt)
                has_finite_bound = "COUNT=" in rule_text.upper() or "UNTIL=" in rule_text.upper()
                if has_finite_bound:
                    event_starts = list(rule)[:1000]
                else:
                    event_starts = list(rule.between(window_start, window_end, inc=True))
                expanded_recurring += max(0, len(event_starts) - 1)
            except Exception:
                event_starts = [start_dt]

        event_starts.extend(_collect_rdate_datetimes(component))
        exdates = _collect_exdate_datetimes(component)
        unique_starts = sorted({dt for dt in event_starts if dt not in exdates})

        for occurrence_start in unique_starts:
            if future_only and occurrence_start < now_et:
                skipped_past += 1
                continue

            out.append(
                {
                    "title": title,
                    "start_dt": occurrence_start,
                    "end_dt": occurrence_start + duration,
                    "location": loc,
                    "url": url,
                    "source": source_name,
                    # Per project policy, all ICS feeds are curated automotive calendars.
                    # Skip non-automotive keyword/exclusion filtering for every ICS event.
                    "bypass_automotive_filter": True,
                    # Trusted calendars (e.g. Joel's iCloud calendar) are never
                    # distance-dropped — geocoding venue-only names is unreliable.
                    "bypass_distance_filter": bool(source.get("bypass_distance_filter", False)),
                }
            )

    log(
        f"ℹ️ ICS parse: source={source_name} total_vevents={total_vevents} "
        f"future_only={'yes' if future_only else 'no'} skipped_past={skipped_past} "
        f"expanded_recurring={expanded_recurring} kept_future={len(out)}"
    )
    diagnostics["total_vevents"] = total_vevents
    diagnostics["skipped_past"] = skipped_past
    diagnostics["expanded_recurring"] = expanded_recurring
    if not out:
        diagnostics["reason"] = "no_results_from_search"
    return out


def _sheet_header_key(header: str) -> str:
    return re.sub(r"[^a-z0-9]", "", clean_ws(header).lower())


def _sheet_value(row: List[str], header_map: Dict[str, int], *keys: str) -> str:
    for key in keys:
        idx = header_map.get(key)
        if idx is not None and idx < len(row):
            value = clean_ws(row[idx])
            if value:
                return value
    return ""


def _parse_sheet_date_value(raw_value, now_et: datetime) -> Tuple[Optional[datetime], Optional[str]]:
    et_tz = EST_TZ
    if raw_value is None:
        return None, "empty"

    raw = clean_ws(str(raw_value))
    if not raw:
        return None, "empty"

    def _normalize(dt: Optional[datetime]) -> Optional[datetime]:
        if not dt:
            return None
        if dt.tzinfo is None:
            return dt.replace(tzinfo=et_tz)
        return dt.astimezone(et_tz)

    parsed = parse_dt(raw)
    parsed = _normalize(parsed)
    if parsed:
        if re.search(r"\b\d{4}\b", raw) is None:
            parsed = parsed.replace(year=now_et.year)
            if parsed < now_et:
                parsed = parsed.replace(year=now_et.year + 1)
        return parsed, None

    try:
        numeric = float(raw)
        if numeric > 20000:
            dt = datetime(1899, 12, 30) + timedelta(days=numeric)
            dt = dt.replace(tzinfo=et_tz)
            if dt < now_et:
                dt = dt.replace(year=dt.year + 1)
            return dt, None
    except Exception:
        pass

    for fmt in ("%d-%b", "%d %b", "%b %d", "%d/%m", "%m/%d"):
        try:
            dt = datetime.strptime(raw, fmt).replace(year=now_et.year, tzinfo=et_tz)
            if dt < now_et:
                dt = dt.replace(year=now_et.year + 1)
            return dt, None
        except Exception:
            continue

    return None, "unrecognized_date_format"


def _find_sheet_header_row(rows: List[List[str]], required_header_keys: List[str]) -> Tuple[Optional[int], Dict[str, int], List[str]]:
    for idx, row in enumerate(rows[:25]):
        normalized_row = [_sheet_header_key(c) for c in row]
        header_map = {key: i for i, key in enumerate(normalized_row) if key}
        if all(req in header_map for req in required_header_keys):
            return idx, header_map, normalized_row
    return None, {}, []


def _parse_google_sheet_events_rows(rows: List[List[str]], source_name: str, tab_name: str) -> Tuple[List[dict], Dict[str, int]]:
    if not rows:
        return [], {"rows_read": 0, "parsed_events": 0}

    now_et = datetime.now(tz=tz.gettz("America/New_York"))
    date_header_keys = [_sheet_header_key("Date"), _sheet_header_key("date")]
    title_header_keys = [_sheet_header_key("Name"), _sheet_header_key("event_name")]
    required_headers = [date_header_keys[0], title_header_keys[0]]
    header_idx, header_map, normalized_headers = _find_sheet_header_row(rows, required_headers)
    if header_idx is None:
        # Support screenshot-import schema (event_name/date/start_time/end_time/...)
        header_idx, header_map, normalized_headers = _find_sheet_header_row(rows, [date_header_keys[1], title_header_keys[1]])
    if header_idx is None:
        stats = {
            "rows_read": max(len(rows) - 1, 0),
            "parsed_events": 0,
            "skipped_missing_title": 0,
            "skipped_parse_failed": 0,
            "skipped_past": 0,
            "bad_sheet_headers": 1,
            "skip_reasons": {"bad_sheet_headers": 1},
            "header_keys_present": [_sheet_header_key(c) for c in (rows[0] if rows else [])],
        }
        return [], stats

    stats_counter: Counter = Counter()
    out: List[dict] = []
    data_rows = rows[header_idx + 1 :]

    for row_offset, row in enumerate(data_rows, start=header_idx + 2):
        title = _sheet_value(row, header_map, *title_header_keys)
        date_raw = _sheet_value(row, header_map, *date_header_keys)
        row_start_time = _sheet_value(row, header_map, _sheet_header_key("start_time"), _sheet_header_key("starttime"))
        row_end_time = _sheet_value(row, header_map, _sheet_header_key("end_time"), _sheet_header_key("endtime"))
        if not title:
            stats_counter["missing_title"] += 1
            continue
        if not date_raw:
            stats_counter["missing_date"] += 1
            continue

        date_for_parse = date_raw
        if row_start_time:
            date_for_parse = f"{date_raw} {row_start_time}"

        start_dt, parse_reason = _parse_sheet_date_value(date_for_parse, now_et)
        if not start_dt:
            stats_counter["parse_failure"] += 1
            log(f"⚠️ Sheet row skipped: row={row_offset} raw_date='{date_for_parse}' reason={parse_reason}")
            continue

        has_explicit_time = bool(re.search(r"\d{1,2}:\d{2}", date_for_parse)) or "t" in date_for_parse.lower()
        if not has_explicit_time:
            start_dt = start_dt.replace(hour=9, minute=0, second=0, microsecond=0)
        end_dt = start_dt + timedelta(hours=2)
        if row_end_time:
            end_dt_parsed, _ = _parse_sheet_date_value(f"{date_raw} {row_end_time}", now_et)
            if end_dt_parsed and end_dt_parsed > start_dt:
                end_dt = end_dt_parsed

        if start_dt < now_et:
            stats_counter["past_event"] += 1
            continue

        location = _sheet_value(
            row,
            header_map,
            _sheet_header_key("City / Where You’d Be"),
            _sheet_header_key("City / Where You'd Be"),
            _sheet_header_key("venue_name"),
            _sheet_header_key("street_address"),
            _sheet_header_key("city"),
        )
        url = _sheet_value(row, header_map, _sheet_header_key("Link"))
        event_type = _sheet_value(row, header_map, _sheet_header_key("Event Type"))
        cost = _sheet_value(row, header_map, _sheet_header_key("Cost"))
        row_source = _sheet_value(row, header_map, _sheet_header_key("Source")) or source_name

        out.append(
            {
                "title": title,
                "start_dt": start_dt,
                "end_dt": end_dt,
                "location": location,
                "url": url,
                "source": row_source,
                "source_tab": tab_name,
                "event_type": event_type,
                "cost": cost,
            }
        )

    stats = {
        "rows_read": len(data_rows),
        "parsed_events": len(out),
        "skipped_missing_title": stats_counter.get("missing_title", 0),
        "skipped_missing_date": stats_counter.get("missing_date", 0),
        "skipped_parse_failed": stats_counter.get("parse_failure", 0),
        "skipped_past": stats_counter.get("past_event", 0),
        "bad_sheet_headers": 0,
        "skip_reasons": dict(stats_counter),
        "header_keys_present": normalized_headers,
    }
    return out, stats


def _fetch_public_sheet_rows_csv(sheet_id: str, tab_name: str) -> Optional[List[List[str]]]:
    if tab_name:
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={tab_name}"
    else:
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    try:
        resp = requests.get(csv_url, headers=DEFAULT_HTTP_HEADERS, timeout=30)
        if resp.status_code in (401, 403):
            return None
        resp.raise_for_status()
        parsed = list(csv.reader(resp.text.splitlines()))
        return parsed if parsed else None
    except Exception:
        return None


def _fetch_sheet_rows_via_api(sheet_id: str, tab_name_candidates: List[str]) -> List[Tuple[str, List[List[str]]]]:
    creds = get_google_credentials()
    if not creds:
        return []
    sheets = build("sheets", "v4", credentials=creds)

    metadata = sheets.spreadsheets().get(spreadsheetId=sheet_id, includeGridData=False).execute()
    sheet_entries = metadata.get("sheets", []) or []
    available_tabs = [clean_ws((entry.get("properties") or {}).get("title", "")) for entry in sheet_entries]

    ordered_tabs: List[str] = []
    for tab in tab_name_candidates:
        if tab in available_tabs and tab not in ordered_tabs:
            ordered_tabs.append(tab)
    for tab in available_tabs:
        if tab and tab not in ordered_tabs:
            ordered_tabs.append(tab)

    rows_by_tab: List[Tuple[str, List[List[str]]]] = []
    for tab in ordered_tabs:
        resp = sheets.spreadsheets().values().get(spreadsheetId=sheet_id, range=f"{tab}!A1:Z4000").execute()
        rows_by_tab.append((tab, resp.get("values", [])))
    return rows_by_tab


def collect_google_sheet_events_import(source: dict, diagnostics: Optional[dict] = None) -> List[dict]:
    diagnostics = diagnostics if diagnostics is not None else {}
    sheet_id = extract_google_spreadsheet_id(clean_ws(source.get("spreadsheet_id") or APEX_IMPORT_SPREADSHEET_ID))
    source_name = source.get("name", "Google Sheet Events Import")
    tab_candidates = ["Rallies", "Events", "Extracted Events", "Sheet1"]
    preferred_tab = clean_ws(source.get("tab") or "")
    if preferred_tab:
        tab_candidates = [preferred_tab] + [t for t in tab_candidates if t != preferred_tab]

    if not sheet_id:
        log("⚠️ Sheet import skipped: missing APEX_IMPORT_SPREADSHEET_ID.")
        diagnostics["reason"] = "missing_env"
        return []

    tab_rows_to_try: List[Tuple[str, List[List[str]]]] = []
    tried_tabs: List[str] = []
    for tab in tab_candidates:
        tried_tabs.append(tab)
        rows = _fetch_public_sheet_rows_csv(sheet_id, tab) or []
        if rows:
            tab_rows_to_try.append((tab, rows))

    if not tab_rows_to_try:
        try:
            tab_rows_to_try = _fetch_sheet_rows_via_api(sheet_id, tab_candidates)
        except HttpError as ex:
            status = getattr(ex, "status_code", None) or getattr(getattr(ex, "resp", None), "status", None)
            reason = f"http_error_{status}" if status else "http_error"
            log(f"⚠️ Sheet import API error for spreadsheet={sheet_id}: {ex}")
            diagnostics["reason"] = reason
            return []
        except Exception as ex:
            log(f"⚠️ Sheet import failed for spreadsheet={sheet_id}: {ex}")
            diagnostics["reason"] = "parse_failed"
            return []

    if not tab_rows_to_try:
        log(f"⚠️ Sheet import returned no rows for spreadsheet={sheet_id}; continuing.")
        diagnostics["reason"] = "no_results_from_search"
        return []

    parsed: List[dict] = []
    rows_seen = False
    tab_stats: List[Dict[str, object]] = []
    selected_tab = ""
    for candidate_tab, candidate_rows in tab_rows_to_try:
        if not candidate_rows:
            continue
        rows_seen = True
        candidate_parsed, candidate_stats = _parse_google_sheet_events_rows(candidate_rows, source_name, candidate_tab)
        tab_stats.append({"tab": candidate_tab, "stats": candidate_stats})
        if candidate_parsed and not selected_tab:
            selected_tab = candidate_tab
        parsed.extend(candidate_parsed)

    if not rows_seen:
        log(f"⚠️ Sheet import returned no rows for spreadsheet={sheet_id}; continuing.")
        diagnostics["reason"] = "no_results_from_search"
        return []

    if not tab_stats:
        diagnostics["reason"] = "parse_failed"
        return []

    primary_stats = next((entry["stats"] for entry in tab_stats if entry["tab"] == selected_tab), tab_stats[0]["stats"])
    total_rows = sum(int((entry["stats"] or {}).get("rows_read", 0)) for entry in tab_stats)
    total_parsed = sum(int((entry["stats"] or {}).get("parsed_events", 0)) for entry in tab_stats)
    total_past = sum(int((entry["stats"] or {}).get("skipped_past", 0)) for entry in tab_stats)
    total_parse_failed = sum(int((entry["stats"] or {}).get("skipped_parse_failed", 0)) for entry in tab_stats)
    merged_skip_reasons: Counter = Counter()
    for entry in tab_stats:
        merged_skip_reasons.update(Counter((entry["stats"] or {}).get("skip_reasons") or {}))

    reason = ""
    if total_parsed == 0 and any((entry["stats"] or {}).get("bad_sheet_headers") for entry in tab_stats):
        reason = "bad_sheet_headers"
    elif total_parsed == 0:
        reason = "parse_failed" if total_parse_failed > 0 else "no_results_from_search"

    top_skip_reasons = merged_skip_reasons.most_common(5)
    tab_summaries = ", ".join(
        f"{entry['tab']}={int((entry['stats'] or {}).get('parsed_events', 0))}"
        for entry in tab_stats
    )
    log(
        f"✅ Sheet import loaded: rows={total_rows} tab={selected_tab or tried_tabs[0]} "
        f"parsed_events={total_parsed} future_only=yes "
        f"skipped_past={total_past} skipped_parse_failed={total_parse_failed}"
    )
    log(f"ℹ️ Sheet import per-tab parsed_events: {tab_summaries}")
    if top_skip_reasons:
        log("ℹ️ Sheet import top skip reasons: " + ", ".join(f"{k}={v}" for k, v in top_skip_reasons))

    diagnostics["reason"] = reason
    diagnostics["sheet_rows_read"] = total_rows
    diagnostics["parsed_events"] = total_parsed
    diagnostics["sheet_skip_reasons"] = dict(merged_skip_reasons)
    diagnostics["sheet_headers"] = primary_stats.get("header_keys_present", [])
    diagnostics["sheet_tab"] = selected_tab or (tried_tabs[0] if tried_tabs else "")
    diagnostics["tab_candidates"] = tab_candidates
    diagnostics["sheet_tabs"] = [
        {"tab": entry["tab"], "parsed_events": int((entry["stats"] or {}).get("parsed_events", 0))}
        for entry in tab_stats
    ]
    return parsed


def collect_facebook_page_events(source: dict, diagnostics: Optional[dict] = None) -> List[dict]:
    targets = source.get("_facebook_targets") or load_facebook_targets()
    pages = targets.get("page", [])
    events = collect_facebook_events_from_pages(pages, diagnostics=diagnostics)
    source_name = source.get("name", "Facebook Page Events")
    for e in events:
        e["source"] = source_name
    return events


def facebook_scroll_enabled() -> bool:
    """Feed scrolling is opt-in: it needs Playwright + a saved logged-in session."""
    return clean_ws(os.getenv("ENABLE_FACEBOOK_SCROLL", "0")).lower() in {"1", "true", "yes", "y", "on"}


def _facebook_event_dict_from_parsed(parsed: dict, fallback_url: str, source_name: str) -> Optional[dict]:
    """Adapt a facebook_event_parser payload into the pipeline's raw-event dict shape."""
    if not parsed or not parsed.get("title") or not parsed.get("start_dt"):
        return None
    return {
        "title": parsed.get("title", ""),
        "start_dt": parsed.get("start_dt"),
        "end_dt": parsed.get("end_dt"),
        "location": simplify_location(parsed.get("location", ""), parsed.get("address", "")),
        "url": parsed.get("canonical_url") or parsed.get("url") or fallback_url,
        "source": source_name,
        "host": parsed.get("host", ""),
    }


def collect_facebook_events_scroll(source: dict, url_cache: Dict[str, dict], diagnostics: Optional[dict] = None) -> List[dict]:
    """Scroll the logged-in Facebook home feed and harvest event links.

    Discovery happens in a headless Playwright session authenticated with a saved
    ``storageState`` (env ``FACEBOOK_SESSION_STATE``); parsing, caching, and all
    downstream filtering (automotive focus, distance caps, date window) reuse the
    existing pipeline. Opt-in via ``ENABLE_FACEBOOK_SCROLL=1`` — otherwise this is
    a no-op so the default cron is unaffected.
    """
    diagnostics = diagnostics if diagnostics is not None else {}
    diagnostics.setdefault("raw_candidates", 0)
    diagnostics.setdefault("parse_failures", 0)
    source_name = source.get("name", "Facebook Feed Scroll")

    if not facebook_scroll_enabled():
        diagnostics["reason"] = "disabled_by_flag"
        log("ℹ️ Facebook feed scroll disabled; set ENABLE_FACEBOOK_SCROLL=1 to enable.")
        return []

    try:
        from scripts import facebook_feed_scraper as fb_scraper
    except Exception as ex:
        diagnostics["reason"] = "scraper_import_failed"
        log(f"⚠️ Facebook feed scraper unavailable: {ex}")
        return []

    storage_state = fb_scraper.load_storage_state()
    if not storage_state:
        diagnostics["reason"] = "disabled_missing_session"
        log("⚠️ Skipping Facebook feed scroll: no saved session (set FACEBOOK_SESSION_STATE).")
        return []

    def _tune(key: str, env: str, default: int) -> int:
        try:
            return int(source.get(key, os.getenv(env, default)) or default)
        except (TypeError, ValueError):
            return default

    scrolls = _tune("scrolls", "FACEBOOK_SCROLL_COUNT", 20)
    scroll_pause_ms = _tune("scroll_pause_ms", "FACEBOOK_SCROLL_PAUSE_MS", 2000)
    max_events = _tune("max_events", "FACEBOOK_SCROLL_MAX_EVENTS", 40)
    feed_url = clean_ws(source.get("feed_url") or os.getenv("FACEBOOK_FEED_URL") or fb_scraper.DEFAULT_FEED_URL)

    result = fb_scraper.scrape_feed(
        storage_state=storage_state,
        feed_url=feed_url,
        scrolls=scrolls,
        scroll_pause_ms=scroll_pause_ms,
        max_events=max_events,
        logger=log,
    )

    if not result.event_urls:
        diagnostics["reason"] = result.reason or "no_results_from_search"
        return []

    FACEBOOK_COVERAGE["urls_discovered"] = int(FACEBOOK_COVERAGE.get("urls_discovered", 0) or 0) + result.discovered

    out: List[dict] = []
    now = datetime.now(tz=tz.gettz("America/New_York"))
    for raw_url in result.event_urls:
        event_url = normalize_facebook_event_url(raw_url) or raw_url
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

        ev = None
        html = result.pages.get(raw_url) or result.pages.get(event_url)
        if html:
            parsed, _reason = parse_facebook_event_html(event_url, html)
            ev = _facebook_event_dict_from_parsed(parsed, event_url, source_name)

        if ev is None:
            # Fall back to an unauthenticated fetch (public events are often readable).
            parsed_page, fail_reason = parse_facebook_event_page(event_url, log)
            ev = _facebook_event_dict_from_parsed(parsed_page, event_url, source_name)
            if ev is None:
                diagnostics["parse_failures"] += 1
                FACEBOOK_COVERAGE["urls_failed"] = int(FACEBOOK_COVERAGE.get("urls_failed", 0) or 0) + 1
                FACEBOOK_COVERAGE["failure_reasons"][fail_reason] += 1
                url_cache[event_url] = {"fetched_at_iso": now.isoformat(), "event": None, "failure_reason": fail_reason}
                continue

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

    if not out and diagnostics.get("reason") is None:
        diagnostics["reason"] = "parsing_schema_changed"
    log(f"📘 Facebook feed scroll: discovered={result.discovered} parsed={len(out)} scrolls={result.scrolls_done}")
    return out


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
    url = "https://serpapi.com/search.json"
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

        if engine == "google_events":
            rows, rows_key, payload_keys = extract_google_events_rows(data)
            meta = data.get("search_metadata") if isinstance(data, dict) else {}
            meta_status = clean_ws(str((meta or {}).get("status", ""))) or "unknown"
            error_value = clean_ws(str(data.get("serpapi_error") or data.get("error") or "")) if isinstance(data, dict) else ""
            log(
                f'ℹ️ SerpAPI google_events: status={response.status_code} '
                f'meta_status={meta_status} count={len(rows)} q="{query}" location="{req_location}"'
            )
            if error_value:
                log(f"⚠️ SerpAPI google_events error: q={query} error={error_value}")
            if not rows_key:
                log(
                    f"⚠️ SerpAPI google_events missing expected event key for q={query}; "
                    f"available_keys={payload_keys or '(none)'}"
                )
        else:
            log_serpapi_request_meta(
                query_name=query_name,
                q=query,
                engine=engine,
                location=req_location,
                htichips=req_htichips,
                start=start,
                payload=data,
            )
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


def extract_google_events_rows(payload: dict) -> Tuple[List[dict], Optional[str], str]:
    """Return rows + selected key + available keys for google_events payload."""
    if not isinstance(payload, dict):
        return [], None, ""

    rows = payload.get("events_results")
    if isinstance(rows, list):
        return rows, "events_results", ", ".join(sorted(payload.keys()))

    return [], None, ", ".join(sorted(payload.keys()))


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
                    "location": simplify_location(parsed_page.get('location', ''), parsed_page.get('address', '')),
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
                "description": clean_ws(str(obj.get("description") or ""))[:2000],
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

def enrich_event_from_source_url(url: str) -> Optional[dict]:
    """Best-effort source page enrichment for sparse SerpAPI google_events rows."""
    u = clean_ws(url)
    if not u:
        return None
    if u in URL_ENRICH_CACHE:
        return URL_ENRICH_CACHE[u]

    try:
        resp = requests.get(u, headers=DEFAULT_HTTP_HEADERS, timeout=20)
        resp.raise_for_status()
        html_text = resp.text or ""
        parsed = parse_schema_org_events_from_html(u, html_text)
        best = parsed[0] if parsed else None
        URL_ENRICH_CACHE[u] = best
        return best
    except Exception as ex:
        URL_ENRICH_CACHE[u] = None
        log(f"⚠️ SerpAPI google_events enrichment skipped: {u} :: {clean_ws(str(ex))[:160]}")
        return None


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
    diagnostics.setdefault("query_errors", 0)

    if not SERPAPI_API_KEY:
        diagnostics["reason"] = "missing_env"
        log("ℹ️ SerpAPI disabled; skipping google_events collector.")
        return []

    source_name = source.get("name", "Google Events (SerpAPI) [serpapi_google_events]")
    location = (
        clean_ws(str(source.get("location", "") or ""))
        or clean_ws(os.getenv("SERPAPI_LOCATION", SERPAPI_LOCATION or "Cincinnati, OH"))
        or "Cincinnati, OH"
    )
    gl = clean_ws(os.getenv("SERPAPI_GL", SERPAPI_GL or "us")) or "us"
    hl = clean_ws(os.getenv("SERPAPI_HL", SERPAPI_HL or "en")) or "en"
    base_filter = clean_ws(os.getenv("SERPAPI_EVENTS_DATE_FILTER", SERPAPI_EVENTS_DATE_FILTER or "date:month")) or "date:month"

    query = clean_ws(source.get("query") or f"car events in {location}") or f"car events in {location}"
    htichips_attempts = [base_filter]
    if base_filter == "date:month":
        htichips_attempts.append("date:next_month")

    out: List[dict] = []
    seen = set()
    et_tz = EST_TZ

    for idx, htichips in enumerate(htichips_attempts):
        _, rows = serpapi_search(
            query,
            max_results=60,
            page_size=10,
            return_payload=True,
            max_pages=4,
            engine="google_events",
            query_name="serpapi_google_events",
            location=location,
            gl=gl,
            hl=hl,
            htichips=htichips,
        )

        # Debug probe (extra API call) only when the paged search came back empty.
        if not rows:
            params = {
                "api_key": SERPAPI_API_KEY,
                "engine": "google_events",
                "q": query,
                "location": location,
                "gl": gl,
                "hl": hl,
                "htichips": htichips,
                "start": 0,
                "num": 10,
            }
            payload = {}
            try:
                probe_resp = requests.get("https://serpapi.com/search.json", params=params, headers=DEFAULT_HTTP_HEADERS, timeout=45)
                if probe_resp.ok:
                    payload = probe_resp.json() if probe_resp.text else {}
                else:
                    payload = {"error": f"http_{probe_resp.status_code}"}
            except Exception as ex:
                payload = {"error": str(ex)}

            meta = payload.get("search_metadata") if isinstance(payload, dict) else {}
            status = clean_ws(str((meta or {}).get("status", "")))
            top_level_keys = sorted(payload.keys()) if isinstance(payload, dict) else []
            payload_rows, payload_key, _ = extract_google_events_rows(payload if isinstance(payload, dict) else {})

            log(
                f"ℹ️ SerpAPI google_events meta: status={status or 'unknown'} keys={top_level_keys} "
                f"parsed_key={payload_key or 'missing'} len(events_results)={len(payload_rows)}"
            )
            if status != "Success":
                log(
                    f"⚠️ SerpAPI google_events non-success: status={status or 'unknown'} "
                    f"error={clean_ws(str((payload or {}).get('error') or (payload or {}).get('serpapi_error') or 'none'))} "
                    f"search_parameters={json.dumps((payload or {}).get('search_parameters') or params, sort_keys=True)}"
                )
            diagnostics["search_status"] = status or "unknown"
            diagnostics["events_results_len"] = len(payload_rows)
            diagnostics["top_level_keys"] = top_level_keys

        diagnostics["raw_candidates"] += len(rows)

        for item in rows:
            if not isinstance(item, dict):
                diagnostics["parse_failures"] += 1
                continue

            title = clean_ws(item.get("title") or item.get("name") or "")
            when = ""
            date_entry = item.get("date")
            if isinstance(date_entry, dict):
                when = clean_ws(date_entry.get("when") or date_entry.get("start_date") or date_entry.get("date") or "")
            when = when or clean_ws(item.get("when") or "")

            if not title:
                diagnostics["parse_failures"] += 1
                continue

            start_dt = parse_dt(when) if when else None
            if start_dt and start_dt.tzinfo is None:
                start_dt = start_dt.replace(tzinfo=et_tz)
            elif start_dt:
                start_dt = start_dt.astimezone(et_tz)
            if not start_dt:
                diagnostics["parse_failures"] += 1
                log(f"⚠️ SerpAPI google_events date parse failed: title='{title[:80]}' when='{when}'")
                continue

            end_dt = start_dt + timedelta(hours=2)
            venue = _stringify_location_like(item.get("venue") or item.get("event_location") or "")
            address = _stringify_location_like(item.get("address") or item.get("location") or "")
            location_text = simplify_location(venue, address)
            url = clean_ws(item.get("link") or item.get("event_link") or item.get("website") or item.get("url") or "")

            enriched = enrich_event_from_source_url(url) if url else None
            if enriched:
                title = clean_ws(enriched.get("title") or title)
                start_dt = enriched.get("start_dt") or start_dt
                end_dt = enriched.get("end_dt") or end_dt
                location_text = clean_ws(enriched.get("location") or location_text)

            dedupe_key = (title.lower(), start_dt.isoformat()[:16], location_text.lower())
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
                    "source": source_name,
                    "bypass_automotive_filter": bool(source.get("bypass_automotive_filter", False)),
                }
            )

        # Accumulate across all date windows (this month AND next month);
        # the `seen` set dedupes overlapping results.
        if idx == 0 and len(htichips_attempts) > 1:
            log(f"ℹ️ SerpAPI google_events: also querying htichips={htichips_attempts[1]} (running total {len(out)})")

    if diagnostics.get("reason") is None and not out:
        diagnostics["reason"] = "parse_failed" if diagnostics.get("parse_failures", 0) > 0 else "no_results_from_search"

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

    # Google frequently ignores `site:` operators in API results, returning junk
    # domains (car dealers, reddit) that 403 or parse to nothing. Enforce the
    # site: restriction ourselves when the query declares one.
    allowed_domains = set()
    for q in queries:
        for m in re.finditer(r"site:([A-Za-z0-9.-]+)", q):
            allowed_domains.add(m.group(1).lower().removeprefix("www."))

    links: List[str] = []
    for q in queries[:20]:
        links.extend(serpapi_search(q, max_results=max_results))
        time.sleep(0.2)

    links = list(dict.fromkeys(clean_ws(u) for u in links if clean_ws(u)))
    if allowed_domains:
        def _domain_ok(u: str) -> bool:
            host = (urlparse(u).netloc or "").lower().removeprefix("www.")
            return any(host == d or host.endswith("." + d) for d in allowed_domains)

        before = len(links)
        links = [u for u in links if _domain_ok(u)]
        dropped = before - len(links)
        if dropped:
            log(f"🧹 {source_name}: dropped {dropped}/{before} off-domain results (site: operator not honored)")
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
    start = clean_ws(ev.start_iso)
    end = clean_ws(ev.end_iso)
    place = re.sub(r"\s+", " ", clean_ws(ev.city_state or ev.location).lower()).strip()
    url = clean_ws(ev.url).lower()
    return f"{title}||{start}||{end}||{place}||{url}"


def log_counter_top(title: str, counter: Counter, top_n: int = 8) -> None:
    if not counter:
        return
    top = ", ".join(f"{k}={v}" for k, v in counter.most_common(top_n))
    log(f"📊 {title}: {top}")


def log_source_health_summary(source_run_stats: List[dict]) -> None:
    """Human-readable summary of what worked vs what did not."""
    log("📋 Collector health summary:")
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
        location = normalize_location_for_output(e.get("location", ""), e.get("city_state", ""))
        url = clean_ws(e.get("url", ""))
        source = clean_ws(e.get("source", "")) or "unknown_source"
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

        bypass_automotive_filter = bool(e.get("bypass_automotive_filter"))
        if not bypass_automotive_filter:
            allowed, auto_reason = evaluate_automotive_focus_event(
                title, location, source, url, cfg, description=clean_ws(str(e.get("description", "") or ""))
            )
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

        city_state = clean_ws(str(e.get("city_state", "") or "")) or guess_city_state(location)
        # Sources can supply an unambiguous geocode hint (e.g. "45014, USA");
        # otherwise only geocode text reliable enough to act on (ZIP or city+state).
        query = clean_ws(str(e.get("geocode_query", "") or "")) or reliable_geocode_query(location, city_state)
        latlon = geocode(query, geocache) if query else None
        lat = lon = None
        miles = None
        if latlon:
            lat, lon = latlon
            miles = miles_from_home(lat, lon, home_lat, home_lon)

        cat = categorize(title, location, cfg)

        # Trusted sources (bypass_distance_filter) are never distance-dropped.
        if miles is not None and not bool(e.get("bypass_distance_filter")):
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

        callout = clean_ws(str(e.get("callout", ""))) or detect_registration_callout(
            title,
            e.get("description", ""),
            ticket_url=str(e.get("ticket_uri", "") or ""),
        )

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
                callout=callout,
                # closest_city is intentionally left blank here; main() backfills
                # it with ZIP-priority geocoding for better accuracy.
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
            if not cur.callout and ev.callout:
                cur.callout = ev.callout
            if not cur.closest_city and ev.closest_city:
                cur.closest_city = ev.closest_city
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


# -------------------------
# Export enrichment
#   1. open each event link and confirm/correct its date & time
#   2. look up a full street address when the source only gave a venue/city
#   3. pull Facebook "going/interested" counts and score every event's size
#   4. merge same-day near-duplicates that slipped past exact dedupe
# -------------------------

US_STATE_NAME_TO_ABBR = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA",
    "colorado": "CO", "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA",
    "hawaii": "HI", "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD", "massachusetts": "MA",
    "michigan": "MI", "minnesota": "MN", "mississippi": "MS", "missouri": "MO", "montana": "MT",
    "nebraska": "NE", "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM",
    "new york": "NY", "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT",
    "virginia": "VA", "washington": "WA", "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY",
    "district of columbia": "DC",
}


def _has_full_street_address(text: str) -> bool:
    """True when text already looks like 'NN Street, City, ST ZIP'."""
    return bool(US_ADDR_FULL_RE.search(clean_ws(text)))


def _span_days(start_iso: str, end_iso: str) -> int:
    start = parse_iso_datetime_safe(start_iso)
    end = parse_iso_datetime_safe(end_iso)
    if not start or not end:
        return 1
    return max(1, (end.date() - start.date()).days + 1)


def geocode_full_address(location: str, city_state: str, cache: Dict[str, dict]) -> Tuple[str, Optional[Tuple[float, float]]]:
    """Best-effort full street address via Nominatim.

    Returns (formatted_address, latlon). formatted_address is "" when no trusted
    street-level match exists (caller then keeps the venue text). Cached under an
    'addr::' namespace inside the geocode cache. MUST be called serially —
    Nominatim's usage policy caps requests at 1/second."""
    loc = clean_ws(location)
    if not loc:
        return "", None

    existing = US_ADDR_FULL_RE.search(loc)
    if existing:
        return _format_address_candidate(existing.group(1), require_zip=True) or clean_ws(existing.group(1)), None

    expected_state = ""
    m = re.search(rf",\s*({US_STATE_ABBR_RE})\b", f"{city_state} {loc}", flags=re.IGNORECASE)
    if m:
        expected_state = m.group(1).upper()

    query = loc
    if city_state and clean_ws(city_state).lower() not in loc.lower():
        query = f"{loc}, {city_state}"
    key = f"addr::{query.lower()}"
    if key in cache:
        v = cache.get(key)
        if not v:
            return "", None
        latlon = tuple(v["latlon"]) if v.get("latlon") else None
        return v.get("formatted", ""), latlon

    try:
        r = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "jsonv2", "limit": 1, "addressdetails": 1, "countrycodes": "us"},
            headers={"User-Agent": "cincy-car-events-bot/1.0 (github actions)"},
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
        time.sleep(1.1)  # Nominatim politeness: <=1 req/sec
    except Exception as ex:
        cache[key] = None
        log(f"⚠️ Address lookup failed for '{query}': {clean_ws(str(ex))[:120]}")
        return "", None

    if not data:
        cache[key] = None
        return "", None

    top = data[0]
    addr = top.get("address", {}) or {}
    try:
        latlon = (float(top["lat"]), float(top["lon"]))
    except Exception:
        latlon = None

    state_raw = clean_ws(str(addr.get("state", "")))
    state = US_STATE_NAME_TO_ABBR.get(state_raw.lower(), state_raw.upper() if len(state_raw) == 2 else "")
    if expected_state and state and state != expected_state:
        # Wrong state -> untrusted; keep coords for distance but no address text.
        cache[key] = {"formatted": "", "latlon": list(latlon) if latlon else None}
        return "", latlon

    road = clean_ws(str(addr.get("road", "")))
    house = clean_ws(str(addr.get("house_number", "")))
    city = clean_ws(str(
        addr.get("city") or addr.get("town") or addr.get("village")
        or addr.get("hamlet") or addr.get("municipality") or addr.get("county") or ""
    ))
    zipc = clean_ws(str(addr.get("postcode", "")))[:10]

    if not road:
        # No street-level detail -> don't fabricate a street address.
        cache[key] = {"formatted": "", "latlon": list(latlon) if latlon else None}
        return "", latlon

    street = clean_ws(f"{house} {road}").strip()
    tail = clean_ws(f"{state} {zipc}").strip()
    formatted = ", ".join(p for p in [street, city, tail] if clean_ws(p))
    cache[key] = {"formatted": formatted, "latlon": list(latlon) if latlon else None}
    return formatted, latlon


_DATE_VERIFY_SKIP_HOSTS = ("facebook.com", "instagram.com", "meetup.com")


def verify_event_datetime_via_link(event: dict, url_cache: Dict[str, dict], allow_fb_graph: bool = True) -> Optional[dict]:
    """Open the event link and extract an authoritative date/time + details.

    Returns a dict with start_iso/end_iso (+ optional location/description/
    attendance) or None when the page yields nothing confident. Cached per URL
    under a 'verify::' namespace with a 7-day freshness window."""
    url = clean_ws(str(event.get("url", "") or event.get("link", "")))
    if not url:
        return None
    norm = _normalize_link_for_export_dedupe(url) or url
    key = f"verify::{norm}"
    now = datetime.now(tz=EST_TZ)
    cached = url_cache.get(key)
    if isinstance(cached, dict) and cached.get("checked_at_iso"):
        try:
            if (now - datetime.fromisoformat(cached["checked_at_iso"])) < timedelta(days=7):
                return cached.get("result")
        except Exception:
            pass

    result: Optional[dict] = None

    event_id = extract_facebook_event_id(url)
    if event_id and allow_fb_graph and facebook_graph_usable():
        g = fetch_facebook_event_via_graph(event_id)
        if g and g.get("start_dt"):
            start_dt = g["start_dt"]
            end_dt = g.get("end_dt") or (start_dt + timedelta(hours=2))
            result = {
                "title": clean_ws(g.get("title", "")),
                "start_iso": start_dt.isoformat(),
                "end_iso": end_dt.isoformat(),
                "location": clean_ws(g.get("location", "")),
                "address": clean_ws(g.get("address", "")),
                "description": clean_ws(g.get("description", "")),
                "attending_count": g.get("attending_count"),
                "interested_count": g.get("interested_count"),
                "provider": "facebook_graph",
                "confident": True,
            }

    if result is None:
        host = (urlparse(url if "://" in url else f"https://{url}").netloc or "").lower()
        if not any(h in host for h in _DATE_VERIFY_SKIP_HOSTS):
            parsed: List[dict] = []
            try:
                resp = requests.get(url, headers=DEFAULT_HTTP_HEADERS, timeout=20)
                resp.raise_for_status()
                parsed = parse_schema_org_events_from_html(url, resp.text or "")
            except Exception:
                parsed = []
            if parsed:
                best = parsed[0]
                start_dt = best.get("start_dt")
                if start_dt:
                    end_dt = best.get("end_dt") or (start_dt + timedelta(hours=2))
                    # Confident only when the page carried a real clock time.
                    confident = not (start_dt.hour == 0 and start_dt.minute == 0)
                    result = {
                        "title": clean_ws(best.get("title", "")),
                        "start_iso": start_dt.isoformat(),
                        "end_iso": end_dt.isoformat(),
                        "location": clean_ws(best.get("location", "")),
                        "description": clean_ws(best.get("description", "")),
                        "provider": "schema_org",
                        "confident": confident,
                    }

    url_cache[key] = {"checked_at_iso": now.isoformat(), "result": result}
    return result


_POP_BIG_TITLE = {
    "nationals": 30, "national": 20, "championship": 26, "world": 16, "invitational": 20,
    "concours": 26, "goodguys": 30, "nsra": 26, "sema": 30, "mecum": 30,
    "barrett-jackson": 32, "spectacular": 16, "grand national": 30, "hall of fame": 12,
}
_POP_FEST = {"festival": 14, "expo": 16, "jamboree": 14, "showdown": 10, "extravaganza": 14}
_POP_VENUE = {
    "speedway": 20, "raceway": 20, "dragway": 18, "motorsports park": 20, "motor speedway": 22,
    "fairgrounds": 16, "convention center": 18, "expo center": 18, "coliseum": 16, "stadium": 16,
}
_POP_SMALL = {
    "cars and coffee": -6, "cars & coffee": -6, "cruise-in": -6, "cruise in": -6,
    "cruise night": -6, "monthly": -8, "weekly": -8, "meetup": -4, "meet up": -4,
}


def _pop_has(text: str, keyword: str) -> bool:
    return bool(_auto_keyword_pattern(keyword).search(text))


def _fmt_count(value) -> str:
    try:
        n = int(value)
    except Exception:
        return ""
    if n < 0:
        return ""
    if n >= 1000:
        s = f"{n / 1000.0:.1f}".rstrip("0").rstrip(".")
        return f"{s}k"
    return str(n)


def _as_optional_int(value) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def attendance_string(attending: Optional[int], interested: Optional[int]) -> str:
    parts = []
    if isinstance(attending, int) and attending > 0:
        parts.append(f"{_fmt_count(attending)} going")
    if isinstance(interested, int) and interested > 0:
        parts.append(f"{_fmt_count(interested)} interested")
    return " · ".join(parts)


def _fb_reach_score(attending: Optional[int], interested: Optional[int]) -> int:
    a = attending if isinstance(attending, int) else 0
    i = interested if isinstance(interested, int) else 0
    if a <= 0 and i <= 0:
        return 0
    reach = a + 0.5 * i
    for threshold, score in ((2000, 96), (1000, 88), (500, 78), (250, 66), (100, 54), (40, 44), (1, 32)):
        if reach >= threshold:
            return score
    return 0


def compute_popularity_and_size(event: dict) -> Tuple[int, str, str]:
    """Return (popularity_0_100, size_label, attendance_string).

    Popularity blends real Facebook engagement (when known) with a heuristic
    from multi-day span, marquee/venue signals, and recurring/local cues so the
    biggest events float to the top when the column is sorted descending."""
    title = clean_ws(str(event.get("title") or event.get("Event Name") or ""))
    venue = clean_ws(str(
        event.get("address") or event.get("Address") or event.get("location") or event.get("Location") or ""
    ))
    text = _auto_normalize_text(f"{title} {venue}")

    attending = _as_optional_int(event.get("attending_count"))
    interested = _as_optional_int(event.get("interested_count"))
    attendance = attendance_string(attending, interested)

    span = _span_days(str(event.get("start_iso", "")), str(event.get("end_iso", "")))

    score = 32
    if span >= 3:
        score += 34
    elif span == 2:
        score += 20

    score += max([v for k, v in _POP_BIG_TITLE.items() if _pop_has(text, k)] or [0])
    if re.search(r"(?<![a-z0-9])\d{1,3}(?:st|nd|rd|th)(?![a-z0-9])", text) or _pop_has(text, "annual"):
        score += 14
    score += max([v for k, v in _POP_FEST.items() if _pop_has(text, k)] or [0])
    score += max([v for k, v in _POP_VENUE.items() if _pop_has(text, k)] or [0])
    score += sum(v for k, v in _POP_SMALL.items() if _pop_has(text, k))

    heuristic = max(6, min(98, score))
    popularity = max(0, min(100, int(round(max(heuristic, _fb_reach_score(attending, interested))))))

    if popularity >= 75:
        size = "Major"
    elif popularity >= 50:
        size = "Regional"
    elif popularity >= 25:
        size = "Local"
    else:
        size = "Small"
    return popularity, size, attendance


_TITLE_STOPWORDS = {
    "the", "a", "an", "of", "at", "in", "on", "and", "for", "to", "with", "car", "cars",
    "show", "annual", "event", "meet", "2025", "2026", "2027", "presented", "by",
}


def _title_tokens(title: str) -> set:
    tokens = re.findall(r"[a-z0-9]+", _auto_normalize_text(title))
    # Keep single digits ("Day 3" vs "Day 4") — they distinguish series entries.
    return {tok for tok in tokens if tok not in _TITLE_STOPWORDS and (len(tok) >= 2 or tok.isdigit())}


def _row_richness(row: dict) -> tuple:
    """Higher is better — used to pick which near-duplicate to keep."""
    return (
        1 if clean_ws(str(row.get("address", ""))) and _has_full_street_address(str(row.get("address", ""))) else 0,
        1 if _as_optional_int(row.get("attending_count")) or _as_optional_int(row.get("interested_count")) else 0,
        1 if clean_ws(str(row.get("url", "") or row.get("link", ""))) else 0,
        len(clean_ws(str(row.get("location", "")))),
        1 if clean_ws(str(row.get("callout", ""))) else 0,
    )


def merge_same_day_near_duplicates(rows: List[dict]) -> List[dict]:
    """Merge events on the same calendar day in the same city whose titles are
    strongly similar but not identical (e.g. '57th NSRA Street Rod Nationals' vs
    '57th Annual Street Rod Nationals'). Conservative: requires same date, same
    closest-city, Jaccard>=0.6 on distinctive title tokens, and >=2 shared tokens.
    Recurring same-name events on different dates never collide (date is in key)."""
    if not rows:
        return []

    def day_key(row: dict) -> str:
        dt = parse_iso_datetime_safe(str(row.get("start_iso", "")))
        return dt.date().isoformat() if dt else ""

    groups: List[List[int]] = []
    meta = []
    for idx, row in enumerate(rows):
        meta.append({
            "day": day_key(row),
            "city": _auto_normalize_text(str(row.get("closest_city", "")) or str(row.get("city_state", ""))),
            "tokens": _title_tokens(str(row.get("title", ""))),
        })

    used = [False] * len(rows)
    survivors: List[dict] = []
    merged_count = 0
    for i in range(len(rows)):
        if used[i]:
            continue
        cluster = [i]
        used[i] = True
        for j in range(i + 1, len(rows)):
            if used[j]:
                continue
            if not meta[i]["day"] or meta[i]["day"] != meta[j]["day"]:
                continue
            if meta[i]["city"] and meta[j]["city"] and meta[i]["city"] != meta[j]["city"]:
                continue
            a, b = meta[i]["tokens"], meta[j]["tokens"]
            if not a or not b:
                continue
            diff = a ^ b
            # Titles differing only by a number are distinct occurrences of a
            # series, not duplicates: "Day 3" vs "Day 4", "Meet 2" vs "Meet 4".
            if diff and all(tok.isdigit() for tok in diff):
                continue
            shared = a & b
            union = a | b
            if len(shared) >= 2 and (len(shared) / len(union)) >= 0.6:
                cluster.append(j)
                used[j] = True
        if len(cluster) == 1:
            survivors.append(rows[i])
            continue
        merged_count += len(cluster) - 1
        keep = max(cluster, key=lambda k: _row_richness(rows[k]))
        base = dict(rows[keep])
        for k in cluster:
            other = rows[k]
            for field in ("address", "url", "callout", "location", "description"):
                if not clean_ws(str(base.get(field, ""))) and clean_ws(str(other.get(field, ""))):
                    base[field] = other[field]
            for field in ("attending_count", "interested_count"):
                if base.get(field) is None and other.get(field) is not None:
                    base[field] = other[field]
            base_src = clean_ws(str(base.get("source", "")))
            other_src = clean_ws(str(other.get("source", "")))
            if other_src and other_src not in base_src:
                base["source"] = clean_ws(f"{base_src}; {other_src}") if base_src else other_src
        survivors.append(base)

    if merged_count:
        log(f"🧹 Near-duplicate merge: combined {merged_count} same-day similar-title rows")
    return survivors


def merge_same_day_same_address_duplicates(rows: List[dict]) -> List[dict]:
    """Merge events on the same calendar day at the exact same street address,
    even when the titles have nothing in common (the same event submitted or
    scraped under two unrelated names). Requires a full street address
    (number + street) on both sides — a bare city/venue name is too weak a
    signal and would wrongly collapse distinct events that just share a city."""
    if not rows:
        return []

    def day_key(row: dict) -> str:
        dt = parse_iso_datetime_safe(str(row.get("start_iso", "")))
        return dt.date().isoformat() if dt else ""

    def address_key(row: dict) -> str:
        addr = clean_ws(str(row.get("address", "")))
        if not addr or not _has_full_street_address(addr):
            return ""
        return _canonicalize_street_address_for_dedupe(addr)

    keys = [(day_key(r), address_key(r)) for r in rows]
    used = [False] * len(rows)
    survivors: List[dict] = []
    merged_count = 0
    for i in range(len(rows)):
        if used[i]:
            continue
        day_i, addr_i = keys[i]
        if not day_i or not addr_i:
            survivors.append(rows[i])
            used[i] = True
            continue
        cluster = [i]
        used[i] = True
        for j in range(i + 1, len(rows)):
            if used[j]:
                continue
            if keys[j] == (day_i, addr_i):
                cluster.append(j)
                used[j] = True
        if len(cluster) == 1:
            survivors.append(rows[i])
            continue
        merged_count += len(cluster) - 1
        keep = max(cluster, key=lambda k: _row_richness(rows[k]))
        base = dict(rows[keep])
        for k in cluster:
            other = rows[k]
            for field in ("address", "url", "callout", "location", "description"):
                if not clean_ws(str(base.get(field, ""))) and clean_ws(str(other.get(field, ""))):
                    base[field] = other[field]
            for field in ("attending_count", "interested_count"):
                if base.get(field) is None and other.get(field) is not None:
                    base[field] = other[field]
            base_src = clean_ws(str(base.get("source", "")))
            other_src = clean_ws(str(other.get("source", "")))
            if other_src and other_src not in base_src:
                base["source"] = clean_ws(f"{base_src}; {other_src}") if base_src else other_src

        removed_titles = [clean_ws(str(rows[k].get("title", ""))) for k in cluster if k != keep]
        if removed_titles:
            log(f"🧹 Same-address duplicate merge: kept '{clean_ws(str(base.get('title', '')))}' over {removed_titles} (day={day_i}, address={addr_i})")
        survivors.append(base)

    if merged_count:
        log(f"🧹 Same-address duplicate merge: combined {merged_count} same-day same-address row(s) with different titles")
    return survivors


def _address_trust_tier(ev: dict) -> str:
    """Accuracy tier for this event's address data: 'ics' (an organizer's own
    calendar entry — trusted as given), 'facebook' (sourced from, or already
    confirmed in step 1 against, Facebook's own Graph API — ~90% trusted), or
    'other' (must be corroborated by a second signal before a specific street
    address is trusted)."""
    if ev.get("address_verified_via") == "facebook_graph":
        return "facebook"
    source = clean_ws(str(ev.get("source", ""))).lower()
    if "(ics)" in source or "icloud" in source:
        return "ics"
    if "facebook" in source:
        return "facebook"
    return "other"


def _guarantee_closest_city(ev: dict, cfg: dict) -> str:
    """Best-effort Closest City when lat/lon geocoding hasn't produced one:
    fall back to a state found in whatever location text we have, then to the
    configured home city. Closest City should never be blank on export."""
    filled = closest_major_city(ev.get("lat"), ev.get("lon"))
    if filled:
        return filled
    text = " ".join(clean_ws(str(ev.get(k, ""))) for k in ("city_state", "address", "location"))
    m = re.search(rf"\b({US_STATE_ABBR_RE})\b", text, flags=re.IGNORECASE)
    if m:
        state = m.group(1).upper()
        for name, _, _ in MAJOR_CITIES:
            if name.endswith(f", {state}"):
                return name
    home = cfg.get("home", {}) or {}
    return clean_ws(str(home.get("name", ""))) or "Cincinnati, OH"


def enrich_events_for_export(
    events: List[dict],
    geocache: Dict[str, dict],
    url_cache: Dict[str, dict],
    cfg: dict,
    *,
    verify_dates: bool = True,
    lookup_addresses: bool = True,
    fetch_facebook: bool = True,
    revet_automotive: bool = True,
    max_workers: int = 6,
    trusted_sources: Optional[set] = None,
) -> List[dict]:
    """Run the four export-quality passes over the final event list.

    Mutates and returns a list of event dicts (asdict(EventItem) shape)."""
    trusted_sources = trusted_sources or set()

    def _trusted(source: str) -> bool:
        name = clean_ws(str(source))
        return (name in trusted_sources) or bool(source_is_car_dedicated(name, cfg))

    # 1) Verify date/time via the event link (parallel across events; cached).
    if verify_dates:
        from concurrent.futures import ThreadPoolExecutor

        allow_fb = fetch_facebook and facebook_graph_usable()
        targets = [ev for ev in events if clean_ws(str(ev.get("url", "")))]
        results: List[Tuple[dict, Optional[dict]]] = []
        if targets:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                results = list(
                    pool.map(
                        lambda ev: (ev, verify_event_datetime_via_link(ev, url_cache, allow_fb_graph=allow_fb)),
                        targets,
                    )
                )
        verified = corrected = unconfirmed = name_corrected = address_corrected = 0
        for ev, res in results:
            if not res:
                ev["date_verified"] = "unconfirmed"
                unconfirmed += 1
                continue
            new_start = parse_iso_datetime_safe(str(res.get("start_iso", "")))
            if new_start and res.get("confident"):
                old_start = parse_iso_datetime_safe(str(ev.get("start_iso", "")))
                changed = (
                    old_start is None
                    or old_start.date() != new_start.date()
                    or abs((new_start - old_start).total_seconds()) > 3600
                )
                if changed:
                    ev["start_iso"] = new_start.isoformat()
                    new_end = parse_iso_datetime_safe(str(res.get("end_iso", "")))
                    if new_end and new_end > new_start:
                        ev["end_iso"] = new_end.isoformat()
                    ev["date_verified"] = "corrected"
                    corrected += 1
                else:
                    ev["date_verified"] = "verified"
                    verified += 1
            else:
                ev["date_verified"] = ev.get("date_verified") or "unconfirmed"

            # Name double-check: Facebook's own event "name" field for this
            # exact event id is authoritative — prefer it over whatever a
            # scraper guessed if the two have actually diverged.
            if res.get("provider") == "facebook_graph":
                page_title = clean_ws(str(res.get("title", "")))
                if page_title and _normalize_title_for_export_dedupe(page_title) != _normalize_title_for_export_dedupe(str(ev.get("title", ""))):
                    ev["title"] = page_title
                    name_corrected += 1

            # Backfill details discovered on the page.
            if res.get("description") and not clean_ws(str(ev.get("description", ""))):
                ev["description"] = clean_ws(str(res["description"]))[:2000]
            if res.get("location") and not clean_ws(str(ev.get("location", ""))):
                ev["location"] = res["location"]
            # Address double-check: a full street address pulled straight off
            # the event's own page outranks a venue-name geocode guess, so it
            # wins even when we already had (a possibly wrong) address.
            page_address = clean_ws(str(res.get("address", "")))
            if page_address and _has_full_street_address(page_address):
                if _canonicalize_street_address_for_dedupe(page_address) != _canonicalize_street_address_for_dedupe(str(ev.get("address", ""))):
                    if clean_ws(str(ev.get("address", ""))):
                        address_corrected += 1
                    ev["address"] = page_address
                if res.get("provider") == "facebook_graph":
                    ev["address_verified_via"] = "facebook_graph"
            elif page_address and not clean_ws(str(ev.get("address", ""))):
                ev["address"] = page_address
            for count_key in ("attending_count", "interested_count"):
                if res.get(count_key) is not None:
                    ev[count_key] = res[count_key]
        if name_corrected:
            log(f"📝 Event names corrected from linked page: {name_corrected}")
        if address_corrected:
            log(f"🏠 Addresses corrected from linked page: {address_corrected}")
        log(f"🔗 Link date-check: verified={verified} corrected={corrected} unconfirmed={unconfirmed} (of {len(targets)} with links)")

    # 2) Re-vet automotive using any description we just fetched (drop non-car).
    if revet_automotive:
        kept: List[dict] = []
        dropped = 0
        examples: List[str] = []
        for ev in events:
            if _trusted(ev.get("source", "")) or bool(ev.get("bypass_automotive_filter")):
                kept.append(ev)
                continue
            verdict, reason = automotive_content_verdict(
                str(ev.get("title", "")), str(ev.get("location", "")), str(ev.get("description", "")), cfg=cfg
            )
            if verdict == "keep":
                kept.append(ev)
            else:
                dropped += 1
                if len(examples) < 5:
                    examples.append(f"{clean_ws(str(ev.get('title', '')))[:60]} ({reason})")
        events = kept
        if dropped:
            log(f"🧹 Export re-vet removed non-automotive events: {dropped}")
            log(f"   Examples: {examples}")

    # 3) Merge same-day near-duplicates missed by exact dedupe.
    events = merge_same_day_near_duplicates(events)

    # 4) Full-address lookup (serial — Nominatim allows <=1 req/sec) then scoring.
    #
    # Accuracy hierarchy: iCalendar-sourced events are trusted 100% (an
    # organizer's own calendar entry), Facebook-verified events ~90% (either
    # sourced from Facebook or already confirmed against Facebook's own Graph
    # API in step 1) — both skip the extra cross-check below. Everything else
    # must be corroborated by a second, independent signal (a coarse
    # city/ZIP-level geocode) before a specific street address is trusted.
    if lookup_addresses:
        home_lat = cfg["home"]["lat"]
        home_lon = cfg["home"]["lon"]
        local_max = float(cfg["filters"]["local_max_miles"])
        rally_max = float(cfg["filters"]["rally_max_miles"])
        # This is a *sanity* check, not the local/rally inclusion filter: plenty
        # of legitimate "local" events (a track day at Mid-Ohio, 146mi out) sail
        # past local_max already, because the earlier ingestion-time geocode
        # often has no coordinates to filter on. Only reject geocodes that are
        # implausible outright (a same-named venue clear across the country),
        # not ones merely farther than the everyday local radius.
        geocode_sanity_max = max(local_max * 3, 300.0)
        # A second-source disagreement this big means the venue-level geocode
        # almost certainly landed in the wrong city entirely.
        second_source_agreement_miles = 30.0
        filled = 0
        rejected_far = 0
        unverified_other = 0
        closest_city_guaranteed = 0
        logged_rejections: set = set()
        for ev in events:
            current = clean_ws(str(ev.get("address", "")))
            if current and _has_full_street_address(current):
                continue
            location = clean_ws(str(ev.get("location", ""))) or current
            city_state = clean_ws(str(ev.get("city_state", "")))
            formatted, latlon = geocode_full_address(location, city_state, geocache)
            if formatted and latlon:
                tier = _address_trust_tier(ev)

                # A generic venue name ("Liberty Collective") can match a
                # same-named business clear across the country; reject any
                # geocode that lands implausibly far from Cincinnati rather
                # than trust it just because Nominatim returned a hit. Applies
                # to every tier — it's a cheap safety net, not the extra
                # scrutiny "other"-tier sources get below.
                is_rally = ev.get("category") == "rally" or categorize(str(ev.get("title", "")), location, cfg) == "rally"
                allowed_max = rally_max if is_rally else geocode_sanity_max
                if miles_from_home(latlon[0], latlon[1], home_lat, home_lon) > allowed_max:
                    rejection_key = (location.lower(), formatted.lower())
                    if rejection_key not in logged_rejections:
                        logged_rejections.add(rejection_key)
                        log(f"⚠️ Rejected implausible geocode for '{clean_ws(str(ev.get('title', '')))}': '{location}' -> {formatted}")
                    rejected_far += 1
                    formatted, latlon = "", None
                elif tier == "other":
                    # Verify against a second, independent signal: a coarse
                    # geocode of just the city/ZIP, ignoring the ambiguous
                    # venue name that produced `formatted` in the first place.
                    coarse_query = reliable_geocode_query("", city_state)
                    coarse_latlon = geocode(coarse_query, geocache) if coarse_query else None
                    if coarse_latlon:
                        disagreement = haversine_miles(latlon[0], latlon[1], coarse_latlon[0], coarse_latlon[1])
                        if disagreement > second_source_agreement_miles:
                            log(f"⚠️ Unverified geocode (disagrees with city/ZIP by {disagreement:.0f}mi) for '{clean_ws(str(ev.get('title', '')))}': '{location}' -> {formatted}")
                            unverified_other += 1
                            formatted, latlon = "", None
                    # No second signal available (no usable city_state) —
                    # fall back to the distance-sanity result already applied
                    # above rather than blocking every venue-only "other" row.
            if formatted:
                ev["address"] = formatted
                filled += 1
                if latlon and (ev.get("lat") is None or ev.get("lon") is None):
                    ev["lat"], ev["lon"] = latlon

            # Guarantee Closest City now, using any lat/lon just resolved
            # above (falls back to a state found in the text, then home).
            if not clean_ws(str(ev.get("closest_city", ""))):
                filled_city = _guarantee_closest_city(ev, cfg)
                if filled_city:
                    ev["closest_city"] = filled_city
                    closest_city_guaranteed += 1

            if not formatted and not current:
                # Still couldn't confirm a street-level address: every event
                # gets a *real*, city-qualified address rather than a bare,
                # ungeocoded venue label.
                qualifier = city_state or clean_ws(str(ev.get("closest_city", "")))
                if qualifier and qualifier.lower() not in location.lower():
                    ev["address"] = f"{location}, {qualifier}" if location else qualifier
                else:
                    ev["address"] = location
        if filled:
            log(f"🏠 Full address looked up for {filled} events")
        if rejected_far:
            log(f"🧹 Rejected {rejected_far} geocoded address(es) that landed implausibly far from Cincinnati")
        if unverified_other:
            log(f"🧹 Rejected {unverified_other} unverified geocoded address(es) from non-ICS/Facebook sources")
        if closest_city_guaranteed:
            log(f"🗺️ Closest City guaranteed (fallback) for {closest_city_guaranteed} events")

    # Events whose address was already a full street address skipped the loop
    # above entirely (via `continue`) — make sure they still get Closest City.
    for ev in events:
        if not clean_ws(str(ev.get("closest_city", ""))):
            filled_city = _guarantee_closest_city(ev, cfg)
            if filled_city:
                ev["closest_city"] = filled_city

    # 5) Now that addresses are resolved, catch duplicates that share the exact
    #    same day + street address even when their titles are unrelated.
    events = merge_same_day_same_address_duplicates(events)

    # 6) Make Location and Address agree: once Address is a confirmed full
    #    street address, drop any stale street/ZIP text embedded in Location
    #    (a source page's own text can drift from what the venue geocodes to,
    #    e.g. a ZIP off by one) so the two columns never contradict each other.
    reconciled = 0
    for ev in events:
        address = clean_ws(str(ev.get("address", "")))
        if not address or not _has_full_street_address(address):
            continue
        location = clean_ws(str(ev.get("location", "")))
        new_location = _reconcile_location_with_address(location, address)
        if new_location != location:
            ev["location"] = new_location
            reconciled += 1
    if reconciled:
        log(f"🧹 Reconciled Location text with confirmed Address for {reconciled} events")

    for ev in events:
        popularity, size, attendance = compute_popularity_and_size(ev)
        ev["popularity"] = popularity
        ev["size"] = size
        ev["attendance"] = attendance

    return events


def write_csv(events: List[dict], path: str) -> None:
    fields = [
        "Event Name",
        "Date",
        "Start Time",
        "End Time",
        "Location",
        "Address",
        "Closest City",
        "Callout",
        "Size",
        "Popularity",
        "Attendance",
        "Source",
        "Event URL",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for ev in events:
            w.writerow(ev)


def normalize_export_schema(rows: List[dict], headers: Optional[List[str]] = None) -> Tuple[List[dict], List[str]]:
    required_headers = [
        "Event Name",
        "Date",
        "Start Time",
        "End Time",
        "Location",
        "Address",
        "Closest City",
        "Callout",
        "Size",
        "Popularity",
        "Attendance",
        "Source",
        "Event URL",
    ]
    alias_map = {
        "title": ["title", "event_title", "name", "Event Name", "event name"],
        "date": ["date", "event_date", "Date"],
        "start_time": ["start_time", "time", "starttime", "Start Time", "start time"],
        "end_time": ["end_time", "endtime", "End Time", "end time"],
        "categ": ["categ", "category", "type"],
        "miles_from_c": ["miles_from_c", "miles_from_cincy", "miles", "distance", "mileage"],
        "location": ["location", "venue", "Location"],
        "address": ["address", "Address", "full_address"],
        "link": ["link", "url", "event_url", "eventlink", "Event URL", "event url"],
        "source": ["source", "event_source", "source_name", "pulled_from", "Source"],
        "start_dt": ["start_iso", "iso", "start", "start_datetime", "startdatetime", "Start ISO", "start iso"],
        "end_dt": ["end_iso", "end", "end_datetime", "enddatetime", "End ISO", "end iso"],
        "closest_city": ["Closest City", "closest_city", "closest city"],
        "callout": ["Callout", "callout", "call_out", "call out"],
        "size": ["size", "Size"],
        "popularity": ["popularity", "Popularity", "popularity_score"],
        "attendance": ["attendance", "Attendance"],
        "attending_count": ["attending_count"],
        "interested_count": ["interested_count"],
        "lat": ["lat", "latitude"],
        "lon": ["lon", "lng", "longitude"],
    }

    if headers is None:
        seen_headers = []
        seen_set = set()
        for row in rows:
            for key in row.keys():
                if key not in seen_set:
                    seen_set.add(key)
                    seen_headers.append(key)
        headers = seen_headers

    def pick_value(row: dict, aliases: List[str]) -> str:
        for alias in aliases:
            if alias in row:
                return clean_ws(str(row.get(alias, "")))
        return ""

    def parse_datetime(raw: str) -> Optional[datetime]:
        text = clean_ws(raw)
        if not text:
            return None
        try:
            return dateutil_parser.isoparse(text)
        except Exception:
            try:
                return datetime.fromisoformat(text)
            except Exception:
                return None

    def format_time(dt_value: datetime) -> str:
        return dt_value.strftime("%I:%M %p").lstrip("0")

    def normalize_source(raw_source: str) -> str:
        source = clean_ws(raw_source)
        if source.lower().startswith("screenshot:"):
            return source
        if "sheet" in source.lower():
            return "sheet"
        if source:
            return source
        return "web"

    normalized_rows: List[dict] = []
    missing_datetime_rows = []
    for row_idx, row in enumerate(rows, start=1):
        normalized = {col: "" for col in required_headers}
        normalized["Event Name"] = pick_value(row, alias_map["title"])
        raw_location = pick_value(row, alias_map["location"])
        explicit_address = pick_value(row, alias_map["address"])
        normalized["Location"] = raw_location or explicit_address
        # Address is the full street address when we have one, else the venue text.
        normalized["Address"] = explicit_address or raw_location
        normalized["Event URL"] = pick_value(row, alias_map["link"])
        normalized["Source"] = normalize_source(pick_value(row, alias_map["source"]) or "")

        closest_city = pick_value(row, alias_map["closest_city"])
        if not closest_city:
            def _coord(aliases: List[str]) -> Optional[float]:
                raw_coord = pick_value(row, aliases)
                try:
                    return float(raw_coord) if raw_coord else None
                except Exception:
                    return None

            closest_city = closest_major_city(_coord(alias_map["lat"]), _coord(alias_map["lon"]))
        normalized["Closest City"] = closest_city

        normalized["Callout"] = (
            pick_value(row, alias_map["callout"])
            or detect_registration_callout(normalized["Event Name"], raw_location)
        )

        # Size / Popularity / Attendance: use enriched values when present,
        # otherwise compute a heuristic so every row (incl. manual sheet rows)
        # is scored and sortable.
        existing_size = pick_value(row, alias_map["size"])
        existing_pop = pick_value(row, alias_map["popularity"])
        existing_att = pick_value(row, alias_map["attendance"])
        if existing_size and existing_pop:
            normalized["Size"] = existing_size
            normalized["Popularity"] = existing_pop
            normalized["Attendance"] = existing_att
        else:
            calc_pop, calc_size, calc_att = compute_popularity_and_size({
                "title": normalized["Event Name"],
                "location": normalized["Location"],
                "address": normalized["Address"],
                "start_iso": pick_value(row, alias_map["start_dt"]),
                "end_iso": pick_value(row, alias_map["end_dt"]),
                "attending_count": pick_value(row, alias_map["attending_count"]),
                "interested_count": pick_value(row, alias_map["interested_count"]),
            })
            normalized["Size"] = existing_size or calc_size
            normalized["Popularity"] = existing_pop or str(calc_pop)
            normalized["Attendance"] = existing_att or calc_att

        date_text = pick_value(row, alias_map["date"])
        start_time_text = pick_value(row, alias_map["start_time"])
        end_time_text = pick_value(row, alias_map["end_time"])

        date_ok = False
        start_ok = False
        end_ok = False

        if date_text and start_time_text:
            try:
                parsed_date = dateutil_parser.parse(date_text)
                parsed_start = dateutil_parser.parse(start_time_text)
                normalized["Date"] = parsed_date.strftime("%Y-%m-%d")
                normalized["Start Time"] = format_time(parsed_start)
                date_ok = True
                start_ok = True
            except Exception:
                pass

            if end_time_text:
                try:
                    parsed_end = dateutil_parser.parse(end_time_text)
                    normalized["End Time"] = format_time(parsed_end)
                    end_ok = True
                except Exception:
                    normalized["End Time"] = ""
        
        if not (date_ok and start_ok):
            start_dt = parse_datetime(pick_value(row, alias_map["start_dt"]))
            if start_dt:
                normalized["Date"] = start_dt.strftime("%Y-%m-%d")
                normalized["Start Time"] = format_time(start_dt)
                date_ok = True
                start_ok = True
            else:
                normalized["Date"] = ""
                normalized["Start Time"] = ""

        if not end_ok:
            end_dt = parse_datetime(pick_value(row, alias_map["end_dt"]))
            if end_dt:
                normalized["End Time"] = format_time(end_dt)
                end_ok = True
            else:
                normalized["End Time"] = ""

        if not (date_ok and start_ok and end_ok):
            missing_datetime_rows.append(row_idx)

        normalized_rows.append(normalized)

    dropped_columns = [h for h in headers if h not in required_headers]
    log(f"🧹 Export schema cleanup dropped columns: {dropped_columns}")
    log(
        "🧾 Rows with missing normalized date/time fields: "
        f"{len(missing_datetime_rows)}"
    )
    if missing_datetime_rows:
        log(f"   Row numbers with missing values: {missing_datetime_rows}")

    final_headers = list(normalized_rows[0].keys()) if normalized_rows else required_headers
    assert final_headers == required_headers, (
        f"Invalid export schema order: expected {required_headers} got {final_headers}"
    )
    forbidden_combined_headers = {
        "iso", "start_iso", "end_iso", "start", "end", "start_datetime", "end_datetime", "datetime"
    }
    assert not any((h in forbidden_combined_headers) or ("datetime" in h.lower()) or h.lower().endswith("_iso") for h in final_headers), (
        f"Invalid combined datetime headers in export: {final_headers}"
    )
    return normalized_rows, required_headers


# -------------------------
# Google Sheets export
# -------------------------
def get_google_credentials() -> Optional[service_account.Credentials]:
    service_account_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")
    service_account_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON") or os.getenv("GDRIVE_SERVICE_ACCOUNT_JSON")
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


def _execute_sheets_request(request, *, context: str, max_attempts: int = 5):
    """Execute a googleapiclient Sheets request, retrying on transient (429/5xx) errors.

    Google's Sheets API returns occasional 503s ("service currently unavailable")
    that clear up within seconds; without a retry, one of these kills the whole
    collector run after it already did all the work of gathering events."""
    for attempt in range(max_attempts):
        try:
            return request.execute()
        except HttpError as ex:
            status = getattr(ex, "status_code", None) or getattr(getattr(ex, "resp", None), "status", None)
            try:
                status = int(status)
            except (TypeError, ValueError):
                status = None
            if status in (429, 500, 502, 503, 504) and attempt < max_attempts - 1:
                delay = (2 ** attempt) * 1.5
                log(f"⚠️ Google Sheets API {status} on {context}, retrying in {delay:.1f}s (attempt {attempt + 1}/{max_attempts})")
                time.sleep(delay)
                continue
            raise


def ensure_sheet_tab(sheets, spreadsheet_id: str, title: str) -> None:
    sheet_info = _execute_sheets_request(
        sheets.spreadsheets().get(spreadsheetId=spreadsheet_id, includeGridData=False),
        context="spreadsheets.get",
    )
    titles = {sheet["properties"]["title"] for sheet in sheet_info.get("sheets", [])}
    if title in titles:
        return
    requests_body = {"requests": [{"addSheet": {"properties": {"title": title}}}]}
    _execute_sheets_request(
        sheets.spreadsheets().batchUpdate(spreadsheetId=spreadsheet_id, body=requests_body),
        context="spreadsheets.batchUpdate(addSheet)",
    )


def _extract_address_components(text: str, city_state: str = "") -> Tuple[str, str, str]:
    """Best-effort (street, city, state) from a venue/address string.

    Prefers a US street-address pattern ("NNN Street, City, ST") found inside the
    text; falls back to the city_state hint for city/state when the free text
    lacks them."""
    raw = clean_ws(text)
    street = ""
    city = ""
    state = ""

    hint = re.match(rf"^\s*(.+?),\s*({US_STATE_ABBR_RE})\s*$", clean_ws(city_state), flags=re.IGNORECASE)
    if hint:
        city = clean_ws(hint.group(1))
        state = hint.group(2).upper()

    full = re.search(
        rf"(\d{{1,6}}\s+[^,]+?),\s*([^,]+?),\s*({US_STATE_ABBR_RE})\b",
        raw,
        flags=re.IGNORECASE,
    )
    if full:
        street = clean_ws(full.group(1))
        city = clean_ws(full.group(2)) or city
        state = full.group(3).upper()
        return street, city, state

    street_only = re.search(
        r"(\d{1,6}\s+[A-Za-z0-9.\-' ]+?"
        r"(?:street|st|avenue|ave|road|rd|drive|dr|boulevard|blvd|lane|ln|way|pike|"
        r"highway|hwy|court|ct|parkway|pkwy|circle|cir|trail|terrace|place|pl))\b",
        raw,
        flags=re.IGNORECASE,
    )
    if street_only:
        street = clean_ws(street_only.group(1))
    return street, city, state


def verify_usps_address(location_text: str, city_state_text: str) -> Dict[str, str]:
    """Verify/normalize a street address via the USPS Web Tools API.

    Optional and off by default: without a ``USPS_USER_ID`` env var this returns
    status ``unverified_missing_usps_user_id`` and makes no network call, so the
    pipeline is unaffected. When configured, it returns USPS-canonical
    street/city/state/ZIP+4 fields for the export."""
    street_in = clean_ws(location_text)
    if not street_in:
        return {
            "status": "skipped_missing_location",
            "street": "", "city": "", "state": "", "zip5": "", "zip4": "",
            "formatted": "", "error": "",
        }

    street, city, state = _extract_address_components(location_text, city_state_text)
    user_id = clean_ws(os.getenv("USPS_USER_ID", ""))
    if not user_id:
        return {
            "status": "unverified_missing_usps_user_id",
            "street": street or street_in, "city": city, "state": state,
            "zip5": "", "zip4": "",
            "formatted": ", ".join(p for p in (street or street_in, clean_ws(city_state_text)) if p),
            "error": "Missing USPS_USER_ID",
        }

    xml = (
        f'<AddressValidateRequest USERID="{html.escape(user_id, quote=True)}">'
        f"<Address ID='0'><Address1></Address1>"
        f"<Address2>{html.escape(street or street_in)}</Address2>"
        f"<City>{html.escape(city)}</City><State>{html.escape(state)}</State>"
        f"<Zip5></Zip5><Zip4></Zip4></Address></AddressValidateRequest>"
    )
    try:
        resp = requests.get(
            "https://secure.shippingapis.com/ShippingAPI.dll",
            params={"API": "Verify", "XML": xml},
            timeout=20,
        )
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
    except Exception as ex:
        return {
            "status": "error", "street": street or street_in, "city": city, "state": state,
            "zip5": "", "zip4": "", "formatted": "", "error": clean_ws(str(ex))[:200],
        }

    err = root.find(".//Error")
    if err is not None:
        return {
            "status": "not_found", "street": street or street_in, "city": city, "state": state,
            "zip5": "", "zip4": "",
            "formatted": "", "error": clean_ws((err.findtext("Description") or "USPS returned an error")),
        }

    def _txt(tag: str) -> str:
        node = root.find(f".//{tag}")
        return clean_ws(node.text) if node is not None and node.text else ""

    out_street = _txt("Address2")
    out_city = _txt("City")
    out_state = _txt("State")
    zip5 = _txt("Zip5")
    zip4 = _txt("Zip4")
    tail = f"{out_state} {zip5}".strip()
    if zip4:
        tail = f"{tail}-{zip4}"
    formatted = ", ".join(p for p in (out_street, out_city, tail) if clean_ws(p))
    return {
        "status": "verified",
        "street": out_street, "city": out_city, "state": out_state,
        "zip5": zip5, "zip4": zip4, "formatted": formatted, "error": "",
    }

MANUAL_SOURCE_TOKENS = {"manual", "manually_added", "user", "user_added"}


def _parse_sheet_row_date_et(row: dict) -> Optional[datetime]:
    date_text = clean_ws(str(row.get("Date", "")))
    if not date_text:
        return None
    start_time = clean_ws(str(row.get("Start Time", "")))
    candidate = f"{date_text} {start_time}".strip()
    try:
        dt = dateutil_parser.parse(candidate)
    except Exception:
        try:
            dt = dateutil_parser.parse(date_text)
        except Exception:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz.gettz("America/New_York"))
    return dt


def _is_manual_event_row(row: dict) -> bool:
    source_raw = clean_ws(str(row.get("Source", "")))
    source = source_raw.lower()
    if source in MANUAL_SOURCE_TOKENS:
        return True
    if source.startswith("screenshot:"):
        return False
    if source in {"web", "sheet"}:
        return False
    return (not source) and bool(clean_ws(str(row.get("Event Name", "")))) and bool(clean_ws(str(row.get("Date", ""))))


def _read_future_manual_events_from_sheet(sheets, spreadsheet_id: str) -> List[dict]:
    now_et = datetime.now(tz=tz.gettz("America/New_York"))
    today_et = now_et.date()
    try:
        resp = _execute_sheets_request(
            sheets.spreadsheets().values().get(
                spreadsheetId=spreadsheet_id,
                range="Events!A1:M4000",
            ),
            context="values.get(Events!A1:M4000)",
        )
    except Exception as ex:
        log(f"⚠️ Unable to read existing Events tab for manual-row preservation: {ex}")
        return []

    rows = resp.get("values", []) or []
    if not rows:
        return []

    header = [clean_ws(c) for c in rows[0]]
    wanted = ["Event Name", "Date", "Start Time", "End Time", "Location", "Address", "Closest City", "Callout", "Source", "Event URL"]
    optional_aliases = {
        "Event URL": ["Event URL", "URL", "Url", "Link", "Event Link", "event_url", "event url"],
    }
    idx = {name: i for i, name in enumerate(header) if name}
    if not all(col in idx for col in ("Event Name", "Date")):
        return []
    for canonical, aliases in optional_aliases.items():
        if canonical in idx:
            continue
        for alias in aliases:
            if alias in idx:
                idx[canonical] = idx[alias]
                break

    preserved: List[dict] = []
    skipped_past = 0
    for raw in rows[1:]:
        row = {
            col: clean_ws(raw[idx[col]] if (col in idx and len(raw) > idx[col]) else "")
            for col in wanted
        }
        if not _is_manual_event_row(row):
            continue
        parsed_dt = _parse_sheet_row_date_et(row)
        if parsed_dt is not None and parsed_dt.date() < today_et:
            skipped_past += 1
            continue
        if not row.get("Source"):
            row["Source"] = "manual"
        preserved.append(row)

    log(f"ℹ️ Manual sheet rows preserved: kept={len(preserved)} skipped_past={skipped_past}")
    return preserved


def update_apex_spreadsheet(events: List[dict]) -> None:
    dry_run = clean_ws(os.getenv("COLLECTOR_DRY_RUN", "")).lower() in ("1", "true", "yes", "y")
    if dry_run:
        log("ℹ️ COLLECTOR_DRY_RUN enabled: skipping Google Sheets write.")
        return

    creds = get_google_credentials()
    if not creds:
        log("⚠️ Skipping Google Sheets update: missing GOOGLE_SERVICE_ACCOUNT_FILE or GOOGLE_SERVICE_ACCOUNT_JSON/GDRIVE_SERVICE_ACCOUNT_JSON.")
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

    normalized_rows, _ = normalize_export_schema(events)
    manual_rows = _read_future_manual_events_from_sheet(sheets, spreadsheet_id)
    combined_rows = dedupe_export_rows(normalized_rows + manual_rows)
    normalized_rows, headers = normalize_export_schema(combined_rows)
    values = [headers]
    for ev in normalized_rows:
        values.append([ev.get(h, "") for h in headers])

    # 1) Clear the sheet range first (prevents leftovers if list shrinks)
    clear_range = "Events!A1:M"
    _execute_sheets_request(
        sheets.spreadsheets().values().clear(
            spreadsheetId=spreadsheet_id,
            range=clear_range,
            body={}
        ),
        context="values.clear(Events!A1:M)",
    )

    # 2) Write data
    _execute_sheets_request(
        sheets.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range="Events!A1",
            valueInputOption="RAW",
            body={"values": values},
        ),
        context="values.update(Events!A1)",
    )

    # 3) Write a visible update stamp (column O is outside the A:M table)
    stamp = f"Updated by bot: {now_et_iso()} | rows={len(values)-1}"
    _execute_sheets_request(
        sheets.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range="Events!O1",
            valueInputOption="RAW",
            body={"values": [[stamp]]},
        ),
        context="values.update(Events!O1)",
    )

    # 4) Read back first few rows to prove it wrote
    preview = _execute_sheets_request(
        sheets.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range="Events!A1:C5",
        ),
        context="values.get(Events!A1:C5 preview)",
    )
    got = preview.get("values", [])
    log(f"🧽 Cleared Events sheet range {clear_range} then wrote {len(values)-1} rows")
    log(f"   Wrote {len(values)-1} events to Events!A1")
    log(f"   Preview A1:C5 = {got}")
    log(f"   Stamp written to Events!O1 = {stamp}")

    # 5) Open the sheet back up and double-check what actually landed: flag any
    # rows missing a start time or address, and catch duplicate rows (same date
    # + same street address, even under different event names) that slipped
    # past pre-write dedup. If any turn up, rewrite the sheet without them.
    verify_written_sheet_output(sheets, spreadsheet_id, clear_range)


def verify_written_sheet_output(sheets, spreadsheet_id: str, data_range: str) -> None:
    """Read back everything just written to the Events tab and sanity-check it."""
    resp = _execute_sheets_request(
        sheets.spreadsheets().values().get(spreadsheetId=spreadsheet_id, range=data_range),
        context="values.get(Events read-back for verification)",
    )
    all_values = resp.get("values", []) or []
    if not all_values:
        log("⚠️ Post-write verification: Events tab came back empty.")
        return

    sheet_headers = all_values[0]
    idx = {name: i for i, name in enumerate(sheet_headers)}
    data_rows = all_values[1:]

    def cell(row: List[str], name: str) -> str:
        i = idx.get(name)
        return clean_ws(row[i]) if i is not None and i < len(row) else ""

    missing_start_time = sum(1 for row in data_rows if cell(row, "Date") and not cell(row, "Start Time"))
    missing_address = sum(1 for row in data_rows if not cell(row, "Address"))

    seen: Dict[Tuple[str, str], int] = {}
    dup_indices: set = set()
    for i, row in enumerate(data_rows):
        date = cell(row, "Date")
        addr = cell(row, "Address")
        if not date or not addr or not _has_full_street_address(addr):
            continue
        key = (date, _canonicalize_street_address_for_dedupe(addr))
        if key in seen:
            dup_indices.add(i)
        else:
            seen[key] = i

    log(
        "🔍 Post-write sheet verification: "
        f"rows={len(data_rows)} missing_start_time={missing_start_time} "
        f"missing_address={missing_address} duplicate_rows_found={len(dup_indices)}"
    )

    if not dup_indices:
        return

    dup_sample = [
        f"{cell(data_rows[i], 'Event Name')} | {cell(data_rows[i], 'Date')} | {cell(data_rows[i], 'Address')}"
        for i in list(dup_indices)[:5]
    ]
    log(f"🧹 Post-write duplicates removed: {dup_sample}")

    cleaned_rows = [row for i, row in enumerate(data_rows) if i not in dup_indices]
    _execute_sheets_request(
        sheets.spreadsheets().values().clear(spreadsheetId=spreadsheet_id, range=data_range, body={}),
        context="values.clear(Events post-verify rewrite)",
    )
    _execute_sheets_request(
        sheets.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range="Events!A1",
            valueInputOption="RAW",
            body={"values": [sheet_headers] + cleaned_rows},
        ),
        context="values.update(Events post-verify rewrite)",
    )
    log(f"🧹 Post-write verification removed {len(dup_indices)} duplicate row(s) still present after write; rewrote sheet.")



def run_prune_self_test() -> None:
    now_et = datetime.now(tz=tz.gettz("America/New_York"))
    dummy_events = [
        {
            "title": "Past Event",
            "start_iso": (now_et - timedelta(days=2)).isoformat(),
            "end_iso": (now_et - timedelta(days=1)).isoformat(),
        },
        {
            "title": "Ongoing Event",
            "start_iso": (now_et - timedelta(hours=1)).isoformat(),
            "end_iso": (now_et + timedelta(hours=1)).isoformat(),
        },
        {
            "title": "Future Event",
            "start_iso": (now_et + timedelta(days=1)).isoformat(),
            "end_iso": (now_et + timedelta(days=1, hours=2)).isoformat(),
        },
    ]

    old_mode = os.getenv("EVENT_PRUNE_MODE")
    os.environ["EVENT_PRUNE_MODE"] = "end_before_now"
    try:
        kept = prune_past_events(dummy_events, now_et)
    finally:
        if old_mode is None:
            os.environ.pop("EVENT_PRUNE_MODE", None)
        else:
            os.environ["EVENT_PRUNE_MODE"] = old_mode

    kept_titles = [e.get("title") for e in kept]
    assert kept_titles == ["Ongoing Event", "Future Event"], f"Unexpected kept titles: {kept_titles}"
    print("✅ prune self-test passed", flush=True)

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
    screenshots_dir = clean_ws(os.getenv("SCREENSHOTS_DIR") or "/screenshots")
    if os.path.isdir(screenshots_dir):
        sources = list(sources) + [{"type": "local_screenshot_folder", "name": "Screenshot Folder OCR", "path": screenshots_dir}]
        log(f"   Screenshot source enabled from {screenshots_dir}")
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
    existing = []
    skipped_existing_rows = 0
    for e in existing_raw.get("events", []):
        ev = event_item_from_stored_row(e)
        if ev is None:
            skipped_existing_rows += 1
            continue
        existing.append(ev)
    if skipped_existing_rows:
        log(f"⚠️ Skipped unreadable legacy existing events: {skipped_existing_rows}")
    for ev in existing:
        ev.location = normalize_location_for_output(ev.location, ev.city_state)

    source_bypass_automotive = {
        clean_ws(str(src.get("name", "")))
        for src in (sources or [])
        if clean_ws(str(src.get("name", "")))
        and clean_ws(str(src.get("bypass_automotive_filter", ""))).lower() in {"1", "true", "yes", "y"}
    }

    def _source_trusted_automotive(source_name: str) -> bool:
        """Bypass-flagged sources plus car-dedicated sources are trusted as
        automotive even when a specific title lacks car keywords."""
        name = clean_ws(str(source_name))
        if name in source_bypass_automotive:
            return True
        return bool(source_is_car_dedicated(name, cfg))

    existing_before_focus_filter = len(existing)
    existing = [
        e
        for e in existing
        if _source_trusted_automotive(e.source) or is_automotive_event_safe(e.title, e.location, cfg)
    ]
    dropped_existing_non_automotive = existing_before_focus_filter - len(existing)
    if dropped_existing_non_automotive:
        log(f"🧹 Removed non-automotive legacy events from existing set: {dropped_existing_non_automotive}")

    raw_events: List[dict] = []
    source_run_stats: List[dict] = []
    source_failures: List[dict] = []
    pipeline_metrics: Dict[str, dict] = {}
    source_filter_stats: Dict[str, Counter] = {}

    serpapi_source_types = {"web_search_serpapi", "web_search_facebook_events_serpapi", "facebook_group_events_serpapi", "serpapi_google_events"}
    enable_fb_site_discovery = clean_ws(os.getenv("ENABLE_FACEBOOK_SERPAPI_SITE_DISCOVERY", "0")).lower() not in ("0", "false", "no")

    for s in sources:
        stype = s.get("type")
        sname = s.get("name", "(unnamed source)")

        if stype in serpapi_source_types and not serpapi_enabled:
            log(f"⚠️ Skipping SerpAPI source (missing key): {sname} [{stype}]")
            source_run_stats.append({"name": sname, "type": stype, "status": "skipped", "collected": 0})
            continue

        if stype in {"web_search_facebook_events_serpapi", "facebook_group_events_serpapi"} and not enable_fb_site_discovery:
            log(f"ℹ️ Skipping Facebook site-search SerpAPI source (disabled): {sname} [{stype}]")
            source_run_stats.append({"name": sname, "type": stype, "status": "skipped", "collected": 0, "reason": "disabled_fb_site_discovery"})
            continue

        source_with_context = dict(s)
        source_with_context["_facebook_targets"] = facebook_targets

        before_count = len(raw_events)
        try:
            diagnostics: dict = {}
            if stype == "html_carsandcoffeeevents_ohio":
                raw_events.extend(collect_carsandcoffeeevents_ohio(source_with_context))
            elif stype == "tribe_events_api":
                raw_events.extend(collect_tribe_events_api(source_with_context, diagnostics=diagnostics))
            elif stype == "html_wordpress_events_list":
                raw_events.extend(collect_wordpress_events_series(source_with_context))
            elif stype == "ics":
                raw_events.extend(collect_ics(source_with_context, diagnostics=diagnostics))
            elif stype == "google_sheet_events_import":
                raw_events.extend(collect_google_sheet_events_import(source_with_context, diagnostics=diagnostics))
            elif stype == "local_screenshot_folder":
                raw_events.extend(collect_screenshot_folder_events(source_with_context, diagnostics=diagnostics))
            elif stype == "facebook_page_events":
                raw_events.extend(collect_facebook_page_events(source_with_context, diagnostics=diagnostics))
            elif stype == "facebook_events_scroll":
                raw_events.extend(collect_facebook_events_scroll(source_with_context, url_cache, diagnostics=diagnostics))
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
            reason_tag = ""
            if collected == 0:
                reason_tag = diag.get("reason", "no_results_from_search")
                stat["reason"] = reason_tag
            source_run_stats.append(stat)
            extra_log = ""
            if stype == "ics" and clean_ws(str(source_with_context.get("future_only", ""))).lower() in {"1", "true", "yes", "y"}:
                extra_log = " future_only_applied=yes"
            if stype == "google_sheet_events_import":
                extra_log = " future_only_applied=yes"
            if collected == 0:
                log(f"🔎 Source complete: {sname} [{stype}] -> {collected} events (reasons_if_0={reason_tag}){extra_log}")
            else:
                log(f"🔎 Source complete: {sname} [{stype}] -> {collected} events{extra_log}")
        except SourceDisabledError as ex:
            source_run_stats.append({"name": sname, "type": stype, "status": "disabled", "collected": 0, "reason": ex.reason, "error": str(ex)})
            log(f"⚪ Source disabled: {sname} [{stype}] :: {ex}")
        except Exception as ex:
            source_run_stats.append({"name": sname, "type": stype, "status": "failed", "collected": 0, "error": str(ex)})
            source_failures.append({"name": sname, "type": stype, "error": str(ex), "traceback": log_exception_context(f"Source failed: {sname} [{stype}]", ex)})

    if serpapi_enabled and not enable_fb_site_discovery:
        log("ℹ️ Facebook SerpAPI site:facebook.com/events discovery is disabled (ENABLE_FACEBOOK_SERPAPI_SITE_DISCOVERY=0).")

    enable_fb_discovery = clean_ws(os.getenv("ENABLE_FACEBOOK_SERP_DISCOVERY", "0")).lower() not in ("0", "false", "no")
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
    merged = [
        ev
        for ev in merged
        if _source_trusted_automotive(ev.source) or is_automotive_event_safe(ev.title, ev.location, cfg)
    ]
    dropped_merged_non_automotive = merged_before_focus_filter - len(merged)
    if dropped_merged_non_automotive:
        log(f"🧹 Removed non-automotive events after merge: {dropped_merged_non_automotive}")
    log(f"✅ Merged total events before prune: {len(merged)}")

    now_et = datetime.now(tz=tz.gettz("America/New_York"))
    merged_dicts = [asdict(e) for e in merged]
    merged_dicts = prune_past_events(merged_dicts, now_et)
    # Final geo pass over ALL merged events (including ones reloaded from
    # events.json, which intake filtering never re-checks):
    #   1. resolve coordinates, preferring the ZIP code — city names alone are
    #      ambiguous (e.g. "Fairfield, OH" resolves to the wrong Fairfield)
    #   2. enforce the local/rally distance caps
    #   3. backfill the Closest City column
    local_max = float(cfg["filters"]["local_max_miles"])
    rally_max = float(cfg["filters"]["rally_max_miles"])
    home_lat, home_lon = float(cfg["home"]["lat"]), float(cfg["home"]["lon"])
    distance_exempt_sources = {
        clean_ws(str(src.get("name", "")))
        for src in (sources or [])
        if clean_ws(str(src.get("name", "")))
        and clean_ws(str(src.get("bypass_distance_filter", ""))).lower() in {"1", "true", "yes", "y"}
    }
    closest_city_backfilled = 0
    dropped_far = 0
    kept_dicts: List[dict] = []
    for ev in merged_dicts:
        ev["location"] = normalize_location_for_output(ev.get("location", ""), ev.get("city_state", ""))

        lat, lon = ev.get("lat"), ev.get("lon")
        # Only geocode text reliable enough to act on (ZIP or city+state) —
        # bare venue names resolve to random places worldwide.
        geo_query = reliable_geocode_query(str(ev.get("location") or ""), str(ev.get("city_state") or ""))
        if geo_query and geo_query.endswith(", USA"):
            latlon = geocode(geo_query, geocache)
            if latlon:
                lat, lon = latlon
        elif (lat is None or lon is None) and geo_query:
            latlon = geocode(geo_query, geocache)
            if latlon:
                lat, lon = latlon

        ev_source = clean_ws(str(ev.get("source", "") or ""))
        distance_exempt = any(name in ev_source for name in distance_exempt_sources)
        if lat is not None and lon is not None and not distance_exempt:
            miles = miles_from_home(lat, lon, home_lat, home_lon)
            # Recompute from title/location (stored rows default to "local").
            cat = categorize(str(ev.get("title", "") or ""), str(ev.get("location", "") or ""), cfg)
            limit = rally_max if cat == "rally" else local_max
            if miles > limit:
                dropped_far += 1
                continue

        if not clean_ws(str(ev.get("closest_city", "") or "")):
            filled = closest_major_city(lat, lon)
            if filled:
                ev["closest_city"] = filled
                closest_city_backfilled += 1
        kept_dicts.append(ev)
    merged_dicts = kept_dicts
    if dropped_far:
        log(f"🧹 Dropped {dropped_far} merged events beyond distance caps (local={local_max}mi rally={rally_max}mi)")
    if closest_city_backfilled:
        log(f"🗺️ Closest City backfilled for {closest_city_backfilled} events")

    merged_dicts = dedupe_export_rows(merged_dicts)

    # Export-quality enrichment: verify each event's date/time against its link,
    # look up a full street address, pull Facebook attendance, score popularity,
    # and merge same-day near-duplicates. Trusted (bypass/car-dedicated) sources
    # are exempt from the automotive re-vet.
    enrich_enabled = clean_ws(os.getenv("ENABLE_EXPORT_ENRICHMENT", "1")).lower() not in ("0", "false", "no")
    if enrich_enabled:
        try:
            merged_dicts = enrich_events_for_export(
                merged_dicts,
                geocache,
                url_cache,
                cfg,
                verify_dates=clean_ws(os.getenv("ENABLE_LINK_DATE_CHECK", "1")).lower() not in ("0", "false", "no"),
                lookup_addresses=clean_ws(os.getenv("ENABLE_ADDRESS_LOOKUP", "1")).lower() not in ("0", "false", "no"),
                fetch_facebook=True,
                revet_automotive=True,
                max_workers=parse_int_env("EXPORT_ENRICH_WORKERS", 6),
                trusted_sources=source_bypass_automotive,
            )
        except Exception as ex:
            log_exception_context("Export enrichment failed (continuing with un-enriched rows)", ex)
    else:
        log("ℹ️ Export enrichment disabled (ENABLE_EXPORT_ENRICHMENT=0).")

    geocache = prune_cache_by_age(geocache, now_et, days=180, label="geocode_cache")
    url_cache = prune_cache_by_age(url_cache, now_et, days=180, label="url_cache")
    save_json(GEOCODE_CACHE_PATH, geocache)
    save_json(URL_CACHE_PATH, url_cache)

    export_rows, _ = normalize_export_schema(merged_dicts)
    screenshot_rows_written = sum(1 for row in export_rows if clean_ws(str(row.get("Source", ""))).lower().startswith("screenshot:"))
    log(f"📸 Screenshot export rows written: {screenshot_rows_written}")

    payload = {
        "generated_at_iso": now_et_iso(),
        "count": len(export_rows),
        "events": export_rows,
    }
    save_json(EVENTS_JSON_PATH, payload)
    write_csv(export_rows, EVENTS_CSV_PATH)

    update_apex_spreadsheet(export_rows)
    spreadsheet_id = os.getenv("APEX_SPREADSHEET_ID")
    if spreadsheet_id:
        log(f"📄 Sheets URL: https://docs.google.com/spreadsheets/d/{spreadsheet_id}")

    run_dir = RUN_ARTIFACT_DIR or os.path.join(RUNS_DIR, datetime.now().strftime('%Y-%m-%d_%H%M%S'))
    os.makedirs(run_dir, exist_ok=True)

    run_report = {
        "generated_at_iso": now_et_iso(),
        "raw_events": len(raw_events),
        "incoming_after_filters": len(incoming),
        "merged_total": len(export_rows),
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
    log(f"📄 Sheets rows written (excluding header): {len(merged_dicts)}")
    log(f"🔎 Source summary: ok={len(ok_sources)} skipped={len(skipped_sources)} failed={len(failed_sources)}")
    log_source_health_summary(source_run_stats)
    sheet_skip_totals: Counter = Counter()
    for stat in source_run_stats:
        diag = SOURCE_DIAGNOSTICS.get(stat.get("name", ""), {})
        if isinstance(diag.get("sheet_skip_reasons"), dict):
            sheet_skip_totals.update(diag.get("sheet_skip_reasons") or {})
    if sheet_skip_totals:
        log("📋 Top sheet skip reasons: " + ", ".join(f"{k}={v}" for k, v in sheet_skip_totals.most_common(5)))
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
    log(f"✅ Done. Incoming: {len(incoming)} | Total: {len(merged_dicts)}")
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
        if "--self-test-prune" in sys.argv:
            run_prune_self_test()
            sys.exit(0)
        main()
    except Exception as ex:
        log(f"❌ Fatal: {ex}")
        raise
