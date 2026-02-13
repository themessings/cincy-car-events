#!/usr/bin/env python3
import csv
import json
import os
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import dateparser
import requests
import yaml
from bs4 import BeautifulSoup
from dateutil import tz
from icalendar import Calendar
from google.oauth2 import service_account
from googleapiclient.discovery import build

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

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

def extract_facebook_event_id(url: str) -> Optional[str]:
    """
    Accepts URLs like:
      https://www.facebook.com/events/1234567890/
      https://m.facebook.com/events/1234567890/?ref=...
    Returns event_id as string.
    """
    if not url:
        return None
    m = re.search(r"facebook\.com/events/(\d+)", url)
    if m:
        return m.group(1)
    m = re.search(r"m\.facebook\.com/events/(\d+)", url)
    if m:
        return m.group(1)
    return None


def normalize_facebook_event_url(url: str) -> str:
    url = clean_ws(url)
    # Strip tracking params
    url = re.sub(r"\?.*$", "", url)
    # Ensure canonical www
    url = url.replace("m.facebook.com", "www.facebook.com")
    return url


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
    if not FACEBOOK_ACCESS_TOKEN:
        return None

    url = f"https://graph.facebook.com/v18.0/{event_id}"
    params = {
        "access_token": FACEBOOK_ACCESS_TOKEN,
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


def collect_web_search_facebook_events_serpapi(source: dict, url_cache: Dict[str, dict]) -> List[dict]:
    """
    1) SerpAPI discovers facebook.com/events/<id> URLs
    2) For each, try Graph API by event_id
    3) If Graph fails, fallback to HTML parsing
    """
    query = source.get("query", "")
    max_results = int(source.get("max_results", 50))
    source_name = source.get("name", "facebook:serpapi")

    links: List[str] = []
    if clean_ws(query):
        links = serpapi_search(query, max_results=max_results)
    elif SERPAPI_API_KEY:
        cfg = load_yaml(CONFIG_PATH)
        for q in build_serpapi_discovery_queries(cfg, for_facebook=True, limit=10):
            links.extend(serpapi_search(q, max_results=min(max_results, 25)))
            time.sleep(0.2)
    else:
        log(f"ℹ️ SerpAPI disabled; skipping {source_name}.")
        return []

    if not links:
        return []

    out: List[dict] = []
    now = datetime.now(tz=tz.gettz("America/New_York"))

    fb_links = []
    for u in links:
        u = clean_ws(u)
        if "facebook.com/events/" in u:
            fb_links.append(normalize_facebook_event_url(u))
    fb_links = list(dict.fromkeys(fb_links))

    for event_url in fb_links:
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
                    })
                    continue
            except Exception:
                pass

        ev = None
        event_id = extract_facebook_event_id(event_url)
        if event_id:
            try:
                ev = fetch_facebook_event_via_graph(event_id)
            except Exception:
                ev = None

        if not ev:
            try:
                html = fetch_text(event_url)
                ev = parse_facebook_event_from_html(event_url, html)
            except Exception:
                ev = None

        if not ev:
            url_cache[event_url] = {"fetched_at_iso": now.isoformat(), "event": None}
            continue

        raw = {
            "title": ev["title"], "start_dt": ev["start_dt"], "end_dt": ev["end_dt"],
            "location": ev.get("location", ""), "url": ev.get("url", event_url), "source": source_name,
        }
        out.append(raw)
        url_cache[event_url] = {
            "fetched_at_iso": now.isoformat(),
            "event": {
                "title": raw["title"], "start_iso": raw["start_dt"].isoformat(), "end_iso": raw["end_dt"].isoformat(),
                "location": raw.get("location", ""), "url": raw.get("url", event_url),
            },
        }
        time.sleep(0.4)

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

def load_facebook_pages_from_sheet() -> List[dict]:
    """
    Tab: Pages
    Headers: page_name | page_id | enabled | link
    Returns enabled rows only.
    """
    sheet_id = os.getenv("APEX_FACEBOOK_PAGES_SHEET_ID")
    if not sheet_id:
        log("⚠️ Missing APEX_FACEBOOK_PAGES_SHEET_ID.")
        return []

    creds = get_google_credentials()
    if not creds:
        log("⚠️ Cannot load Facebook pages: missing Google credentials.")
        return []

    sheets = build("sheets", "v4", credentials=creds)

    resp = sheets.spreadsheets().values().get(
        spreadsheetId=sheet_id,
        range="Pages!A1:D2000",
    ).execute()

    rows = resp.get("values", [])
    if not rows or len(rows) < 2:
        log("⚠️ Pages tab is empty (need headers + at least 1 row).")
        return []

    headers = [clean_ws(h).lower() for h in rows[0]]
    idx = {h: i for i, h in enumerate(headers)}

    def cell(r: List[str], key: str) -> str:
        i = idx.get(key)
        if i is None or i >= len(r):
            return ""
        return clean_ws(r[i])

    out: List[dict] = []
    for r in rows[1:]:
        page_name = cell(r, "page_name")
        page_id = cell(r, "page_id")
        enabled = cell(r, "enabled").upper() in ("TRUE", "YES", "Y", "1")
        link = cell(r, "link")

        if not enabled or not page_id:
            continue

        out.append({"page_name": page_name or page_id, "page_id": page_id, "link": link})

    log(f"✅ Pages tab loaded: enabled={len(out)}")
    return out


def collect_facebook_events_from_pages(pages: List[dict]) -> List[dict]:
    """
    Graph API pull for /{id}/events (page/group IDs that your token can access).
    """
    if not FACEBOOK_ACCESS_TOKEN:
        log("⚠️ Skipping Facebook events: missing FACEBOOK_ACCESS_TOKEN.")
        return []

    if not pages:
        log("ℹ️ No enabled Facebook IDs provided; skipping FB events.")
        return []

    out: List[dict] = []

    for p in pages:
        page_id = (p.get("page_id") or "").strip()
        page_name = clean_ws(p.get("page_name") or page_id)
        if not page_id:
            continue

        log(f"--- Facebook pull start: {page_name} ({page_id})")

        try:
            url = f"https://graph.facebook.com/v18.0/{page_id}/events"
            params = {
                "access_token": FACEBOOK_ACCESS_TOKEN,
                "fields": "name,start_time,end_time,place,timezone,description",
                "limit": 100,
            }

            entity_count = 0
            while url:
                r = requests.get(url, params=params, timeout=30)
                if r.status_code != 200:
                    log(f"⚠️ FB non-200 for {page_name} ({page_id}): {r.status_code} :: {(r.text or '')[:250]}")
                    break

                payload = r.json()

                for item in payload.get("data", []) or []:
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
                        continue

                    event_id = item.get("id")
                    out.append(
                        {
                            "title": title,
                            "start_dt": start_dt,
                            "end_dt": end_dt or (start_dt + timedelta(hours=2)),
                            "location": location,
                            "url": f"https://www.facebook.com/events/{event_id}" if event_id else "",
                            "source": f"Facebook: {page_name}",
                        }
                    )
                    entity_count += 1

                paging = payload.get("paging", {}) or {}
                url = paging.get("next")
                params = None
                time.sleep(0.2)

            log(f"--- Facebook pull done: {page_name} ({page_id}) events={entity_count}")
        except Exception as ex:
            log(f"⚠️ Facebook events failed for {page_name} ({page_id}): {ex}")

    return out


def collect_facebook_from_pages_sheet() -> List[dict]:
    """
    Reads Pages tab then pulls events for those IDs.
    Use this in main() instead of calling read_facebook_pages_sheet() directly.
    """
    pages = load_facebook_pages_from_sheet()
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
def is_automotive_focus_event(title: str, location: str, source: str, cfg: dict, url: str = "") -> bool:
    """
    Strong automotive gate: include if focus keyword matches OR trusted source/platform,
    and exclude common non-automotive false positives.
    """
    filters = (cfg or {}).get("filters", {})
    text = clean_ws(f"{title} {location} {source} {url}").lower()
    if not text:
        return False

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

    if any(x in text for x in hard_exclusions):
        return False
    if exclude_keywords and any(x in text for x in exclude_keywords):
        return False

    has_focus = any(k in text for k in focus_keywords) if focus_keywords else False
    is_trusted = any(k in text for k in trusted_platforms)

    return has_focus or is_trusted

def log(msg: str) -> None:
    print(msg, flush=True)


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
    headers = {"User-Agent": "cincy-car-events-bot/1.0 (github actions)"}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


def fetch_text(url: str) -> str:
    """
    Fetch page text with basic retries and a real browser-ish UA.
    This helps when sites rate limit or return transient 403/429/5xx.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    last_err = None
    for attempt in range(1, 4):
        try:
            r = requests.get(url, headers=headers, timeout=35)
            # Some sites return 429/403 transiently; retry
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


def collect_facebook_page_events(source: dict) -> List[dict]:
    page_id = source.get("page_id")
    if not page_id:
        page_id_env = source.get("page_id_env")
        if page_id_env:
            page_id = os.getenv(page_id_env)
    if not page_id:
        log("⚠️ Skipping Facebook events: missing page_id/page_id_env.")
        return []
    if not FACEBOOK_ACCESS_TOKEN:
        log("⚠️ Skipping Facebook events: missing FACEBOOK_ACCESS_TOKEN.")
        return []

    url = f"https://graph.facebook.com/v18.0/{page_id}/events"
    params = {
        "access_token": FACEBOOK_ACCESS_TOKEN,
        "fields": "name,start_time,end_time,place,timezone,description",
        "limit": 100,
    }

    events: List[dict] = []
    while url:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        payload = r.json()

        for item in payload.get("data", []):
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
                continue

            events.append(
                {
                    "title": title,
                    "start_dt": start_dt,
                    "end_dt": end_dt or (start_dt + timedelta(hours=2)),
                    "location": location,
                    "url": f"https://www.facebook.com/events/{item.get('id')}" if item.get("id") else "",
                    "source": source["name"],
                }
            )

        paging = payload.get("paging", {})
        url = paging.get("next")
        params = None

    return events


# -------------------------
# NEW: Search the web (SerpAPI) + parse schema.org Event
# -------------------------
def serpapi_search(query: str, max_results: int = 20) -> List[str]:
    """
    Returns a list of organic result links using SerpAPI (Google engine).
    """
    if not SERPAPI_API_KEY:
        log("⚠️ Skipping SerpAPI search: missing SERPAPI_API_KEY.")
        return []

    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google",
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "num": min(max_results, 100),
        "hl": "en",
        "gl": "us",
    }

    r = requests.get(url, params=params, timeout=45)
    r.raise_for_status()
    data = r.json()

    links: List[str] = []
    for item in data.get("organic_results", []) or []:
        link = item.get("link")
        if link:
            links.append(clean_ws(link))

    # Dedup while preserving order
    seen = set()
    out = []
    for u in links:
        if u in seen:
            continue
        seen.add(u)
        out.append(u)

    return out[:max_results]




def build_serpapi_discovery_queries(cfg: dict, for_facebook: bool = False, limit: int = 16) -> List[str]:
    """Build rotating car-focused SerpAPI queries across geos, time windows, and event types."""
    now = datetime.now(tz=tz.gettz("America/New_York"))
    y = now.year
    month_name = now.strftime("%B")
    next_month_name = (now + timedelta(days=32)).strftime("%B")

    geos = [
        "Cincinnati", "Northern Kentucky", "NKY", "Mason OH", "West Chester OH", "Loveland OH",
        "Dayton OH", "Columbus OH", "Louisville KY", "Indianapolis IN",
    ]
    event_types = [
        '"cars and coffee"', '"cars & coffee"', '"car meet"', '"car meetup"', '"cruise-in"',
        '"car show"', 'autocross', '"track day"', 'HPDE', 'rally', '"driving tour"', '"road rally"',
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
    merged = configured + out
    deduped = list(dict.fromkeys(merged))
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
def parse_facebook_event_page(event_url: str) -> Optional[dict]:
    """Parse public Facebook event page: Graph first (if ID+token), then HTML fallback."""
    event_id = extract_facebook_event_id(event_url)
    if event_id:
        try:
            g = fetch_facebook_event_via_graph(event_id)
            if g:
                g["source"] = "facebook:discovered"
                return g
        except Exception:
            pass

    try:
        html = fetch_text(event_url)
    except Exception as ex:
        log(f"⚠️ FB event fetch failed: {event_url} :: {ex}")
        return None

    if any(x in html.lower() for x in ["log in to facebook", "login_form", "consent", "accept all cookies"]):
        log(f"⚠️ FB returned consent/login page: {event_url}")
        return None

    parsed = parse_facebook_event_from_html(event_url, html)
    if not parsed:
        return None
    parsed["source"] = "facebook:discovered"
    return parsed

def collect_facebook_events_serpapi_discovery(cfg: dict, url_cache: Dict[str, dict]) -> List[dict]:
    """
    Discovers FB event URLs via SerpAPI, then parses each event page.
    Uses url_cache to avoid re-fetching within 24h.
    """
    event_urls = collect_facebook_event_urls_serpapi(cfg)
    if not event_urls:
        return []

    out: List[dict] = []
    now = datetime.now(tz=tz.gettz("America/New_York"))

    for u in event_urls:
        u = clean_ws(u)
        if not u:
            continue

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
                        }
                    )
                    continue
            except Exception:
                pass

        ev = parse_facebook_event_page(u)
        if ev:
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
                },
            }
        else:
            url_cache[u] = {"fetched_at_iso": now.isoformat(), "event": None}

        time.sleep(0.4)

    log(f"   Discovered+parsed {len(out)} Facebook events from SerpAPI URLs.")
    return out

def collect_facebook_event_urls_serpapi(cfg: dict) -> List[str]:
    """Discover public FB event URLs via dynamic SerpAPI queries."""
    if not SERPAPI_API_KEY:
        log("ℹ️ SerpAPI disabled (missing SERPAPI_API_KEY); skipping FB URL discovery.")
        return []

    queries = build_serpapi_discovery_queries(cfg, for_facebook=True, limit=12)
    found: List[str] = []
    for q in queries:
        try:
            found.extend(serpapi_search(q, max_results=25))
        except Exception as ex:
            log(f"⚠️ SerpAPI discovery query failed: {q} :: {ex}")
        time.sleep(0.2)

    event_urls = []
    for u in found:
        u = normalize_facebook_event_url(u)
        m = re.search(r"(https?://(www\.)?facebook\.com/events/\d+)", u)
        if m:
            event_urls.append(m.group(1))

    out = list(dict.fromkeys(event_urls))
    log(f"   SerpAPI found {len(out)} Facebook event URLs (pre-parse).")
    return out

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

def collect_web_search_serpapi(source: dict, url_cache: Dict[str, dict]) -> List[dict]:
    """SerpAPI discovery -> URL dedupe -> schema/fallback parse with caching."""
    max_results = int(source.get("max_results", 20))
    source_name = source.get("name", "web_search_serpapi")

    queries = [clean_ws(source.get("query", ""))] if clean_ws(source.get("query", "")) else []
    if not queries:
        cfg = load_yaml(CONFIG_PATH)
        queries = build_serpapi_discovery_queries(cfg, for_facebook=False, limit=12)

    if not SERPAPI_API_KEY:
        log(f"ℹ️ SerpAPI disabled; skipping {source_name}.")
        return []

    links: List[str] = []
    for q in queries[:20]:
        links.extend(serpapi_search(q, max_results=max_results))
        time.sleep(0.2)

    links = list(dict.fromkeys(clean_ws(u) for u in links if clean_ws(u)))
    if not links:
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
            html = fetch_text(u)
            events = parse_schema_org_events_from_html(u, html)
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
        except Exception as ex:
            log(f"⚠️ Web page parse failed: {u} :: {ex}")
            url_cache[u] = {"fetched_at_iso": now.isoformat(), "events": []}

    return out

# -------------------------
# Normalize + filter + store
# -------------------------
def to_event_items(raw_events: List[dict], cfg: dict, geocache: Dict[str, dict]) -> List[EventItem]:
    home_lat = cfg["home"]["lat"]
    home_lon = cfg["home"]["lon"]

    lookahead_days = int(cfg["filters"]["lookahead_days"])
    drop_past_days = int(cfg["filters"]["drop_past_days"])
    now_et = datetime.now(tz=tz.gettz("America/New_York"))
    window_start = now_et - timedelta(days=drop_past_days)
    # Set lookahead_days to a negative value to include all future events.
    window_end = None if lookahead_days < 0 else (now_et + timedelta(days=lookahead_days))

    local_max = float(cfg["filters"]["local_max_miles"])
    rally_max = float(cfg["filters"]["rally_max_miles"])

    out: List[EventItem] = []

    for e in raw_events:
        title = clean_ws(e.get("title", ""))
        location = clean_ws(e.get("location", ""))
        url = clean_ws(e.get("url", ""))
        source = clean_ws(e.get("source", ""))
        start_dt: datetime = e["start_dt"]
        end_dt: datetime = e["end_dt"]

        if not title or not start_dt:
            continue

        if not is_automotive_focus_event(title, location, source, cfg, url=url):
            continue

        if start_dt < window_start:
            continue
        if window_end is not None and start_dt > window_end:
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
                continue
            if cat == "rally" and miles > rally_max:
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

    return out


def dedupe_merge(existing: List[EventItem], incoming: List[EventItem]) -> List[EventItem]:
    def key(ev: EventItem) -> str:
        t = re.sub(r"\s+", " ", ev.title.lower()).strip()
        s = ev.start_iso[:16]
        u = ev.url or ""
        return f"{t}||{s}||{u}"

    merged: Dict[str, EventItem] = {key(e): e for e in existing}

    for ev in incoming:
        k = key(ev)
        if k in merged:
            cur = merged[k]
            cur.last_seen_iso = ev.last_seen_iso
            if (cur.miles_from_cincy is None) and (ev.miles_from_cincy is not None):
                cur.miles_from_cincy = ev.miles_from_cincy
                cur.lat = ev.lat
                cur.lon = ev.lon
            if not cur.location and ev.location:
                cur.location = ev.location
            if not cur.city_state and ev.city_state:
                cur.city_state = ev.city_state
            if not cur.source and ev.source:
                cur.source = ev.source
            merged[k] = cur
        else:
            merged[k] = ev

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
        range="Events!A1:K",
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
    log(f"   FACEBOOK_ACCESS_TOKEN set? {'YES' if bool(FACEBOOK_ACCESS_TOKEN) else 'NO'}")
    log(f"   APEX_FACEBOOK_PAGES_SHEET_ID set? {'YES' if bool(os.getenv('APEX_FACEBOOK_PAGES_SHEET_ID')) else 'NO'}")
    log(f"   APEX_SPREADSHEET_ID set? {'YES' if bool(os.getenv('APEX_SPREADSHEET_ID')) else 'NO'}")

    cfg = load_yaml(CONFIG_PATH)
    sources = cfg.get("sources", [])
    log(f"   Config sources loaded: {len(sources)}")

    geocache = load_json(GEOCODE_CACHE_PATH, {})
    url_cache = load_json(URL_CACHE_PATH, {})

    existing_raw = load_json(EVENTS_JSON_PATH, {"events": []})
    existing = [EventItem(**e) for e in existing_raw.get("events", [])]
    existing = filter_existing_automotive_events(existing, cfg)

    raw_events: List[dict] = []
    source_run_stats: List[dict] = []

    serpapi_source_types = {"web_search_serpapi", "web_search_facebook_events_serpapi"}

    for s in sources:
        stype = s.get("type")
        sname = s.get("name", "(unnamed source)")

        if stype in serpapi_source_types and not serpapi_enabled:
            log(f"⚠️ Skipping SerpAPI source (missing key): {sname} [{stype}]")
            source_run_stats.append({"name": sname, "type": stype, "status": "skipped", "collected": 0})
            continue

        before_count = len(raw_events)
        try:
            if stype == "html_carsandcoffeeevents_ohio":
                raw_events.extend(collect_carsandcoffeeevents_ohio(s))
            elif stype == "html_wordpress_events_list":
                raw_events.extend(collect_wordpress_events_series(s))
            elif stype == "ics":
                raw_events.extend(collect_ics(s))
            elif stype == "facebook_page_events":
                raw_events.extend(collect_facebook_page_events(s))
            elif stype == "web_search_serpapi":
                raw_events.extend(collect_web_search_serpapi(s, url_cache))
            elif stype == "web_search_facebook_events_serpapi":
                raw_events.extend(collect_web_search_facebook_events_serpapi(s, url_cache))
            else:
                log(f"Skipping unknown source type: {stype} ({sname})")

            collected = len(raw_events) - before_count
            source_run_stats.append({"name": sname, "type": stype, "status": "ok", "collected": collected})
            log(f"🔎 Source complete: {sname} [{stype}] -> {collected} events")
        except Exception as ex:
            source_run_stats.append({"name": sname, "type": stype, "status": "failed", "collected": 0, "error": str(ex)})
            log(f"⚠️ Source failed: {sname} [{stype}] :: {ex}")

    try:
        fb_pages = load_facebook_pages_from_sheet()
        fb_events = collect_facebook_events_from_pages(fb_pages)
        raw_events.extend(fb_events)
        log(f"🔎 Source complete: Facebook Pages Sheet -> {len(fb_events)} events")
    except Exception as ex:
        log(f"⚠️ Facebook Pages sheet collection failed: {ex}")

    if serpapi_enabled:
        try:
            discovered = collect_facebook_events_serpapi_discovery(cfg, url_cache)
            raw_events.extend(discovered)
            log(f"🔎 Source complete: SerpAPI FB discovery -> {len(discovered)} events")
        except Exception as ex:
            log(f"⚠️ Facebook SerpAPI discovery failed: {ex}")
    else:
        log("⚠️ SERPAPI_API_KEY missing; skipping broad web discovery collectors.")

    log(f"📦 Raw events collected (pre-filter): {len(raw_events)}")
    ok_sources = [x for x in source_run_stats if x.get("status") == "ok"]
    failed_sources = [x for x in source_run_stats if x.get("status") == "failed"]
    log(f"🔎 Source summary: ok={len(ok_sources)} failed={len(failed_sources)}")

    incoming = to_event_items(raw_events, cfg, geocache)
    log(f"✅ Incoming after filters: {len(incoming)}")

    merged = dedupe_merge(existing, incoming)
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

    run_report = {
        "generated_at_iso": now_et_iso(),
        "raw_events": len(raw_events),
        "incoming_after_filters": len(incoming),
        "merged_total": len(merged),
        "source_stats": source_run_stats,
        "serpapi_enabled": serpapi_enabled,
    }
    run_path = os.path.join(RUNS_DIR, f"run_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.json")
    save_json(run_path, run_report)

    skipped_sources = [x for x in source_run_stats if x.get("status") == "skipped"]
    log(f"📄 Sheets rows written (excluding header): {len(merged)}")
    log(f"🔎 Source summary: ok={len(ok_sources)} skipped={len(skipped_sources)} failed={len(failed_sources)}")
    log(f"✅ Done. Incoming: {len(incoming)} | Total: {len(merged)}")
    log(f"   Wrote: {EVENTS_JSON_PATH}")
    log(f"   Wrote: {EVENTS_CSV_PATH}")
    log(f"   Wrote: {GEOCODE_CACHE_PATH}")
    log(f"   Wrote: {URL_CACHE_PATH}")
    log(f"   Wrote: {run_path}")
    log(f"⏱️ Total runtime seconds: {round(_time.time() - t0, 1)}")



if __name__ == "__main__":
    main()
