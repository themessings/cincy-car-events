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

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)

DEFAULT_APEX_FOLDER_ID = "1bd9LR2JfE7AJm9Z5dwlIc_u5WpO3atQM"
APEX_PARENT_FOLDER_ID = os.getenv("APEX_PARENT_FOLDER_ID", DEFAULT_APEX_FOLDER_ID)
APEX_PARENT_FOLDER_URL = os.getenv("APEX_PARENT_FOLDER_URL")
APEX_SUBFOLDER_NAME = os.getenv("APEX_SUBFOLDER_NAME", "Apex events")
APEX_SPREADSHEET_NAME = os.getenv("APEX_SPREADSHEET_NAME", "Apex Events")
APEX_SHARED_DRIVE_ID = os.getenv("APEX_SHARED_DRIVE_ID")
APEX_SHARE_WITH_EMAIL = os.getenv("APEX_SHARE_WITH_EMAIL")


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
    # Earth radius in miles
    R = 3958.7613
    import math

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dl / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


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
    # Lightweight heuristic: look for "City, ST"
    m = re.search(r"([A-Za-z .'-]+),\s*([A-Z]{2})\b", location or "")
    if m:
        return f"{m.group(1).strip()}, {m.group(2)}"
    return ""


def categorize(title: str, location: str, cfg: dict) -> str:
    t = (title + " " + location).lower()
    for kw in cfg["categorization"]["rally_keywords"]:
        if kw.lower() in t:
            return "rally"
    # default to local unless rally keyword triggers
    return "local"


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

    # Nominatim usage policy: be gentle
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": place, "format": "json", "limit": 1}
    headers = {"User-Agent": "cincy-car-events-bot/1.0 (github actions)"}

    try:
        r = requests.get(url, params=params, headers=headers, timeout=20)
        r.raise_for_status()
        data = r.json()
        time.sleep(1.1)  # polite throttle
        if data:
            lat, lon = data[0]["lat"], data[0]["lon"]
            cache[place] = {"lat": lat, "lon": lon}
            return float(lat), float(lon)
        cache[place] = None
        return None
    except Exception:
        cache[place] = None
        return None


def miles_from_home(lat: float, lon: float, home_lat: float, home_lon: float) -> float:
    return haversine_miles(home_lat, home_lon, lat, lon)


# -------------------------
# Source collectors
# -------------------------
def fetch_html(url: str) -> BeautifulSoup:
    headers = {"User-Agent": "cincy-car-events-bot/1.0 (github actions)"}
    r = requests.get(url, headers=headers, timeout=30)
    r.raise_for_status()
    return BeautifulSoup(r.text, "html.parser")


def collect_carsandcoffeeevents_ohio(source: dict) -> List[dict]:
    """
    Parses the Ohio listing page where events are listed in repeating blocks.
    """
    soup = fetch_html(source["url"])
    events = []

    # The site uses a typical "tribe-events" (Modern Tribe) structure.
    # We'll grab event blocks by common selectors and degrade gracefully.
    blocks = soup.select(".tribe-events-calendar-list__event-row, article.tribe-events-calendar-list__event")
    if not blocks:
        # fallback for older markup
        blocks = soup.select(".tribe-events-calendar-list__event")

    for b in blocks:
        title_el = b.select_one(".tribe-events-calendar-list__event-title a, .tribe-event-url, a.tribe-events-calendar-list__event-title-link")
        title = clean_ws(title_el.get_text()) if title_el else ""
        url = title_el["href"].strip() if title_el and title_el.has_attr("href") else source["url"]

        time_el = b.select_one("time")
        start_txt = clean_ws(time_el.get("datetime") or time_el.get_text()) if time_el else ""
        # If datetime attr exists it's usually ISO; otherwise human text.
        start_dt = parse_dt(start_txt) if start_txt else None

        # Location
        loc_el = b.select_one(".tribe-events-calendar-list__event-venue-title, .tribe-events-calendar-list__event-venue")
        addr_el = b.select_one(".tribe-events-calendar-list__event-venue-address, .tribe-events-venue-details")
        location = clean_ws((loc_el.get_text() if loc_el else "") + " " + (addr_el.get_text() if addr_el else ""))

        # If no start, skip (we can’t reliably filter)
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
    """
    Parses a "series" events page on carsandcoffeeevents.com where each entry is a date row.
    """
    soup = fetch_html(source["url"])
    events = []

    # Try to find recurring date blocks
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

        # normalize tz-aware
        if isinstance(start_dt, datetime) and start_dt.tzinfo is None:
            start_dt = start_dt.replace(tzinfo=timezone.utc).astimezone(tz.gettz("America/New_York"))
        elif isinstance(start_dt, datetime):
            start_dt = start_dt.astimezone(tz.gettz("America/New_York"))
        else:
            # date-only event
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


# -------------------------
# Normalize + filter + store
# -------------------------
def to_event_items(raw_events: List[dict], cfg: dict, geocache: Dict[str, dict]) -> List[EventItem]:
    home_lat = cfg["home"]["lat"]
    home_lon = cfg["home"]["lon"]

    lookahead_days = int(cfg["filters"]["lookahead_days"])
    drop_past_days = int(cfg["filters"]["drop_past_days"])
    window_start = datetime.now(tz=tz.gettz("America/New_York")) - timedelta(days=drop_past_days)
    window_end = datetime.now(tz=tz.gettz("America/New_York")) + timedelta(days=lookahead_days)

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

        # time window filter
        if start_dt < window_start or start_dt > window_end:
            continue

        city_state = guess_city_state(location)

        # geocode location (use city_state first if present)
        query = city_state or location
        latlon = geocode(query, geocache)
        lat = lon = None
        miles = None
        if latlon:
            lat, lon = latlon
            miles = miles_from_home(lat, lon, home_lat, home_lon)

        cat = categorize(title, location, cfg)

        # distance filter
        if miles is not None:
            if cat == "local" and miles > local_max:
                continue
            if cat == "rally" and miles > rally_max:
                continue
        else:
            # If we can't geocode, keep it but mark unknown distance;
            # you can choose to drop these if you prefer.
            pass

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
    # Dedup key: (title normalized + start date + url fallback)
    def key(ev: EventItem) -> str:
        t = re.sub(r"\s+", " ", ev.title.lower()).strip()
        s = ev.start_iso[:16]  # minute-level
        u = ev.url or ""
        return f"{t}||{s}||{u}"

    merged: Dict[str, EventItem] = {key(e): e for e in existing}

    for ev in incoming:
        k = key(ev)
        if k in merged:
            # update last_seen + any missing fields
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

    # Sort by start time
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
            d = asdict(ev)
            w.writerow(d)


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
    response = drive.files().list(
        q=query,
        fields="files(id, name)",
        **drive_list_kwargs(),
    ).execute()
    files = response.get("files", [])
    if files:
        return files[0]["id"]

    metadata = {
        "name": folder_name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    created = drive.files().create(body=metadata, fields="id", supportsAllDrives=True).execute()
    return created["id"]


def find_or_create_spreadsheet(drive, parent_id: str, name: str) -> str:
    query = (
        "mimeType='application/vnd.google-apps.spreadsheet' "
        f"and name='{name}' "
        f"and '{parent_id}' in parents "
        "and trashed=false"
    )
    response = drive.files().list(
        q=query,
        fields="files(id, name)",
        **drive_list_kwargs(),
    ).execute()
    files = response.get("files", [])
    if files:
        return files[0]["id"]

    metadata = {
        "name": name,
        "mimeType": "application/vnd.google-apps.spreadsheet",
        "parents": [parent_id],
    }
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
        owners = ", ".join(
            owner.get("emailAddress", "unknown")
            for owner in meta.get("owners", [])
            if owner.get("emailAddress")
        )
        print(f"   Access verified for folder: {name} (drive: {drive_id})")
        if owners:
            print(f"   Folder owners: {owners}")
        return True
    except Exception as ex:
        print(f"❌ Unable to access parent folder {parent_id}: {ex}")
        print("   Ensure the service account has Editor access to the APEX folder.")
        return False


def ensure_sheet_tab(sheets, spreadsheet_id: str, title: str) -> None:
    sheet_info = sheets.spreadsheets().get(
        spreadsheetId=spreadsheet_id,
        includeGridData=False,
    ).execute()
    titles = {sheet["properties"]["title"] for sheet in sheet_info.get("sheets", [])}
    if title in titles:
        return
    requests_body = {"requests": [{"addSheet": {"properties": {"title": title}}}]}
    sheets.spreadsheets().batchUpdate(spreadsheetId=spreadsheet_id, body=requests_body).execute()


def update_apex_spreadsheet(events: List[EventItem]) -> None:
    creds = get_google_credentials()
    if not creds:
        print("⚠️ Skipping Google Sheets update: missing GOOGLE_SERVICE_ACCOUNT_FILE or GOOGLE_SERVICE_ACCOUNT_JSON.")
        return

    if getattr(creds, "service_account_email", None):
        print(f"   Using Google service account: {creds.service_account_email}")

    drive = build("drive", "v3", credentials=creds)
    sheets = build("sheets", "v4", credentials=creds)

    parent_id = resolve_parent_folder_id()
    if parent_id != APEX_PARENT_FOLDER_ID:
        print(f"   Using Apex folder from URL: {parent_id}")

    if not verify_parent_access(drive, parent_id):
        return

    subfolder_id = find_or_create_subfolder(drive, parent_id, APEX_SUBFOLDER_NAME)
    spreadsheet_id = find_or_create_spreadsheet(drive, subfolder_id, APEX_SPREADSHEET_NAME)
    ensure_sheet_tab(sheets, spreadsheet_id, "Events")
    if APEX_SHARE_WITH_EMAIL:
        try:
            drive.permissions().create(
                fileId=spreadsheet_id,
                body={
                    "type": "user",
                    "role": "writer",
                    "emailAddress": APEX_SHARE_WITH_EMAIL,
                },
                sendNotificationEmail=False,
                supportsAllDrives=True,
            ).execute()
            print(f"   Shared sheet with: {APEX_SHARE_WITH_EMAIL}")
        except Exception as ex:
            print(f"⚠️ Unable to share sheet with {APEX_SHARE_WITH_EMAIL}: {ex}")

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

    sheets.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range="Events!A1",
        valueInputOption="RAW",
        body={"values": values},
    ).execute()

    print(f"   Updated Google Sheet: {APEX_SPREADSHEET_NAME} ({spreadsheet_id})")
    print(f"   Sheet URL: https://docs.google.com/spreadsheets/d/{spreadsheet_id}")
    print(f"   Folder URL: https://drive.google.com/drive/folders/{subfolder_id}")


def main():
    cfg = load_yaml(CONFIG_PATH)

    geocache = load_json(GEOCODE_CACHE_PATH, {})
    existing_raw = load_json(EVENTS_JSON_PATH, {"events": []})
    existing = [EventItem(**e) for e in existing_raw.get("events", [])]

    raw_events: List[dict] = []

    for s in cfg.get("sources", []):
        stype = s.get("type")
        try:
            if stype == "html_carsandcoffeeevents_ohio":
                raw_events.extend(collect_carsandcoffeeevents_ohio(s))
            elif stype == "html_wordpress_events_list":
                raw_events.extend(collect_wordpress_events_series(s))
            elif stype == "ics":
                raw_events.extend(collect_ics(s))
            else:
                print(f"Skipping unknown source type: {stype} ({s.get('name')})")
        except Exception as ex:
            print(f"Source failed: {s.get('name')} :: {ex}")

    incoming = to_event_items(raw_events, cfg, geocache)
    merged = dedupe_merge(existing, incoming)

    # Persist
    save_json(GEOCODE_CACHE_PATH, geocache)

    payload = {
        "generated_at_iso": now_et_iso(),
        "count": len(merged),
        "events": [asdict(e) for e in merged],
    }
    save_json(EVENTS_JSON_PATH, payload)
    write_csv(merged, EVENTS_CSV_PATH)
    update_apex_spreadsheet(merged)

    # Snapshot this run (append-only history)
    run_stamp = datetime.now(tz=tz.gettz("America/New_York")).strftime("%Y-%m-%d_%H%M%S")
    run_path = os.path.join(RUNS_DIR, f"run_{run_stamp}.json")
    save_json(
        run_path,
        {
            "generated_at_iso": payload["generated_at_iso"],
            "incoming_count": len(incoming),
            "merged_count": len(merged),
            "incoming_sample": [asdict(e) for e in incoming[:50]],
        },
    )

    print(f"✅ Done. Incoming: {len(incoming)} | Total: {len(merged)}")
    print(f"   Wrote: {EVENTS_JSON_PATH}")
    print(f"   Wrote: {EVENTS_CSV_PATH}")
    print(f"   Wrote: {run_path}")


if __name__ == "__main__":
    main()
