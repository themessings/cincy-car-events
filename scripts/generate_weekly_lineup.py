#!/usr/bin/env python3
"""
Generate the Apex weekend/weekly lineup carousel (JPEGs) and social caption
(.txt) from the car-events Google Sheet, then upload them to the same Drive
folder that holds the Apex logo.

Ported from a Colab notebook the same lineup was previously generated with
manually, one run per week, after the collector finished. This script is the
automated equivalent: it runs as a step in car_events.yml right after the
collector, reading the just-updated sheet and picking its own date range
(Friday through the following Thursday) instead of prompting for one.

Auth: reuses the same service account as the collector
(GOOGLE_SERVICE_ACCOUNT_FILE or GOOGLE_SERVICE_ACCOUNT_JSON), which already
has Sheets + Drive scopes and already has an established, working Drive
integration in this repo (scripts/import_drive_event_screenshots.py).

Dependencies (see requirements.txt): pandas, python-dateutil, pytz, requests,
rapidfuzz, google-api-python-client, google-auth, Pillow, geopy, qrcode.
"""

import json
import os
import io
import math
import datetime as dt
import re
import glob
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any
from collections import defaultdict

import requests
import pandas as pd
from dateutil import parser as dateparser
import pytz
from rapidfuzz import fuzz
from geopy.distance import geodesic

from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageChops
import qrcode

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaFileUpload


# ----------------------------
# CONFIG
# ----------------------------
SPREADSHEET_ID = os.getenv("APEX_SPREADSHEET_ID", "1lVpqhmUOQDZywjGeYxgm7ILNXqP3l6Z74pVXw1oKSQ8")
SHEET_RANGE = "Events!A1:M4000"

TZ_NAME = "America/New_York"

ORIGIN_LABEL = "45215"
ORIGIN_COORDS = (39.2400, -84.4570)

# This is Joel's own "Thursday Post" folder (My Drive root) — the real,
# actively-used destination for these weekly outputs, confirmed directly by
# him 2026-09-04 (an earlier attempt targeted a different, disused "Apex
# events" folder under APEX/Apex Events by mistake). Shared with the
# collector's own service account
# (car-events-bot@cincy-car-events-export.iam.gserviceaccount.com) as
# Editor, with the same fixed-filename set pre-seeded there too, so uploads
# always work regardless of what happens to any single file's sharing. The
# logo is fetched independently and degrades gracefully on its own if it
# ever goes missing.
DRIVE_OUTPUT_FOLDER_ID = "15Kd4_-eR16U--_E4Uzr0_MF5zBp1SWzI"
LOGO_FILE_ID = "1GM3Kj9FrPxMxGpn0v6Afe9MvM2TFdWVM"
LOGO_FORCE_WHITE = True

# ==================== SPOTTID SPONSOR CALLOUT ====================
# Logo pulled from the SpottID website (getspottid.com). It's the square
# app icon (white car + signal waves on a blue tile).
SPOTTID_LOGO_URL = "https://assets.cdn.filesafe.space/SCrJwytxiBvS7QAjT03w/media/69f404db22c9963731c28ded.png"

# Two QR codes that go STRAIGHT to each app store (Apple + Android), so the
# callout can show one per platform. Short Apple URL keeps the QR less dense.
SPOTTID_APPLE_URL = "https://apps.apple.com/app/id6751887198"
SPOTTID_PLAY_URL  = "https://play.google.com/store/apps/details?id=com.apex.spottid"

SPONSOR_LABEL        = "SPONSORED BY"
SPOTTID_WORDMARK     = "SpottID"
SPOTTID_TAGLINE_MAIN = "Where great Cars get noticed."
SPOTTID_CTA          = "Download the app now"
# ================================================================

OUTPUT_DIR = "lineup_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOGO_PATH = os.path.join(OUTPUT_DIR, "apex_logo.png")
LOGO_WHITE_PATH = os.path.join(OUTPUT_DIR, "apex_logo_white.png")
SPOTTID_LOGO_PATH = os.path.join(OUTPUT_DIR, "spottid_logo.png")
SPOTTID_LOGO_ROUND_PATH = os.path.join(OUTPUT_DIR, "spottid_logo_round.png")
SPOTTID_QR_APPLE_PATH = os.path.join(OUTPUT_DIR, "spottid_qr_apple.png")
SPOTTID_QR_PLAY_PATH  = os.path.join(OUTPUT_DIR, "spottid_qr_play.png")
CLOCK_ICON_PATH = os.path.join(OUTPUT_DIR, "clock.png")
PIN_ICON_PATH   = os.path.join(OUTPUT_DIR, "pin.png")

YELLOW   = (245, 204, 55)
BG       = (14, 14, 16)
WHITE    = (242, 242, 242)
GRAY     = (198, 198, 198)
MIDGRAY  = (145, 145, 145)
DARKGRAY = (88, 88, 88)
TOPLINE  = (110, 110, 110)

CALL_OUT_COLOR_MAP = {
    "FEATURED": YELLOW,
    "WORTH THE DRIVE": (250, 228, 140),
    "LOCAL STAPLE": (220, 220, 220),
}
DEFAULT_CALLOUT_COLOR = YELLOW

CANVAS_W, CANVAS_H = 1080, 1350
SAFE_PAD = 56

TOPLINE_Y = 16
HEADER_Y = 54
HEADER_LINE2_OFFSET = 78
SUB_Y = 214
DIVIDER_Y = 274
CONTENT_Y = 306

LEFT_COL_X = 54
RIGHT_COL_X = 552
GUTTER = 22
LEFT_W = RIGHT_COL_X - LEFT_COL_X - GUTTER
RIGHT_W = CANVAS_W - SAFE_PAD - RIGHT_COL_X

LOGO_MAX_W, LOGO_MAX_H = 230, 150
LOGO_X = CANVAS_W - SAFE_PAD - LOGO_MAX_W + 6
LOGO_Y = 38

WEATHER_X = 632
WEATHER_Y = 1088
WEATHER_W, WEATHER_H = 386, 168

# CTA text is drawn *upward* from this anchor (see make_base_canvas), so it
# must clear the weather box's bottom edge (WEATHER_Y + WEATHER_H = 1256)
# even at its full 2-line wrap — 1292 put the first line right through the
# box border. 1320 leaves a clean ~20px gap above and ~30px below within
# the 1350px canvas.
CTA_Y = 1320

# Left column now runs deeper to use the empty lower-left space.
# Right column still stops above the weather module.
LEFT_CONTENT_MAX_Y = 1240
RIGHT_CONTENT_MAX_Y = 1048

USE_PNG_ICONS = True
CLOCK_ICON_URL = "https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f553.png"
PIN_ICON_URL   = "https://raw.githubusercontent.com/twitter/twemoji/master/assets/72x72/1f4cd.png"

NO_EVENTS_LINE = "No standout posted events found - go drive something anyway."
CTA_TEXT = "Want your event featured in the weekend lineup? Send details to admin@ApexAutoLounge.com"

BASE_HASHTAGS = ["#theapexautolounge", "#carculture", "#carmeet", "#carsandcoffee", "#spottid"]

SPOTTID_CLAIM_URL = "https://link.getspottid.com/plate"

REGION_DISPLAY_NAME = {
    "Cincinnati Core": "Cincinnati",
    "North / Dayton": "Dayton",
    "Columbus": "Columbus",
    "Lexington": "Lexington",
    "Louisville": "Louisville",
    "Indiana": "Indy",
    "Other": "Other",
}

NUMBER_WORDS = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six", 7: "Seven"}

REGION_PRIORITY = [
    "Cincinnati Core",
    "North / Dayton",
    "Columbus",
    "Lexington",
    "Louisville",
    "Indiana",
    "Other",
]

REGION_CENTERS = {
    "Cincinnati Core": {"coords": (39.1031, -84.5120), "cities": {"cincinnati", "covington", "newport", "norwood", "blue ash", "oakley", "sharonville"}},
    "North / Dayton": {"coords": (39.7589, -84.1916), "cities": {"dayton", "hamilton", "middletown", "west chester", "springboro", "beavercreek", "centerville", "kettering", "lebanon"}},
    "Columbus": {"coords": (39.9612, -82.9988), "cities": {"columbus", "dublin", "hilliard", "powell", "westerville", "reynoldsburg", "grove city", "new albany"}},
    "Lexington": {"coords": (38.0406, -84.5037), "cities": {"lexington", "versailles", "nicholasville", "georgetown", "richmond"}},
    "Louisville": {"coords": (38.2527, -85.7585), "cities": {"louisville", "jeffersontown", "shepherdsville"}},
    "Indiana": {"coords": (39.7684, -86.1581), "cities": {"indianapolis", "greenwood", "carmel", "fishers", "lawrenceburg", "batesville", "clarksville", "new albany"}},
}

CITY_COORDS = {
    "cincinnati, oh": (39.1031, -84.5120),
    "hamilton, oh": (39.3995, -84.5613),
    "dayton, oh": (39.7589, -84.1916),
    "west chester, oh": (39.3440, -84.4083),
    "middletown, oh": (39.5151, -84.3983),
    "lebanon, oh": (39.4353, -84.2027),
    "columbus, oh": (39.9612, -82.9988),
    "reynoldsburg, oh": (39.9548, -82.8121),
    "versailles, ky": (38.0520, -84.7294),
    "lexington, ky": (38.0406, -84.5037),
    "louisville, ky": (38.2527, -85.7585),
    "indianapolis, in": (39.7684, -86.1581),
    "clarksville, in": (38.2967, -85.7600),
    "lawrenceburg, in": (39.0909, -84.8494),
    "batesville, in": (39.3006, -85.2222),
    "new albany, in": (38.2856, -85.8241),
    "bowling green, ky": (36.9685, -86.4808),
    "mansfield, oh": (40.7584, -82.5154),
    "marysville, oh": (40.2364, -83.3671),
}

AVG_SPEED_MPH_FALLBACK = 52
ROAD_FACTOR_FALLBACK = 1.15

IMAGE_TITLE_MAX_CHARS = 52
MAX_DRIVE_MINUTES = 105


# ----------------------------
# GOOGLE AUTH (service account — same one the collector uses)
# ----------------------------
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


# ----------------------------
# HELPERS
# ----------------------------
def coalesce_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and math.isnan(x):
        return ""
    s = str(x).strip()
    return "" if s.lower() in {"nan", "none", "null"} else s

def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def clean_title(s: str) -> str:
    s = coalesce_str(s)
    s = s.replace("‚Äú", '"').replace("‚Äù", '"').replace("‚Äô", "'")
    return normalize_spaces(s)

def clean_place(s: str) -> str:
    if not s:
        return ""
    s = normalize_spaces(s)
    s = s.replace(" ,", ",")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    out = []
    seen = set()
    for p in parts:
        key = p.lower()
        if key not in seen:
            out.append(p)
            seen.add(key)
    return ", ".join(out)

def truncate_text(s: str, max_chars: int) -> str:
    s = normalize_spaces(s)
    if len(s) <= max_chars:
        return s
    cut = s[:max_chars].rstrip()
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut + "..."

def normalize_colname(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else " " for ch in str(s or "")).strip()

def find_column_exact(df: pd.DataFrame, expected_name: str) -> Optional[str]:
    target = normalize_colname(expected_name)
    for c in df.columns:
        if normalize_colname(c) == target:
            return c
    return None

def best_match_column(columns: List[str], candidates: List[str], min_score: int = 65) -> Optional[str]:
    norm_cols = {c: normalize_colname(c) for c in columns}
    for c, nc in norm_cols.items():
        for cand in candidates:
            if cand in nc:
                return c
    best_score, best_col = -1, None
    for c, nc in norm_cols.items():
        for cand in candidates:
            score = fuzz.token_set_ratio(nc, cand)
            if score > best_score:
                best_score, best_col = score, c
    return best_col if best_score >= min_score else None

def get_exact_or_fuzzy_column(df: pd.DataFrame, exact_names: List[str], fuzzy_candidates: List[str]) -> Optional[str]:
    for name in exact_names:
        found = find_column_exact(df, name)
        if found is not None:
            return found
    return best_match_column(list(df.columns), fuzzy_candidates)

def parse_dt_safe(x) -> Optional[dt.datetime]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    s = str(x).strip()
    if not s:
        return None
    try:
        return dateparser.parse(s, fuzzy=True)
    except Exception:
        return None

def parse_date_only(x) -> Optional[dt.date]:
    d = parse_dt_safe(x)
    return d.date() if d else None

def parse_time_only(x) -> Optional[dt.time]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    s = str(x).strip()
    if not s or s.lower() in {"tba", "n/a", "na", "none", "-", "--"}:
        return None
    try:
        t = dateparser.parse(s, fuzzy=True)
        return t.time() if t else None
    except Exception:
        return None

def compute_default_week_range(now: Optional[dt.datetime] = None) -> List[dt.date]:
    """Friday through the following Thursday (7 days): today's Friday if run
    on a Friday, otherwise the coming Friday. No prompt — this always runs
    unattended in CI, right after the collector.

    This single rule covers both real invocation shapes: the scheduled run
    (always a Thursday) rolls forward to tomorrow's Friday and runs 6 days
    past it; a manual run on any other day does the same; a manual run on a
    Friday itself starts from that same day."""
    tz = pytz.timezone(TZ_NAME)
    now = now or dt.datetime.now(tz)
    today = now.date()
    wd = today.weekday()  # Monday=0 ... Friday=4, Saturday=5, Sunday=6
    days_until_friday = (4 - wd) % 7
    friday = today + dt.timedelta(days=days_until_friday)
    return [friday + dt.timedelta(days=i) for i in range(7)]

def ordinal(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        suf = "th"
    else:
        suf = {1:"st", 2:"nd", 3:"rd"}.get(n % 10, "th")
    return f"{n}{suf}"

def day_header_text(d: dt.date) -> str:
    return f"{d.strftime('%A').upper()} {ordinal(d.day).upper()}"

def time_sort_key(t: Optional[dt.time]):
    return (1, 0, 0) if t is None else (0, t.hour, t.minute)

def approx_time_block(start: Optional[dt.time], end: Optional[dt.time]) -> str:
    ref = start or end
    if ref is None:
        return "Time TBA"
    mins = ref.hour * 60 + ref.minute
    if mins < 660:
        return "Morning"
    if mins < 960:
        return "Midday"
    return "Evening"

def format_time_range(start: Optional[dt.time], end: Optional[dt.time]) -> str:
    def fmt(t: dt.time) -> str:
        hour = t.hour
        minute = t.minute
        ampm = "AM" if hour < 12 else "PM"
        h12 = hour % 12 or 12
        return f"{h12}:{minute:02d} {ampm}" if minute else f"{h12}:00 {ampm}"
    # An end time identical to the start time is a missing-data fallback
    # (e.g. an event whose real end time was never found), not a real
    # zero-length event — treat it the same as no end time at all.
    if end == start:
        end = None
    block = approx_time_block(start, end)
    if start and end:
        return f"{block} | {fmt(start)}-{fmt(end)}"
    if start:
        return f"{block} | {fmt(start)}"
    if end:
        return f"{block} | Until {fmt(end)}"
    return "Time TBA"

def format_drive_time(mins: Optional[int]) -> str:
    if mins is None:
        return ""
    if mins < 60:
        return f"{mins} min"
    h = mins // 60
    m = mins % 60
    if m == 0:
        return f"{h} hr"
    return f"{h} hr {m:02d} min"

def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_w: int) -> List[str]:
    if not text:
        return [""]
    words = text.split()
    lines, cur = [], ""
    for w in words:
        test = (cur + " " + w).strip()
        if draw.textbbox((0, 0), test, font=font)[2] <= max_w:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines

def clamp_lines(lines: List[str], max_lines: int) -> List[str]:
    if len(lines) <= max_lines:
        return lines
    clipped = lines[:max_lines]
    last = clipped[-1]
    clipped[-1] = (last[:-1] + "...") if len(last) > 1 else "..."
    return clipped

def download_file(url: str, dest_path: str, timeout=30):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    with open(dest_path, "wb") as f:
        f.write(r.content)

def _find_in_folder(drive_service, folder_id: str, filename: str, trashed: bool) -> Optional[str]:
    found = drive_service.files().list(
        q=f"name = '{filename}' and '{folder_id}' in parents and trashed = {'true' if trashed else 'false'}",
        fields="files(id)",
        supportsAllDrives=True,
        includeItemsFromAllDrives=True,
    ).execute().get("files", [])
    return found[0]["id"] if found else None

def upload_to_folder(drive_service, folder_id: str, local_path: str, drive_filename: Optional[str] = None) -> str:
    """Upload local_path into folder_id under drive_filename (or the local
    basename if not given), overwriting any existing file of that name —
    live if it's there, restored from trash if a quiet week parked it there.

    Service accounts have no Drive storage quota of their own and can only
    create new files inside a Shared Drive (unavailable on a personal
    account) — but updating an *existing* file's content never touches
    quota, and neither does trashing/restoring one, since no new file is
    ever being created. So this always targets a fixed filename and updates
    it in place rather than creating a fresh dated file every run; see
    DRIVE_OUTPUT_FOLDER_ID's pre-seeded files (created once, by hand, under
    the real account) for why those specific names must keep matching
    what's already sitting in that folder (live or trashed)."""
    filename = drive_filename or os.path.basename(local_path)
    media = MediaFileUpload(local_path, resumable=True)

    live_id = _find_in_folder(drive_service, folder_id, filename, trashed=False)
    if live_id:
        updated = drive_service.files().update(
            fileId=live_id, media_body=media, fields="id,webViewLink", supportsAllDrives=True
        ).execute()
        return updated["webViewLink"]

    trashed_id = _find_in_folder(drive_service, folder_id, filename, trashed=True)
    if trashed_id:
        updated = drive_service.files().update(
            fileId=trashed_id, body={"trashed": False}, media_body=media,
            fields="id,webViewLink", supportsAllDrives=True
        ).execute()
        return updated["webViewLink"]

    print(f"[WARN] No pre-seeded '{filename}' found (live or trashed) in the target "
          f"folder — creating it fresh, which needs Drive storage quota the service "
          f"account doesn't have and will likely fail. Seed this filename once under "
          f"the real account first (see DRIVE_OUTPUT_FOLDER_ID).")
    created = drive_service.files().create(
        body={"name": filename, "parents": [folder_id]}, media_body=media, fields="id,webViewLink", supportsAllDrives=True
    ).execute()
    return created["webViewLink"]

def trash_unused_drive_slides(drive_service, folder_id: str, used_count: int, max_seeded: int, name_prefix: str) -> None:
    """Park any pre-seeded slide slot beyond what this week actually needs in
    the trash, so a quiet week doesn't leave confusing near-blank placeholder
    images sitting visible in the folder. They come right back (restored,
    not recreated) the moment a busier week needs them again."""
    for i in range(used_count + 1, max_seeded + 1):
        name = f"{name_prefix}_{i:02d}.jpg"
        live_id = _find_in_folder(drive_service, folder_id, name, trashed=False)
        if live_id:
            drive_service.files().update(fileId=live_id, body={"trashed": True}, supportsAllDrives=True).execute()

def make_logo_white(logo_rgba: Image.Image) -> Image.Image:
    rgb = logo_rgba.convert("RGB")
    a = logo_rgba.split()[-1] if logo_rgba.mode == "RGBA" else Image.new("L", logo_rgba.size, 255)
    lum = ImageOps.grayscale(rgb)
    ink = ImageOps.invert(lum)
    ink = ImageChops.multiply(ink, a)
    white = Image.new("RGBA", logo_rgba.size, (255,255,255,255))
    white.putalpha(ink)
    return white

def round_icon_corners(im: Image.Image, frac: float = 0.20) -> Image.Image:
    """Return an app-icon-style rounded-corner copy of a square logo."""
    im = im.convert("RGBA")
    w, h = im.size
    r = int(min(w, h) * frac)
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w - 1, h - 1], radius=r, fill=255)
    mask = ImageChops.multiply(mask, im.split()[-1])
    out = im.copy()
    out.putalpha(mask)
    return out

def parse_city_state_text(s: str) -> Tuple[str, str]:
    s = clean_place(s)
    if not s:
        return "", ""

    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) >= 2:
        city = normalize_spaces(parts[0].title())
        state = parts[1].strip().upper()[:2]
        return city, state

    tokens = s.split()
    if len(tokens) >= 2 and re.fullmatch(r"[A-Za-z]{2}", tokens[-1]):
        return normalize_spaces(" ".join(tokens[:-1]).title()), tokens[-1].upper()

    return normalize_spaces(s.title()), ""

def maybe_city_key(city: str, state: str) -> str:
    if city and state:
        return f"{city.lower()}, {state.lower()}"
    return ""

def fallback_drive_minutes(origin: Tuple[float, float], dest: Tuple[float, float]) -> int:
    crow = geodesic(origin, dest).miles
    road_miles = crow * ROAD_FACTOR_FALLBACK
    base = (road_miles / AVG_SPEED_MPH_FALLBACK) * 60

    if crow <= 25:
        penalty = 10
    elif crow <= 60:
        penalty = 13
    else:
        penalty = 16

    mins = int(round(base + penalty))
    mins = min(mins, MAX_DRIVE_MINUTES)
    return max(8, mins)

def city_group_sort_key(city_name: str) -> Tuple[int, str]:
    city, state = parse_city_state_text(city_name)
    key = maybe_city_key(city, state)
    coords = CITY_COORDS.get(key)
    if coords:
        mins = fallback_drive_minutes(ORIGIN_COORDS, coords)
        return (mins, city_name.lower())
    return (9999, city_name.lower())

def infer_city_state_from_address(address: str, location: str) -> Tuple[str, str]:
    def extract_city_state(text: str) -> Tuple[str, str]:
        raw = normalize_spaces(text)
        if not raw:
            return "", ""
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        city, state = "", ""
        if len(parts) >= 2:
            city = parts[-2]
            state_chunk = parts[-1]
            m = re.search(r"\b([A-Z]{2})\b", state_chunk.upper())
            state = m.group(1) if m else state_chunk[:2].upper()
        else:
            tokens = raw.split()
            if len(tokens) >= 2 and re.fullmatch(r"[A-Za-z]{2}", tokens[-1]):
                state = tokens[-1].upper()
                city = " ".join(tokens[:-1])
        return city.strip(), state.strip()

    city, state = extract_city_state(address)
    if city and state:
        return city, state
    return extract_city_state(location)

def infer_region_from_city_state(city: str, state: str, coords: Optional[Tuple[float, float]]) -> str:
    city_lower = (city or "").lower()
    state_upper = (state or "").upper()

    if state_upper == "OH":
        if city_lower in REGION_CENTERS["Cincinnati Core"]["cities"]:
            return "Cincinnati Core"
        if city_lower in REGION_CENTERS["North / Dayton"]["cities"]:
            return "North / Dayton"
        if city_lower in REGION_CENTERS["Columbus"]["cities"]:
            return "Columbus"

    if state_upper == "KY":
        if city_lower in REGION_CENTERS["Lexington"]["cities"]:
            return "Lexington"
        if city_lower in REGION_CENTERS["Louisville"]["cities"]:
            return "Louisville"

    if state_upper == "IN":
        return "Indiana"

    if coords:
        dists = {r: geodesic(coords, cfg["coords"]).miles for r, cfg in REGION_CENTERS.items()}
        return min(dists.items(), key=lambda kv: kv[1])[0]

    return "Other"

def callout_fill_color(callout: str) -> Tuple[int, int, int]:
    if not callout:
        return DEFAULT_CALLOUT_COLOR
    return CALL_OUT_COLOR_MAP.get(callout.upper(), DEFAULT_CALLOUT_COLOR)


# ----------------------------
# DATA CLASSES
# ----------------------------
@dataclass
class Event:
    title: str
    date: dt.date
    start: Optional[dt.time]
    end: Optional[dt.time]
    location: str
    address: str
    url: str
    source_row: Dict[str, Any] = field(default_factory=dict)

    city: str = ""
    state: str = ""
    region: str = "Other"
    coords: Optional[Tuple[float, float]] = None
    drive_minutes: Optional[int] = None

    city_group: str = ""
    callout: str = ""

    display_place_caption: str = ""
    display_place_image: str = ""

    sheet_closest_city: str = ""
    sheet_callout: str = ""
    closest_city_source: str = ""
    callout_source: str = ""


# ----------------------------
# DATE RANGE
# ----------------------------
selected_dates = compute_default_week_range()
selected_dates = sorted(list(dict.fromkeys(selected_dates)))
selected_set = set(selected_dates)
print("\nSelected dates:", selected_dates)


# ----------------------------
# GOOGLE AUTH + LOGO
# ----------------------------
creds = get_google_credentials()
if creds is None:
    raise RuntimeError("Missing Google credentials: set GOOGLE_SERVICE_ACCOUNT_FILE or GOOGLE_SERVICE_ACCOUNT_JSON.")

drive_service = build("drive", "v3", credentials=creds)
sheets_service = build("sheets", "v4", credentials=creds)

TARGET_FOLDER_ID = DRIVE_OUTPUT_FOLDER_ID

try:
    req = drive_service.files().get_media(fileId=LOGO_FILE_ID, supportsAllDrives=True)
    fh = io.FileIO(LOGO_PATH, "wb")
    downloader = MediaIoBaseDownload(fh, req)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.close()
except Exception as ex:
    print(f"[WARN] Could not download Apex logo (fileId={LOGO_FILE_ID}): {ex}")
    print("[WARN] Continuing without a logo; slides/caption/Drive upload are unaffected.")
    # A failed download can still leave a truncated/empty file behind (the
    # file is opened for writing before the request runs) — remove it so the
    # os.path.exists() checks downstream correctly treat this as "no logo"
    # instead of crashing PIL on a 0-byte file.
    if os.path.exists(LOGO_PATH):
        os.remove(LOGO_PATH)

if os.path.exists(LOGO_PATH) and LOGO_FORCE_WHITE:
    try:
        _logo = Image.open(LOGO_PATH).convert("RGBA")
        _white = make_logo_white(_logo)
        _white.save(LOGO_WHITE_PATH, "PNG")
    except Exception:
        LOGO_WHITE_PATH = LOGO_PATH
else:
    LOGO_WHITE_PATH = LOGO_PATH


# ----------------------------
# FONTS
# ----------------------------
def find_font_file(preferred_names: List[str]) -> Optional[str]:
    roots = [
        "/usr/share/fonts",
        "/usr/local/share/fonts",
        "/usr/share/fonts/truetype",
        "/usr/share/fonts/opentype",
    ]
    hits = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for r, _, files in os.walk(root):
            for fn in files:
                if not fn.lower().endswith(".ttf"):
                    continue
                lower = fn.lower()
                for name in preferred_names:
                    if name in lower:
                        hits.append(os.path.join(r, fn))
    return hits[0] if hits else None

ARIAL_BOLD_PATH = find_font_file(["arialbd", "arial-bold", "arial_bold", "liberationsans-bold", "dejavusans-bold"])
ARIAL_REG_PATH  = find_font_file(["arial", "liberationsans-regular", "liberationsans", "dejavusans"])

if ARIAL_BOLD_PATH is None:
    ARIAL_BOLD_PATH = ARIAL_REG_PATH

def load_font(path: Optional[str], size: int):
    if path and os.path.exists(path):
        return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()

FONT_HEADER = load_font(ARIAL_BOLD_PATH, 72)
FONT_SUB    = load_font(ARIAL_BOLD_PATH, 27)
FONT_DAY    = load_font(ARIAL_BOLD_PATH, 32)
FONT_REGION = load_font(ARIAL_BOLD_PATH, 23)
FONT_TIME   = load_font(ARIAL_REG_PATH, 16)
FONT_EVENT  = load_font(ARIAL_BOLD_PATH, 18)
FONT_LOC    = load_font(ARIAL_REG_PATH, 15)
FONT_SMALL  = load_font(ARIAL_REG_PATH, 14)
FONT_TAG    = load_font(ARIAL_BOLD_PATH, 11)
FONT_WEATHER_TITLE = load_font(ARIAL_BOLD_PATH, 20)
FONT_WEATHER_CITY  = load_font(ARIAL_REG_PATH, 14)
FONT_WEATHER_TEMP  = load_font(ARIAL_BOLD_PATH, 21)
FONT_WEATHER_DESC  = load_font(ARIAL_REG_PATH, 12)
FONT_CTA = load_font(ARIAL_REG_PATH, 20)

# SpottID sponsor callout fonts
FONT_SPON_LABEL = load_font(ARIAL_BOLD_PATH, 12)
FONT_SPON_WORD  = load_font(ARIAL_BOLD_PATH, 20)
FONT_SPON_TAG   = load_font(ARIAL_REG_PATH, 13)
FONT_SPON_STORE = load_font(ARIAL_BOLD_PATH, 11)
FONT_SPON_CTA   = load_font(ARIAL_BOLD_PATH, 13)


# ----------------------------
# ICONS
# ----------------------------
if USE_PNG_ICONS:
    for url, path in [(CLOCK_ICON_URL, CLOCK_ICON_PATH), (PIN_ICON_URL, PIN_ICON_PATH)]:
        if not os.path.exists(path):
            try:
                download_file(url, path)
            except Exception:
                USE_PNG_ICONS = False

def load_icon(path: str, size_px: int) -> Optional[Image.Image]:
    if not os.path.exists(path):
        return None
    return Image.open(path).convert("RGBA").resize((size_px, size_px), Image.LANCZOS)

CLOCK_ICON = load_icon(CLOCK_ICON_PATH, 16) if USE_PNG_ICONS else None
PIN_ICON   = load_icon(PIN_ICON_PATH, 16) if USE_PNG_ICONS else None


# ----------------------------
# SPOTTID SPONSOR ASSETS (logo + QR)
# ----------------------------
SPOTTID_LOGO_OK = False
try:
    _r = requests.get(SPOTTID_LOGO_URL, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    _r.raise_for_status()
    with open(SPOTTID_LOGO_PATH, "wb") as _f:
        _f.write(_r.content)
    _spot = Image.open(SPOTTID_LOGO_PATH).convert("RGBA")
    round_icon_corners(_spot, 0.20).save(SPOTTID_LOGO_ROUND_PATH, "PNG")
    SPOTTID_LOGO_OK = True
except Exception as _e:
    print("[WARN] SpottID logo unavailable, using text wordmark only:", _e)

def _make_store_qr(url, path):
    _q = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, box_size=2, border=4)
    _q.add_data(url)
    _q.make(fit=True)
    # box_size=2 + a proper 4-module quiet zone -> crisp and scannable, and small
    # enough to sit above the sub-headline. It is pasted at native size (never
    # fractionally resized), which keeps the module grid sharp so it decodes.
    _q.make_image(fill_color="black", back_color="white").save(path)

SPOTTID_QR_OK = False
try:
    _make_store_qr(SPOTTID_APPLE_URL, SPOTTID_QR_APPLE_PATH)
    _make_store_qr(SPOTTID_PLAY_URL,  SPOTTID_QR_PLAY_PATH)
    SPOTTID_QR_OK = True
except Exception as _e:
    print("[WARN] SpottID QR generation failed:", _e)


# ----------------------------
# LOAD SHEET
# ----------------------------
def load_events_df(sheets_svc, spreadsheet_id: str, sheet_range: str) -> pd.DataFrame:
    resp = sheets_svc.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id, range=sheet_range
    ).execute()
    rows = resp.get("values", []) or []
    if not rows:
        return pd.DataFrame()
    header = rows[0]
    width = len(header)
    data_rows = [r + [""] * (width - len(r)) for r in rows[1:]]
    return pd.DataFrame(data_rows, columns=header)

raw_df = load_events_df(sheets_service, SPREADSHEET_ID, SHEET_RANGE)
print("Raw shape:", raw_df.shape)
print("RAW COLUMNS:")
for c in raw_df.columns:
    print(repr(c))


# ----------------------------
# PARSE EVENTS
# ----------------------------
ISO_HINT_RE = re.compile(r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}")

def looks_like_datetime(val) -> bool:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return False
    sv = str(val).strip()
    if not sv:
        return False
    if ISO_HINT_RE.search(sv):
        return True
    if re.search(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", sv) and re.search(r"\b\d{1,2}:\d{2}\b", sv):
        return True
    return False

def score_datetime_col(series: pd.Series, sample_n: int = 25) -> int:
    hits = 0
    for v in series.head(sample_n).tolist():
        if looks_like_datetime(v):
            hits += 1
    return hits

def autodetect_datetime_cols(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    cols = list(df.columns)
    norm = {c: normalize_colname(c) for c in cols}
    start_candidates, end_candidates = [], []
    for c, nc in norm.items():
        if any(k in nc for k in ["start datetime","start date time","dtstart","start_at","start at","begins datetime"]):
            start_candidates.append(c)
        if any(k in nc for k in ["end datetime","end date time","dtend","end_at","end at","until datetime","finish datetime"]):
            end_candidates.append(c)

    start_best = max(start_candidates, key=lambda c: score_datetime_col(df[c])) if start_candidates else None
    end_best   = max(end_candidates, key=lambda c: score_datetime_col(df[c])) if end_candidates else None

    if start_best is None or end_best is None:
        scored = [(score_datetime_col(df[c]), c) for c in cols]
        scored = [(s, c) for s, c in scored if s > 0]
        scored.sort(reverse=True)
        if scored:
            if start_best is None:
                start_best = scored[0][1]
            if end_best is None:
                for _, c in scored:
                    if c != start_best:
                        end_best = c
                        break
    return start_best, end_best

def build_events(df: pd.DataFrame) -> List[Event]:
    if df is None or df.empty:
        return []

    cols = list(df.columns)

    col_title = get_exact_or_fuzzy_column(df, ["Title", "Event", "Event Name", "Event Title"], ["event name","event title","title","name","event","summary","what"])
    col_loc   = get_exact_or_fuzzy_column(df, ["Location", "Venue"], ["venue","location","place","where","venue name"])
    col_addr  = get_exact_or_fuzzy_column(df, ["Address"], ["address","addr","street","full address","venue address"])
    col_url   = get_exact_or_fuzzy_column(df, ["Link", "URL"], ["link","url","website","rsvp","event link","registration","tickets"])

    col_date       = get_exact_or_fuzzy_column(df, ["Date", "Event Date"], ["event date","date","day"])
    col_start_time = get_exact_or_fuzzy_column(df, ["Start Time"], ["start time","starttime","begin time","begins time","from time"])
    col_end_time   = get_exact_or_fuzzy_column(df, ["End Time"], ["end time","endtime","finish time","to time","until time"])

    col_start_dt = get_exact_or_fuzzy_column(df, ["Start Datetime", "Start DateTime"], ["start datetime","start date time","dtstart","start at","start_at"])
    col_end_dt   = get_exact_or_fuzzy_column(df, ["End Datetime", "End DateTime"], ["end datetime","end date time","dtend","end at","end_at"])

    col_closest_city = find_column_exact(df, "Closest City")
    col_callout = find_column_exact(df, "Callout")

    if col_closest_city is None:
        raise ValueError(f"Could not find exact column 'Closest City'. Actual headers: {[repr(c) for c in df.columns]}")
    if col_callout is None:
        raise ValueError(f"Could not find exact column 'Callout'. Actual headers: {[repr(c) for c in df.columns]}")

    print("\nResolved columns:")
    print("title:", repr(col_title))
    print("location:", repr(col_loc))
    print("address:", repr(col_addr))
    print("url:", repr(col_url))
    print("date:", repr(col_date))
    print("start_time:", repr(col_start_time))
    print("end_time:", repr(col_end_time))
    print("start_dt:", repr(col_start_dt))
    print("end_dt:", repr(col_end_dt))
    print("closest_city:", repr(col_closest_city))
    print("callout:", repr(col_callout))

    auto_start, auto_end = autodetect_datetime_cols(df)
    if col_start_dt is None and auto_start is not None and score_datetime_col(df[auto_start]) > 0:
        col_start_dt = auto_start
    if col_end_dt is None and auto_end is not None and score_datetime_col(df[auto_end]) > 0:
        col_end_dt = auto_end

    if col_start_dt is not None and score_datetime_col(df[col_start_dt]) == 0:
        if col_start_time is None:
            col_start_time = col_start_dt
        col_start_dt = None
    if col_end_dt is not None and score_datetime_col(df[col_end_dt]) == 0:
        if col_end_time is None:
            col_end_time = col_end_dt
        col_end_dt = None

    out = []
    for _, row in df.iterrows():
        title = clean_title(row.get(col_title) if col_title else None)
        loc   = clean_place(coalesce_str(row.get(col_loc) if col_loc else None))
        addr  = clean_place(coalesce_str(row.get(col_addr) if col_addr else None))
        url   = coalesce_str(row.get(col_url) if col_url else None)

        sheet_closest_city = clean_place(coalesce_str(row.get(col_closest_city)))
        sheet_callout = normalize_spaces(coalesce_str(row.get(col_callout)))

        start_dt = parse_dt_safe(row.get(col_start_dt)) if col_start_dt else None
        end_dt   = parse_dt_safe(row.get(col_end_dt)) if col_end_dt else None

        d = None
        st = None
        en = None

        if start_dt:
            d = start_dt.date()
            st = start_dt.time()
        if end_dt:
            en = end_dt.time()

        if d is None and col_date:
            d = parse_date_only(row.get(col_date))
        if st is None and col_start_time:
            st = parse_time_only(row.get(col_start_time))
        if en is None and col_end_time:
            en = parse_time_only(row.get(col_end_time))

        if d is None:
            continue
        if not title and not loc and not addr:
            continue

        out.append(Event(
            title=title if title else "Untitled event",
            date=d,
            start=st,
            end=en,
            location=loc,
            address=addr,
            url=url,
            source_row={k: row.get(k) for k in cols},
            sheet_closest_city=sheet_closest_city,
            sheet_callout=sheet_callout
        ))
    return out


# ----------------------------
# DEDUPE / PLACE FIXES
# ----------------------------
def normalize_for_dedupe(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[\"'`]", "", s)
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\b(official|annual|season opener|opener)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def smart_place(location: str, address: str) -> str:
    loc = clean_place(location)
    addr = clean_place(address)

    if not loc and not addr:
        return "Location TBA"
    if loc and not addr:
        return loc
    if addr and not loc:
        return addr

    loc_l = loc.lower()
    addr_l = addr.lower()

    if addr_l in loc_l:
        return loc
    if loc_l in addr_l:
        return addr

    return f"{loc}, {addr}"

def dedupe_and_merge_events(events: List[Event]) -> List[Event]:
    if not events:
        return []

    events = sorted(events, key=lambda e: (e.date, time_sort_key(e.start), e.title.lower()))
    used = [False] * len(events)
    merged = []

    for i, e in enumerate(events):
        if used[i]:
            continue
        cluster = [e]
        used[i] = True

        for j in range(i + 1, len(events)):
            if used[j]:
                continue
            other = events[j]
            if e.date != other.date:
                continue

            title_score = fuzz.token_set_ratio(normalize_for_dedupe(e.title), normalize_for_dedupe(other.title))
            place_score = fuzz.token_set_ratio(
                normalize_for_dedupe(smart_place(e.location, e.address)),
                normalize_for_dedupe(smart_place(other.location, other.address))
            )

            time_close = True
            if e.start and other.start:
                m1 = e.start.hour * 60 + e.start.minute
                m2 = other.start.hour * 60 + other.start.minute
                time_close = abs(m1 - m2) <= 90

            if title_score >= 88 and (place_score >= 70 or time_close):
                cluster.append(other)
                used[j] = True

        best = sorted(
            cluster,
            key=lambda x: (
                len(x.address or ""),
                len(x.location or ""),
                1 if x.start else 0,
                len(x.url or ""),
                len(x.title or "")
            ),
            reverse=True
        )[0]

        if len(cluster) > 1:
            titles = [c.title for c in cluster if c.title]
            best.title = max(titles, key=len) if titles else best.title
            if not best.location:
                for c in cluster:
                    if c.location:
                        best.location = c.location
                        break
            if not best.address:
                for c in cluster:
                    if c.address:
                        best.address = c.address
                        break
            if not best.url:
                for c in cluster:
                    if c.url:
                        best.url = c.url
                        break
            if best.start is None:
                for c in cluster:
                    if c.start:
                        best.start = c.start
                        break
            if best.end is None:
                for c in cluster:
                    if c.end:
                        best.end = c.end
                        break
            if not best.sheet_closest_city:
                for c in cluster:
                    if c.sheet_closest_city:
                        best.sheet_closest_city = c.sheet_closest_city
                        break
            if not best.sheet_callout:
                for c in cluster:
                    if c.sheet_callout:
                        best.sheet_callout = c.sheet_callout
                        break

        merged.append(best)

    return merged


# ----------------------------
# ENRICH EVENTS
# ----------------------------
def enrich_events(events: List[Event]) -> List[Event]:
    out = []
    for e in events:
        if e.sheet_closest_city:
            e.city_group = e.sheet_closest_city
            e.closest_city_source = "sheet"
            sheet_city, sheet_state = parse_city_state_text(e.sheet_closest_city)
            e.city = sheet_city
            e.state = sheet_state
        else:
            e.city, e.state = infer_city_state_from_address(e.address, e.location)
            fallback_group = ", ".join([x for x in [e.city, e.state] if x]).strip(", ")
            e.city_group = fallback_group if fallback_group else "Other"
            e.closest_city_source = "auto"

        key = maybe_city_key(e.city, e.state)
        e.coords = CITY_COORDS.get(key)

        if e.coords is None:
            inferred_city, inferred_state = infer_city_state_from_address(e.address, e.location)
            inferred_key = maybe_city_key(inferred_city, inferred_state)
            e.coords = CITY_COORDS.get(inferred_key)

        e.region = infer_region_from_city_state(e.city, e.state, e.coords)

        # Still computed: it is what MAX_DRIVE_MINUTES filters on and what the
        # city sections are ordered by. It is simply never printed anymore.
        if e.coords:
            mins = fallback_drive_minutes(ORIGIN_COORDS, e.coords)
            e.drive_minutes = None if (e.region == "Cincinnati Core" or mins <= 20) else mins
        else:
            e.drive_minutes = None

        e.callout = e.sheet_callout.strip()
        e.callout_source = "sheet" if e.callout else "blank"

        full_place = smart_place(e.location, e.address)
        e.display_place_caption = full_place
        e.display_place_image = full_place

        if len(e.title) > IMAGE_TITLE_MAX_CHARS + 12:
            e.title = truncate_text(e.title, IMAGE_TITLE_MAX_CHARS)

        out.append(e)
    return out


# ----------------------------
# WEATHER
# ----------------------------
WEATHER_CODE_MAP = [
    ({0}, "Sunny"),
    ({1, 2}, "Partly Cloudy"),
    ({3}, "Cloudy"),
    ({45, 48}, "Fog"),
    ({51, 53, 55, 56, 57}, "Light Drizzle"),
    ({61, 63, 65, 66, 67}, "Rain"),
    ({71, 73, 75, 77}, "Snow"),
    ({80, 81, 82}, "Showers"),
    ({85, 86}, "Snow Showers"),
    ({95, 96, 99}, "Storms"),
]

REGION_WEATHER_CITY = {
    "Cincinnati Core": ("Cincinnati", 39.1031, -84.5120),
    "North / Dayton": ("Dayton", 39.7589, -84.1916),
    "Columbus": ("Columbus", 39.9612, -82.9988),
    "Lexington": ("Lexington", 38.0406, -84.5037),
    "Louisville": ("Louisville", 38.2527, -85.7585),
    "Indiana": ("Indianapolis", 39.7684, -86.1581),
}

def map_weather_code(code: Optional[int]) -> str:
    if code is None:
        return "Weather"
    for codes, label in WEATHER_CODE_MAP:
        if code in codes:
            return label
    return "Weather"

def fetch_city_weather_summary(name: str, lat: float, lon: float, dates: List[dt.date]) -> Dict[str, str]:
    try:
        start = min(dates)
        end = max(dates)
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat,
                "longitude": lon,
                "timezone": TZ_NAME,
                "temperature_unit": "fahrenheit",
                "daily": "temperature_2m_min,temperature_2m_max,weathercode",
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
            },
            timeout=30
        )
        if r.status_code != 200:
            return {"city": name, "temp": "--", "desc": "Unavailable"}

        daily = r.json().get("daily", {})
        mins = daily.get("temperature_2m_min", [])
        maxs = daily.get("temperature_2m_max", [])
        codes = daily.get("weathercode", [])

        if not mins or not maxs:
            return {"city": name, "temp": "--", "desc": "Unavailable"}

        low = int(round(min(mins)))
        high = int(round(max(maxs)))
        desc = map_weather_code(codes[0] if codes else None)
        return {"city": name, "temp": f"{low}°/{high}°", "desc": desc}
    except Exception:
        return {"city": name, "temp": "--", "desc": "Unavailable"}

def build_regional_weather(regions_present: List[str], dates: List[dt.date]) -> List[Dict[str, str]]:
    uniq_regions = [r for r in REGION_PRIORITY if r in set(regions_present)]
    if not uniq_regions:
        uniq_regions = ["Cincinnati Core", "Columbus", "Louisville"]

    cards = []
    for r in uniq_regions[:3]:
        city_name, lat, lon = REGION_WEATHER_CITY.get(r, ("Cincinnati", 39.1031, -84.5120))
        cards.append(fetch_city_weather_summary(city_name, lat, lon, dates))

    while len(cards) < 3:
        fallback = ["Cincinnati Core", "Columbus", "Louisville"][len(cards)]
        city_name, lat, lon = REGION_WEATHER_CITY[fallback]
        cards.append(fetch_city_weather_summary(city_name, lat, lon, dates))

    return cards[:3]


# ----------------------------
# LOAD + BUILD EVENT SETS
# ----------------------------
events_all = dedupe_and_merge_events(build_events(raw_df))
events_all = enrich_events(events_all)

print("\nFIRST 15 EVENT AUDIT AFTER READ:")
for e in events_all[:15]:
    print(
        "title=", repr(e.title),
        "| sheet_closest_city=", repr(e.sheet_closest_city),
        "| city_group=", repr(e.city_group),
        "| sheet_callout=", repr(e.sheet_callout),
        "| callout=", repr(e.callout),
    )

events_selected = [e for e in events_all if e.date in selected_set]
events_selected = dedupe_and_merge_events(events_selected)
events_selected = enrich_events(events_selected)

events_selected = [
    e for e in events_selected
    if (e.drive_minutes is None or e.drive_minutes <= MAX_DRIVE_MINUTES)
]

events_by_date_city_full: Dict[dt.date, Dict[str, List[Event]]] = defaultdict(lambda: defaultdict(list))
for e in events_selected:
    group_name = e.city_group if e.city_group else "Other"
    events_by_date_city_full[e.date][group_name].append(e)

for d in selected_dates:
    ordered_groups = {}
    for city_group in sorted(events_by_date_city_full[d].keys(), key=city_group_sort_key):
        ordered_groups[city_group] = sorted(events_by_date_city_full[d][city_group], key=lambda e: time_sort_key(e.start))
    events_by_date_city_full[d] = ordered_groups

regions_present_full = [e.region for e in events_selected]
weather_cards = build_regional_weather(regions_present_full, selected_dates)


# ----------------------------
# BUILD BLOCKS FOR ALL EVENTS
# ----------------------------
blocks = []

for d in selected_dates:
    blocks.append({"type": "day_header", "text": day_header_text(d)})
    day_groups = events_by_date_city_full.get(d, {})
    if not day_groups:
        blocks.append({"type": "note", "text": NO_EVENTS_LINE})
        continue

    for city_group in sorted(day_groups.keys(), key=city_group_sort_key):
        city_events = day_groups[city_group]
        if not city_events:
            continue

        blocks.append({"type": "city_header", "text": city_group})

        for e in city_events:
            blocks.append({
                "type": "event",
                "callout": e.callout,
                "title": e.title,
                "time": format_time_range(e.start, e.end),
                "place": e.display_place_image,
            })


# ----------------------------
# IMAGE RENDER HELPERS
# ----------------------------
def render_callout_badge(draw, x, y, callout, font):
    if not callout:
        return 0, 0
    fill = callout_fill_color(callout)
    pad_x, pad_y = 8, 4
    bb = draw.textbbox((0, 0), callout, font=font)
    w = (bb[2] - bb[0]) + pad_x * 2
    h = (bb[3] - bb[1]) + pad_y * 2
    draw.rounded_rectangle([x, y, x + w, y + h], radius=8, fill=fill)
    draw.text((x + pad_x, y + pad_y - 1), callout, font=font, fill=BG)
    return w, h

def draw_degree(draw, x, y, font, color):
    """Font-independent degree mark drawn as a small ring; returns advance width."""
    s = font.size
    r = max(2, int(round(s * 0.11)))
    w = max(1, int(round(s * 0.07)))
    cx = x + r + 1
    cy = y + int(round(s * 0.20))
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=w)
    return 2 * r + 4

def draw_temp(draw, x, y, temp_str, font, color):
    """Render a temperature like '64°/82°' using hand-drawn degree rings, so it
    never turns into a tofu box when the resolved font lacks a degree glyph."""
    for i, part in enumerate((temp_str or "").replace("°", "").split("/")):
        if i > 0:
            draw.text((x, y), "/", font=font, fill=color)
            x += int(draw.textlength("/", font=font))
        draw.text((x, y), part, font=font, fill=color)
        x += int(draw.textlength(part, font=font))
        if part.strip() not in {"", "--"}:
            x += draw_degree(draw, x, y, font, color)
    return x

def draw_qr_labeled(img, draw, qr_im, label, cx, top, card_inner):
    """White rounded card holding a QR (pasted at native size) with a store label."""
    pad = 6
    card = card_inner + 2 * pad
    x0 = cx - card // 2
    draw.rounded_rectangle([x0, top, x0 + card, top + card], radius=9, fill=WHITE)
    qw, qh = qr_im.size
    img.alpha_composite(qr_im, (x0 + (card - qw) // 2, top + (card - qh) // 2))
    lb = draw.textbbox((0, 0), label, font=FONT_SPON_STORE)
    draw.text((cx - (lb[2] - lb[0]) // 2, top + card + 3), label, font=FONT_SPON_STORE, fill=GRAY)

def draw_spottid_sponsor(img, draw, x_left, x_right, y_top):
    """Centered sponsor callout between the title and the Apex logo: an
    [icon | SPONSORED BY / SpottID] lockup and the tagline centered over two
    store QR codes (App Store + Google Play), with a 'Download the app now' CTA
    below them. Everything stays above the sub-headline so nothing overlaps."""
    span_l, span_r = x_left, x_right
    if span_r - span_l < 200:
        return

    have_qr = (SPOTTID_QR_OK
               and os.path.exists(SPOTTID_QR_APPLE_PATH)
               and os.path.exists(SPOTTID_QR_PLAY_PATH))
    if have_qr:
        qr_apple = Image.open(SPOTTID_QR_APPLE_PATH).convert("RGBA")
        qr_play  = Image.open(SPOTTID_QR_PLAY_PATH).convert("RGBA")
        card_inner = max(qr_apple.size[0], qr_play.size[0])
    else:
        card_inner = 82

    pad = 6
    card = card_inner + 2 * pad
    gap = 16
    block_w = card * 2 + gap

    # center the whole block between the title and the Apex logo, clamped so it
    # always fits inside the available span
    half = block_w // 2
    cx = max(span_l + half, min((span_l + span_r) // 2, span_r - half))

    y = y_top
    # --- brand lockup [icon | SPONSORED BY / SpottID], centered on cx ---
    icon_h = 36
    text_w = max(draw.textlength(SPONSOR_LABEL, font=FONT_SPON_LABEL),
                 draw.textlength(SPOTTID_WORDMARK, font=FONT_SPON_WORD))
    has_icon = SPOTTID_LOGO_OK and os.path.exists(SPOTTID_LOGO_ROUND_PATH)
    lock_w = (icon_h + 8 + text_w) if has_icon else text_w
    lx = int(cx - lock_w // 2)
    if has_icon:
        icon = Image.open(SPOTTID_LOGO_ROUND_PATH).convert("RGBA").resize((icon_h, icon_h), Image.LANCZOS)
        img.alpha_composite(icon, (lx, y))
        tx0 = lx + icon_h + 8
    else:
        tx0 = lx
    draw.text((tx0, y + 1), SPONSOR_LABEL, font=FONT_SPON_LABEL, fill=MIDGRAY)
    draw.text((tx0, y + FONT_SPON_LABEL.size + 4), SPOTTID_WORDMARK, font=FONT_SPON_WORD, fill=WHITE)
    y += icon_h + 3

    # --- tagline centered ---
    tw = draw.textlength(SPOTTID_TAGLINE_MAIN, font=FONT_SPON_TAG)
    draw.text((int(cx - tw // 2), y), SPOTTID_TAGLINE_MAIN, font=FONT_SPON_TAG, fill=GRAY)
    y += FONT_SPON_TAG.size + 4

    if not have_qr:
        return

    # --- two store QR codes, centered ---
    startx = cx - block_w // 2
    c1 = startx + card // 2
    c2 = startx + card + gap + card // 2
    draw_qr_labeled(img, draw, qr_apple, "App Store", c1, y, card_inner)
    draw_qr_labeled(img, draw, qr_play, "Google Play", c2, y, card_inner)
    y += card + 3 + FONT_SPON_STORE.size + 3

    # --- CTA centered under the QRs ---
    ctw = draw.textlength(SPOTTID_CTA, font=FONT_SPON_CTA)
    draw.text((int(cx - ctw // 2), y), SPOTTID_CTA, font=FONT_SPON_CTA, fill=YELLOW)

def measure_block_height(draw, block, max_w):
    if block["type"] == "day_header":
        return FONT_DAY.size + 12
    if block["type"] == "city_header":
        return FONT_REGION.size + 10
    if block["type"] == "note":
        lines = clamp_lines(wrap_text(draw, block["text"], FONT_LOC, max_w), 2)
        return len(lines) * (FONT_LOC.size + 4) + 12

    title_lines = clamp_lines(wrap_text(draw, block["title"], FONT_EVENT, max_w), 2)
    place_lines = clamp_lines(wrap_text(draw, block["place"], FONT_LOC, max_w - 24), 2)

    badge_h = 0
    if block["callout"]:
        bb = draw.textbbox((0, 0), block["callout"], font=FONT_TAG)
        badge_h = (bb[3] - bb[1]) + 8 + 5

    h = 0
    h += badge_h
    h += len(title_lines) * (FONT_EVENT.size + 3)
    h += FONT_TIME.size + 4
    h += len(place_lines) * (FONT_LOC.size + 4)
    h += 12
    return h

def make_base_canvas(page_num: int, total_pages: int) -> Tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGBA", (CANVAS_W, CANVAS_H), BG)
    draw = ImageDraw.Draw(img)

    draw.line([(LEFT_COL_X, TOPLINE_Y), (CANVAS_W - SAFE_PAD, TOPLINE_Y)], fill=TOPLINE, width=3)
    draw.text((LEFT_COL_X, HEADER_Y), "PULL UP OR", font=FONT_HEADER, fill=YELLOW)
    draw.text((LEFT_COL_X, HEADER_Y + HEADER_LINE2_OFFSET), "PARK IT", font=FONT_HEADER, fill=YELLOW)
    draw.text((LEFT_COL_X, SUB_Y), "THE APEX REGIONAL WEEKEND GUIDE", font=FONT_SUB, fill=WHITE)
    draw.line([(LEFT_COL_X, DIVIDER_Y), (CANVAS_W - SAFE_PAD, DIVIDER_Y)], fill=YELLOW, width=4)

    logo_src = LOGO_WHITE_PATH if (LOGO_FORCE_WHITE and os.path.exists(LOGO_WHITE_PATH)) else LOGO_PATH
    if os.path.exists(logo_src):
        logo = Image.open(logo_src).convert("RGBA")
        lw, lh = logo.size
        scale = min(LOGO_MAX_W / lw, LOGO_MAX_H / lh, 1.0)
        ns = (max(1, int(lw * scale)), max(1, int(lh * scale)))
        logo_r = logo.resize(ns, Image.LANCZOS)
        logo_pos = (
            LOGO_X + (LOGO_MAX_W - ns[0]) // 2,
            LOGO_Y + (LOGO_MAX_H - ns[1]) // 2
        )
        img.alpha_composite(logo_r, logo_pos)

    # --- SpottID sponsor callout: left of the Apex logo, above the sub-headline ---
    _r1 = draw.textbbox((LEFT_COL_X, HEADER_Y), "PULL UP OR", font=FONT_HEADER)[2]
    _r2 = draw.textbbox((LEFT_COL_X, HEADER_Y + HEADER_LINE2_OFFSET), "PARK IT", font=FONT_HEADER)[2]
    _title_right = max(_r1, _r2)
    draw_spottid_sponsor(img, draw, int(_title_right) + 24, LOGO_X - 14, 24)

    page_label = f"Slide {page_num} of {total_pages}"
    bb = draw.textbbox((0, 0), page_label, font=FONT_SMALL)
    px = CANVAS_W - SAFE_PAD - (bb[2] - bb[0])
    py = SUB_Y + 2
    draw.text((px, py), page_label, font=FONT_SMALL, fill=MIDGRAY)

    draw.rounded_rectangle([WEATHER_X, WEATHER_Y, WEATHER_X + WEATHER_W, WEATHER_Y + WEATHER_H], radius=18, outline=YELLOW, width=4)
    draw.text((WEATHER_X + 16, WEATHER_Y + 10), "Regional Weather Outlook", font=FONT_WEATHER_TITLE, fill=WHITE)
    draw.line([(WEATHER_X + 16, WEATHER_Y + 40), (WEATHER_X + WEATHER_W - 16, WEATHER_Y + 40)], fill=TOPLINE, width=2)

    col_w = (WEATHER_W - 30) // 3
    x = WEATHER_X + 12
    for card in weather_cards:
        draw.text((x, WEATHER_Y + 50), card["city"], font=FONT_WEATHER_CITY, fill=GRAY)
        draw_temp(draw, x, WEATHER_Y + 72, card["temp"], FONT_WEATHER_TEMP, WHITE)
        draw.text((x, WEATHER_Y + 104), card["desc"], font=FONT_WEATHER_DESC, fill=MIDGRAY)
        x += col_w

    cta_lines = clamp_lines(
      wrap_text(draw, CTA_TEXT, FONT_CTA, CANVAS_W - 120),
      2
    )

    yy = CTA_Y - len(cta_lines) * (FONT_CTA.size + 2)

    for ln in cta_lines:
      bb = draw.textbbox((0, 0), ln, font=FONT_CTA)
      xx = (CANVAS_W - (bb[2] - bb[0])) // 2

      # subtle shadow for readability
      draw.text((xx + 1, yy + 1), ln, font=FONT_CTA, fill=(40, 40, 40))

      # main text
      draw.text((xx, yy), ln, font=FONT_CTA, fill=(215, 215, 215))

      yy += FONT_CTA.size + 2

    return img, draw

def draw_icon_line(img, draw, x0, y0, icon, fallback, text, font, color, max_lines, max_w):
    if not text:
        return y0
    if icon is not None:
        img.alpha_composite(icon, (x0, y0 + 1))
        tx = x0 + 24
    else:
        draw.text((x0, y0), fallback, font=font, fill=color)
        tx = x0 + 20
    lines = clamp_lines(wrap_text(draw, text, font, max_w - (tx - x0)), max_lines)
    line_h = font.size + 4
    for i, ln in enumerate(lines):
        draw.text((tx, y0 + i * line_h), ln, font=font, fill=color)
    return y0 + len(lines) * line_h

def draw_event_block(img, draw, x0, y, block, max_w):
    if block["callout"]:
        _, badge_h = render_callout_badge(draw, x0, y, block["callout"], FONT_TAG)
        y += badge_h + 5

    title_lines = clamp_lines(wrap_text(draw, block["title"], FONT_EVENT, max_w), 2)
    for ln in title_lines:
        draw.text((x0, y), ln, font=FONT_EVENT, fill=WHITE)
        y += FONT_EVENT.size + 3

    y = draw_icon_line(img, draw, x0, y, CLOCK_ICON, "\U0001f553", block["time"], FONT_TIME, GRAY, 1, max_w)
    y = draw_icon_line(img, draw, x0, y + 2, PIN_ICON, "\U0001f4cd", block["place"], FONT_LOC, GRAY, 2, max_w)
    y += 12
    return y

def draw_blocks_on_column(img, draw, block_list, start_idx, x0, y0, max_y, max_w):
    y = y0
    i = start_idx

    while i < len(block_list):
        b = block_list[i]

        if b["type"] == "day_header":
            needed = measure_block_height(draw, b, max_w)
            if i + 1 < len(block_list):
                needed += measure_block_height(draw, block_list[i + 1], max_w)
            if y + needed >= max_y:
                break

            draw.text((x0, y), b["text"], font=FONT_DAY, fill=WHITE)
            y += FONT_DAY.size + 10
            i += 1
            continue

        if b["type"] == "city_header":
            needed = measure_block_height(draw, b, max_w)
            next_event_height = 0
            if i + 1 < len(block_list) and block_list[i + 1]["type"] == "event":
                next_event_height = measure_block_height(draw, block_list[i + 1], max_w)

            if y + needed + next_event_height >= max_y:
                break

            draw.text((x0, y), b["text"], font=FONT_REGION, fill=YELLOW)
            underline = min(max_w, draw.textbbox((0, 0), b["text"], font=FONT_REGION)[2] + 18)
            draw.line([(x0, y + FONT_REGION.size + 3), (x0 + underline, y + FONT_REGION.size + 3)], fill=DARKGRAY, width=1)
            y += FONT_REGION.size + 8
            i += 1
            continue

        if b["type"] == "note":
            needed = measure_block_height(draw, b, max_w)
            if y + needed >= max_y:
                break

            lines = clamp_lines(wrap_text(draw, b["text"], FONT_LOC, max_w), 2)
            for ln in lines:
                draw.text((x0, y), ln, font=FONT_LOC, fill=MIDGRAY)
                y += FONT_LOC.size + 4
            y += 12
            i += 1
            continue

        needed = measure_block_height(draw, b, max_w)
        if y + needed >= max_y:
            break

        y = draw_event_block(img, draw, x0, y, b, max_w)
        i += 1

    return i, y

def paginate_blocks(block_list: List[Dict[str, str]]) -> List[Tuple[int, int]]:
    probe_img = Image.new("RGBA", (CANVAS_W, CANVAS_H), BG)
    probe_draw = ImageDraw.Draw(probe_img)

    pages = []
    idx = 0

    while idx < len(block_list):
        start_idx = idx

        idx, _ = draw_blocks_on_column(
            probe_img,
            probe_draw,
            block_list,
            idx,
            LEFT_COL_X,
            CONTENT_Y,
            LEFT_CONTENT_MAX_Y,
            LEFT_W
        )

        idx, _ = draw_blocks_on_column(
            probe_img,
            probe_draw,
            block_list,
            idx,
            RIGHT_COL_X,
            CONTENT_Y,
            RIGHT_CONTENT_MAX_Y,
            RIGHT_W
        )

        if idx == start_idx:
            raise RuntimeError("A block is too tall to fit on a blank slide. Reduce font size or event block height.")

        pages.append((start_idx, idx))

    return pages

def render_carousel_images(block_list: List[Dict[str, str]], base_path_no_ext: str) -> List[str]:
    page_ranges = paginate_blocks(block_list)
    total_pages = len(page_ranges)
    output_paths = []

    for page_num, (start_idx, end_idx) in enumerate(page_ranges, start=1):
        img, draw = make_base_canvas(page_num, total_pages)

        idx, _ = draw_blocks_on_column(
            img,
            draw,
            block_list,
            start_idx,
            LEFT_COL_X,
            CONTENT_Y,
            LEFT_CONTENT_MAX_Y,
            LEFT_W
        )

        idx, _ = draw_blocks_on_column(
            img,
            draw,
            block_list,
            idx,
            RIGHT_COL_X,
            CONTENT_Y,
            RIGHT_CONTENT_MAX_Y,
            RIGHT_W
        )

        final = Image.new("RGB", (CANVAS_W, CANVAS_H), BG)
        final.paste(img, mask=img.split()[-1])

        out_path = f"{base_path_no_ext}_slide_{page_num:02d}.jpg"
        final.save(out_path, "JPEG", quality=95, subsampling=0)
        output_paths.append(out_path)

    return output_paths


# ----------------------------
# TEXT OUTPUT
# ----------------------------
def compact_address(addr: str) -> str:
    addr = clean_place(addr)
    addr = re.sub(r"\s+", " ", addr).strip()
    return addr

def compact_address_short(addr: str) -> str:
    """Street + city only — drop the state/ZIP tail for the caption's
    one-line featured-event address (the slides still carry the full one)."""
    addr = clean_place(addr)
    parts = [p.strip() for p in addr.split(",") if p.strip()]
    return ", ".join(parts[:2]) if len(parts) > 2 else addr

def format_time_compact(start: Optional[dt.time], end: Optional[dt.time]) -> str:
    """Condensed range like '5-8 PM' or '9 AM-1 PM' for the caption; the
    slides use the fuller format_time_range with the Morning/Midday/Evening tag."""
    def h(t: dt.time) -> str:
        h12 = t.hour % 12 or 12
        return f"{h12}:{t.minute:02d}" if t.minute else f"{h12}"
    # Same fallback case as format_time_range: a start/end that match
    # exactly means no real end time was found, not a real instant event.
    if end == start:
        end = None
    if start and end:
        ap_s = "AM" if start.hour < 12 else "PM"
        ap_e = "AM" if end.hour < 12 else "PM"
        if ap_s == ap_e:
            return f"{h(start)}-{h(end)} {ap_e}"
        return f"{h(start)} {ap_s}-{h(end)} {ap_e}"
    if start:
        return f"{h(start)} {'AM' if start.hour < 12 else 'PM'}"
    if end:
        return f"Until {h(end)} {'AM' if end.hour < 12 else 'PM'}"
    return "Time TBA"

def find_featured_event(events: List[Event]) -> Optional[Event]:
    """The one event to spotlight in the caption: whichever row has Callout
    set to exactly 'Featured' in the sheet (a manual weekly pick), earliest
    first if more than one is flagged. None if nobody flagged one this week."""
    featured = [e for e in events if e.callout.strip().upper() == "FEATURED"]
    if not featured:
        return None
    return sorted(featured, key=lambda e: (e.date, time_sort_key(e.start)))[0]

def build_hashtags() -> str:
    return " ".join(BASE_HASHTAGS[:8])

def make_caption(
    dates: List[dt.date],
    events_by_date_city: Dict[dt.date, Dict[str, List[Event]]],
    events_selected: List[Event],
) -> str:
    """Short social-post caption: market summary, one spotlighted event, a
    call-out for any fully dead day, the SpottID pitch/CTA, and hashtags. The
    exhaustive per-event breakdown lives on the carousel slides now, not here."""
    out: List[str] = []

    seen_regions = set()
    for e in events_selected:
        seen_regions.add(e.region)
    ordered_regions = [r for r in REGION_PRIORITY if r in seen_regions and r != "Other"]
    market_names = [REGION_DISPLAY_NAME.get(r, r) for r in ordered_regions]

    n = len(market_names)
    if n:
        count_word = NUMBER_WORDS.get(n, str(n))
        market_word = "market" if n == 1 else "markets"
        out.append(f"{count_word} {market_word}. One weekend. {', '.join(market_names)} — all loaded.")
    else:
        out.append("This week's regional lineup is here.")
    out.append("")
    out.append(
        "Cars & coffee, cruise-ins, and full shows from Friday morning through "
        "Thursday night. Whatever you drive, there's a spot for it within an "
        "hour of you. Swipe for the full regional lineup — times, addresses, "
        "weather, all of it."
    )
    out.append("")

    featured = find_featured_event(events_selected)
    if featured:
        out.append(f"🔥 {featured.date.strftime('%A').upper()} IS THE ONE 🔥")
        out.append(f"{featured.title} — {format_time_compact(featured.start, featured.end)}")
        out.append(compact_address_short(featured.display_place_caption))
        out.append("")

    out.append(
        "SpottID is rolling in. Come find us, scan a ride, claim your car, and "
        "watch your build land where people are actually looking for it. Bring "
        "the good stuff. We're tagging it."
    )
    out.append("")

    dead_days = [d for d in dates if not any(events_by_date_city.get(d, {}).values())]
    if dead_days and len(dead_days) < len(dates):
        names = [d.strftime("%A") for d in dead_days]
        if len(names) == 1:
            joined = names[0]
        elif len(names) == 2:
            joined = f"{names[0]} and {names[1]}"
        else:
            joined = ", ".join(names[:-1]) + f", and {names[-1]}"
        out.append(f"{joined}? Nothing posted. Go drive something anyway.")
        out.append("")

    cta_day = f" before {featured.date.strftime('%A')}" if featured else ""
    out.append(f"📱 Get SpottID{cta_day}:")
    out.append(f"Claim Your Car → {SPOTTID_CLAIM_URL.replace('https://', '')}")
    out.append(f"iOS → {SPOTTID_APPLE_URL.replace('https://', '')}")
    out.append(f"Android → {SPOTTID_PLAY_URL.replace('https://', '')}")
    out.append("")

    out.append("Want your event in next week's lineup?")
    out.append("admin@ApexAutoLounge.com")
    out.append("")

    out.append(build_hashtags())

    cleaned = []
    prev_blank = False
    for line in out:
        blank = (line.strip() == "")
        if blank and prev_blank:
            continue
        cleaned.append(line.rstrip())
        prev_blank = blank

    return "\n".join(cleaned).strip()

# ----------------------------
# OUTPUTS
# ----------------------------
run_date = dt.datetime.now(pytz.timezone(TZ_NAME)).date().isoformat()

out_base_no_ext = os.path.join(OUTPUT_DIR, f"apex_regional_lineup_{run_date}")
out_caption_path = os.path.join(OUTPUT_DIR, f"apex_regional_lineup_{run_date}_caption.txt")

for p in glob.glob(f"{out_base_no_ext}_slide_*.jpg"):
    try:
        os.remove(p)
    except Exception:
        pass

slide_paths = render_carousel_images(blocks, out_base_no_ext)

caption = make_caption(selected_dates, events_by_date_city_full, events_selected)
with open(out_caption_path, "w", encoding="utf-8") as f:
    f.write(caption)

# Fixed names, not date-stamped: the service account can only overwrite an
# existing file (no create quota of its own), so these must keep matching
# whatever was pre-seeded once under the real account in DRIVE_OUTPUT_FOLDER_ID.
DRIVE_SLIDE_NAME_PREFIX = "apex_weekly_lineup_slide"
DRIVE_CAPTION_NAME = "apex_weekly_lineup_caption.txt"
DRIVE_SEEDED_SLIDE_COUNT = 6

slide_links = []
txt_link = ""
try:
    if len(slide_paths) > DRIVE_SEEDED_SLIDE_COUNT:
        print(f"[WARN] {len(slide_paths)} slides this week, but only "
              f"{DRIVE_SEEDED_SLIDE_COUNT} pre-seeded Drive filenames exist — "
              f"the extra ones will fail to upload until more are seeded.")
    for i, p in enumerate(slide_paths, start=1):
        drive_name = f"{DRIVE_SLIDE_NAME_PREFIX}_{i:02d}.jpg"
        slide_links.append(upload_to_folder(drive_service, TARGET_FOLDER_ID, p, drive_filename=drive_name))
    txt_link = upload_to_folder(drive_service, TARGET_FOLDER_ID, out_caption_path, drive_filename=DRIVE_CAPTION_NAME)
    trash_unused_drive_slides(drive_service, TARGET_FOLDER_ID, len(slide_paths), DRIVE_SEEDED_SLIDE_COUNT, DRIVE_SLIDE_NAME_PREFIX)
except Exception as ex:
    print(f"[WARN] Drive upload failed partway through: {ex}")
    print("[WARN] Outputs are still local (see the 'apex-weekly-lineup' GitHub Actions artifact).")

# Drive time is not shown on the slides/caption. These lines stay in the log
# only, because drive time still decides the city order and still drops
# anything farther than MAX_DRIVE_MINUTES.
print("\nDistance filter origin ZIP (not printed on slides):", ORIGIN_LABEL)
print("Origin coordinates used:", ORIGIN_COORDS)
print("Max allowed drive time:", format_drive_time(MAX_DRIVE_MINUTES))
print("Total events included:", len(events_selected))
print("Total carousel slides:", len(slide_paths))

print("\nFINAL AUDIT:")
for e in sorted(events_selected, key=lambda x: (x.date, city_group_sort_key(x.city_group), time_sort_key(x.start), x.title.lower())):
    print(
        f"{e.date} | {e.title} | "
        f"sheet_closest_city={repr(e.sheet_closest_city)} -> city_group={repr(e.city_group)} [{e.closest_city_source}] | "
        f"sheet_callout={repr(e.sheet_callout)} -> callout={repr(e.callout)} [{e.callout_source}] | "
        f"region={repr(e.region)} | drive={repr(format_drive_time(e.drive_minutes) if e.drive_minutes else '')}"
    )

print("\nLocal outputs:")
for p in slide_paths:
    print("Slide:", p)
print("Caption:", out_caption_path)

if slide_links or txt_link:
    print("\nUploaded to Drive:")
    for i, link in enumerate(slide_links, start=1):
        print(f"Slide {i} link:", link)
    if txt_link:
        print("Caption link:", txt_link)

print("\n" + "="*48 + "\nCAPTION PREVIEW\n" + "="*48 + "\n")
print(caption)
