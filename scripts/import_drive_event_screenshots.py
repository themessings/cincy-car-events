#!/usr/bin/env python3
"""
Import event details from screenshot images in a Google Drive folder.

Setup
-----
1) Install Python dependencies (minimum):
   pip install google-api-python-client google-auth google-auth-httplib2 \
       google-auth-oauthlib python-dateutil dateparser pandas pillow pytesseract

2) Optional OCR backends:
   - Preferred (higher accuracy): Google Cloud Vision
     pip install google-cloud-vision
   - Fallback: Tesseract OCR binary + pytesseract
     * Ubuntu/Debian: sudo apt-get install tesseract-ocr
     * macOS (brew): brew install tesseract
     * Windows: install from UB Mannheim builds and add to PATH

3) Authentication options for Google Drive/Sheets APIs:
   A) Service account (recommended for automation)
      export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json
      Make sure the Drive folder / Sheet is shared with this service account.

   B) OAuth user auth fallback
      - Put OAuth client secrets JSON at ./client_secret.json (or set --oauth-client-secrets)
      - First run opens consent flow and stores token in ./token.json

Usage
-----
python scripts/import_drive_event_screenshots.py \
  --folder-id 13ex_jE_1zAtCbBPcgsVNde8vkPTbA7JS \
  [--sheet-id YOUR_SHEET_ID] \
  [--max-files 25] \
  [--dry-run]

Outputs are written to ./output/
- output/events_extracted_raw.csv
- output/events_extracted_deduped.csv
- output/events_extracted.csv (alias of deduped)
- output/review_queue.csv
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import tempfile
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import dateparser
except Exception:
    dateparser = None
try:
    import pytesseract
except Exception:
    pytesseract = None
from dateutil import parser as dateutil_parser
try:
    from PIL import Image, ImageEnhance, ImageOps
except Exception:
    Image = ImageEnhance = ImageOps = None

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google.oauth2.service_account import Credentials as ServiceAccountCredentials
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaIoBaseDownload
except Exception:
    Request = Credentials = ServiceAccountCredentials = build = HttpError = MediaIoBaseDownload = None

try:
    from google.cloud import vision  # type: ignore

    HAS_CLOUD_VISION = True
except Exception:
    HAS_CLOUD_VISION = False

try:
    import pillow_heif  # type: ignore

    pillow_heif.register_heif_opener()
except Exception:
    pass

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
SHEETS_SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
ALL_SCOPES = DRIVE_SCOPES + SHEETS_SCOPES

TIMEZONE = "America/New_York"
DEFAULT_CITY = "Cincinnati"
DEFAULT_STATE = "OH"
OUTPUT_DIR = Path("output")
SUPPORTED_MIME_PREFIXES = ("image/",)
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".heic", ".heif"}


@dataclass
class EventRecord:
    event_name: str = ""
    date: str = ""
    start_time: str = ""
    end_time: str = ""
    timezone: str = TIMEZONE
    venue_name: str = ""
    street_address: str = ""
    city: str = ""
    state: str = ""
    zip: str = ""
    source: str = ""
    source_file_name: str = ""
    source_drive_file_id: str = ""
    source_drive_link: str = ""
    raw_text: str = ""
    confidence_score: float = 0.0
    needs_review: bool = True




def ensure_runtime_dependencies(dry_run: bool) -> None:
    missing = []
    if dateparser is None:
        missing.append("dateparser")
    if Image is None and not dry_run:
        missing.append("Pillow")
    if build is None or Credentials is None:
        missing.append("google-api-python-client/google-auth")
    if missing:
        raise RuntimeError("Missing required Python packages: " + ", ".join(missing))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract events from Drive screenshots")
    parser.add_argument("--folder-id", default="13ex_jE_1zAtCbBPcgsVNde8vkPTbA7JS")
    parser.add_argument(
        "--sheet-id",
        default=(
            os.getenv("APEX_IMPORT_SPREADSHEET_ID", "").strip()
            or os.getenv("SHEET_ID", "").strip()
        ),
    )
    parser.add_argument("--max-files", type=int, default=0, help="0 means no limit")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--oauth-token", default="token.json")
    parser.add_argument("--oauth-client-secrets", default="client_secret.json")
    parser.add_argument("--source-label", default="google_drive_screenshots")
    return parser.parse_args()


def get_credentials(args: argparse.Namespace) -> Credentials:
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    if creds_path and Path(creds_path).exists():
        # Service account path supplied.
        return ServiceAccountCredentials.from_service_account_file(creds_path, scopes=ALL_SCOPES)

    service_account_json = (
        os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
        or os.getenv("GDRIVE_SERVICE_ACCOUNT_JSON", "").strip()
    )
    if service_account_json:
        try:
            info = json.loads(service_account_json)
            return ServiceAccountCredentials.from_service_account_info(info, scopes=ALL_SCOPES)
        except Exception as exc:
            raise RuntimeError(f"Invalid GOOGLE_SERVICE_ACCOUNT_JSON/GDRIVE_SERVICE_ACCOUNT_JSON: {exc}")

    # OAuth user fallback.
    token_path = Path(args.oauth_token)
    creds: Optional[Credentials] = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), ALL_SCOPES)

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())

    if creds and creds.valid:
        return creds

    client_secret = Path(args.oauth_client_secrets)
    if not client_secret.exists():
        raise RuntimeError(
            "No service-account credentials found and OAuth client secrets missing. "
            "Set GOOGLE_APPLICATION_CREDENTIALS or provide --oauth-client-secrets."
        )

    from google_auth_oauthlib.flow import InstalledAppFlow

    flow = InstalledAppFlow.from_client_secrets_file(str(client_secret), ALL_SCOPES)
    creds = flow.run_local_server(port=0)
    token_path.write_text(creds.to_json(), encoding="utf-8")
    return creds


def drive_service(creds: Credentials):
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def sheets_service(creds: Credentials):
    return build("sheets", "v4", credentials=creds, cache_discovery=False)


def list_files_recursive(drive, folder_id: str, max_files: int = 0) -> List[Dict[str, str]]:
    queue = [folder_id]
    images: List[Dict[str, str]] = []

    while queue:
        current = queue.pop(0)
        page_token = None
        while True:
            resp = (
                drive.files()
                .list(
                    q=f"'{current}' in parents and trashed=false",
                    fields="nextPageToken, files(id,name,mimeType,webViewLink,modifiedTime)",
                    pageToken=page_token,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute()
            )
            files = resp.get("files", [])
            for f in files:
                mime = f.get("mimeType", "")
                ext = Path(f.get("name", "")).suffix.lower()
                if mime == "application/vnd.google-apps.folder":
                    queue.append(f["id"])
                elif mime.startswith(SUPPORTED_MIME_PREFIXES) or ext in SUPPORTED_EXTS:
                    images.append(f)
                    if max_files and len(images) >= max_files:
                        return images
            page_token = resp.get("nextPageToken")
            if not page_token:
                break
    return images


def preprocess_image(path: Path) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")

    w, h = img.size
    if min(w, h) < 1200:
        img = img.resize((w * 2, h * 2), Image.Resampling.LANCZOS)

    img = ImageEnhance.Contrast(img).enhance(1.35)
    img = ImageEnhance.Sharpness(img).enhance(1.25)
    gray = ImageOps.grayscale(img)
    bw = gray.point(lambda x: 0 if x < 170 else 255, mode="1")
    return bw.convert("RGB")


def extract_text_cloud_vision(image_bytes: bytes) -> str:
    if not HAS_CLOUD_VISION:
        return ""
    try:
        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=image_bytes)
        response = client.document_text_detection(image=image)
        if response.error.message:
            return ""
        if response.full_text_annotation and response.full_text_annotation.text:
            return response.full_text_annotation.text
        if response.text_annotations:
            return response.text_annotations[0].description
    except Exception:
        return ""
    return ""


def extract_text_tesseract(image: Image.Image) -> str:
    if pytesseract is None:
        raise RuntimeError("pytesseract is not installed. Install with: pip install pytesseract")
    return pytesseract.image_to_string(image, config="--oem 3 --psm 6")


def normalize_ws(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def normalize_event_name(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^a-zA-Z0-9\s]", " ", value.lower())
    return normalize_ws(value)


def parse_time_token(token: str) -> Optional[str]:
    token = token.strip().lower().replace(".", "")
    token = token.replace(" ", "")
    match = re.match(r"^(\d{1,2})(?::(\d{2}))?(am|pm)?$", token)
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    ampm = match.group(3)
    if ampm == "pm" and hour < 12:
        hour += 12
    if ampm == "am" and hour == 12:
        hour = 0
    if hour > 23 or minute > 59:
        return None
    return f"{hour:02d}:{minute:02d}"


def extract_date(text: str) -> Optional[datetime]:
    now = datetime.now()
    year_match = re.search(r"\b(20\d{2})\b", text)
    preferred_year = int(year_match.group(1)) if year_match else now.year

    patterns = [
        r"\b(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun),?\s+[A-Za-z]{3,9}\s+\d{1,2}(?:,?\s*\d{2,4})?\b",
        r"\b[A-Za-z]{3,9}\s+\d{1,2}(?:,?\s*\d{2,4})?\b",
        r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
        r"\b\d{1,2}-\d{1,2}-\d{2,4}\b",
    ]

    for pattern in patterns:
        for candidate in re.findall(pattern, text, flags=re.IGNORECASE):
            if dateparser is None:
                continue
            dt = dateparser.parse(
                candidate,
                settings={
                    "PREFER_DATES_FROM": "future",
                    "RELATIVE_BASE": now,
                    "DATE_ORDER": "MDY",
                },
            )
            if dt:
                if not re.search(r"\b\d{4}\b", candidate):
                    dt = dt.replace(year=preferred_year)
                return dt

    try:
        dt = dateutil_parser.parse(text, fuzzy=True, default=now.replace(month=1, day=1))
        return dt.replace(year=preferred_year if dt.year == now.year else dt.year)
    except Exception:
        return None


def extract_times(text: str) -> Tuple[str, str]:
    compact = text.replace("\u2013", "-").replace("\u2014", "-")
    range_match = re.search(
        r"(\d{1,2}(?::\d{2})?\s?(?:am|pm)?)\s*(?:-|to|until)\s*(\d{1,2}(?::\d{2})?\s?(?:am|pm)?)",
        compact,
        flags=re.IGNORECASE,
    )
    if range_match:
        s = parse_time_token(range_match.group(1))
        e = parse_time_token(range_match.group(2))
        return s or "", e or ""

    single = re.search(r"\b(\d{1,2}(?::\d{2})?\s?(?:am|pm))\b", compact, flags=re.IGNORECASE)
    if single:
        s = parse_time_token(single.group(1))
        return s or "", ""

    return "", ""


def extract_address(text: str) -> Tuple[str, str, str, str, str]:
    m = re.search(
        r"(\d{1,6}\s+[\w\s.#-]+?),?\s*([A-Za-z\s]+),?\s*(OH|Ohio)?\s*(\d{5})?",
        text,
        flags=re.IGNORECASE,
    )
    if not m:
        return "", "", "", "", ""

    street = normalize_ws(m.group(1))
    city = normalize_ws(m.group(2)) or DEFAULT_CITY
    state = "OH"
    zip_code = m.group(4) or ""
    normalized_street = re.sub(r"\s+,", ",", street)
    return "", normalized_street, city, state, zip_code


def parse_event_text(text: str, source_name: str, source_id: str, source_link: str, source_label: str) -> EventRecord:
    lines = [normalize_ws(line) for line in text.splitlines() if normalize_ws(line)]
    event_name = lines[0] if lines else ""

    date_obj = extract_date(text)
    date_str = date_obj.strftime("%Y-%m-%d") if date_obj else ""

    start_time, end_time = extract_times(text)
    venue = ""
    street, city, state, zip_code = "", "", "", ""

    if len(lines) > 1:
        for line in lines[1:4]:
            if any(x in line.lower() for x in ["ave", "st", "rd", "blvd", "pike", "dr", "way", "ct"]):
                venue = lines[1] if line != lines[1] else ""
                _, street, city, state, zip_code = extract_address(line)
                break

    if street and not city:
        city = DEFAULT_CITY
    if street and not state:
        state = DEFAULT_STATE

    confidence = 0.1
    confidence += 0.25 if event_name else 0
    confidence += 0.25 if date_str else 0
    confidence += 0.2 if start_time else 0
    confidence += 0.1 if end_time else 0
    confidence += 0.1 if street or venue else 0

    needs_review = False
    if not event_name or not date_str or not start_time:
        needs_review = True
    if not end_time:
        needs_review = True

    if start_time and end_time and date_obj:
        start_dt = datetime.strptime(f"{date_str} {start_time}", "%Y-%m-%d %H:%M")
        end_dt = datetime.strptime(f"{date_str} {end_time}", "%Y-%m-%d %H:%M")
        if end_dt < start_dt:
            end_dt += timedelta(days=1)
            if (end_dt - start_dt) > timedelta(hours=14):
                needs_review = True

    return EventRecord(
        event_name=event_name,
        date=date_str,
        start_time=start_time,
        end_time=end_time,
        timezone=TIMEZONE,
        venue_name=venue,
        street_address=street,
        city=city,
        state=state,
        zip=zip_code,
        source=source_label,
        source_file_name=source_name,
        source_drive_file_id=source_id,
        source_drive_link=source_link,
        raw_text=text,
        confidence_score=round(min(confidence, 1.0), 3),
        needs_review=needs_review,
    )


def dedupe_events(events: List[EventRecord]) -> List[EventRecord]:
    by_key: Dict[str, EventRecord] = {}

    for event in events:
        key = "|".join(
            [
                normalize_event_name(event.event_name),
                event.date,
                event.start_time,
                normalize_event_name(event.venue_name or event.street_address),
            ]
        )
        existing = by_key.get(key)
        if not existing:
            by_key[key] = event
            continue

        primary = event if event.confidence_score > existing.confidence_score else existing
        secondary = existing if primary is event else event

        for field in EventRecord.__dataclass_fields__.keys():
            if not getattr(primary, field) and getattr(secondary, field):
                setattr(primary, field, getattr(secondary, field))
        primary.needs_review = bool(primary.needs_review or secondary.needs_review)
        by_key[key] = primary

    return list(by_key.values())


def download_file(drive, file_id: str, destination: Path) -> None:
    request = drive.files().get_media(fileId=file_id, supportsAllDrives=True)
    with destination.open("wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()


def write_csv(path: Path, events: Sequence[EventRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(EventRecord.__dataclass_fields__.keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for e in events:
            row = asdict(e)
            row["needs_review"] = "TRUE" if e.needs_review else "FALSE"
            writer.writerow(row)


def write_sheet(sheet_service, sheet_id: str, tab_name: str, events: Sequence[EventRecord]) -> None:
    values = [list(EventRecord.__dataclass_fields__.keys())]
    for e in events:
        row = [getattr(e, key) for key in EventRecord.__dataclass_fields__.keys()]
        values.append(row)

    # Ensure tab exists.
    meta = sheet_service.spreadsheets().get(spreadsheetId=sheet_id).execute()
    sheet_titles = {s["properties"]["title"] for s in meta.get("sheets", [])}
    if tab_name not in sheet_titles:
        sheet_service.spreadsheets().batchUpdate(
            spreadsheetId=sheet_id,
            body={"requests": [{"addSheet": {"properties": {"title": tab_name}}}]},
        ).execute()

    rng = f"{tab_name}!A1"
    sheet_service.spreadsheets().values().clear(spreadsheetId=sheet_id, range=tab_name).execute()
    sheet_service.spreadsheets().values().update(
        spreadsheetId=sheet_id,
        range=rng,
        valueInputOption="RAW",
        body={"values": values},
    ).execute()


def sanity_checks() -> None:
    assert parse_time_token("10am") == "10:00"
    assert parse_time_token("12:30 PM") == "12:30"
    s, e = extract_times("Saturday 10am-2pm")
    assert s == "10:00" and e == "14:00"
    d = extract_date("Sat, Mar 7, 2026")
    assert d is not None and d.year == 2026 and d.month == 3 and d.day == 7


def main() -> None:
    args = parse_args()
    ensure_runtime_dependencies(args.dry_run)
    sanity_checks()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        creds = get_credentials(args)
        drive = drive_service(creds)
        sheet = sheets_service(creds) if args.sheet_id else None
    except Exception as exc:
        raise SystemExit(f"Authentication failed: {exc}")

    try:
        files = list_files_recursive(drive, args.folder_id, args.max_files)
    except HttpError as exc:
        raise SystemExit(f"Failed to list folder files: {exc}")

    raw_events: List[EventRecord] = []

    source_label = normalize_ws(f"{args.source_label}:{args.folder_id}")

    with tempfile.TemporaryDirectory(prefix="drive_screenshots_") as tmpdir:
        tmp_path = Path(tmpdir)

        for item in files:
            file_id = item["id"]
            name = item.get("name", file_id)
            local_path = tmp_path / name
            try:
                if not args.dry_run:
                    download_file(drive, file_id, local_path)
                    processed = preprocess_image(local_path)
                    buf = io.BytesIO()
                    processed.save(buf, format="PNG")
                    vision_text = extract_text_cloud_vision(buf.getvalue())
                    text = vision_text or extract_text_tesseract(processed)
                else:
                    text = f"DRY RUN PLACEHOLDER FOR {name}"

                event = parse_event_text(text, name, file_id, item.get("webViewLink", ""), source_label)
                raw_events.append(event)
            except Exception as exc:
                err_event = EventRecord(
                    source=source_label,
                    source_file_name=name,
                    source_drive_file_id=file_id,
                    source_drive_link=item.get("webViewLink", ""),
                    raw_text=f"ERROR: {exc}",
                    needs_review=True,
                    confidence_score=0.0,
                )
                raw_events.append(err_event)

    deduped = dedupe_events(raw_events)
    review = [e for e in deduped if e.needs_review]

    raw_path = OUTPUT_DIR / "events_extracted_raw.csv"
    dedupe_path = OUTPUT_DIR / "events_extracted_deduped.csv"
    final_alias = OUTPUT_DIR / "events_extracted.csv"
    review_path = OUTPUT_DIR / "review_queue.csv"

    write_csv(raw_path, raw_events)
    write_csv(dedupe_path, deduped)
    write_csv(final_alias, deduped)
    write_csv(review_path, review)

    if args.sheet_id and not args.dry_run and sheet is not None:
        try:
            write_sheet(sheet, args.sheet_id, "Extracted Events", deduped)
        except Exception as exc:
            print(f"Warning: failed writing to Google Sheet: {exc}")

    print("=== Extraction Summary ===")
    print(f"Total files scanned: {len(files)}")
    print(f"Total events extracted (raw): {len(raw_events)}")
    print(f"Total after dedupe: {len(deduped)}")
    print(f"Needs review count: {len(review)}")
    print(f"Raw CSV: {raw_path}")
    print(f"Deduped CSV: {dedupe_path}")
    print(f"Review queue: {review_path}")


if __name__ == "__main__":
    main()
