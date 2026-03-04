#!/usr/bin/env python3
"""Verify and normalize Apex event times in a Google Sheet tab."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup
from dateutil import parser as dtparser
from google.oauth2 import service_account
from googleapiclient.discovery import build

ET_TZ = ZoneInfo("America/New_York")
TIMEZONE_LABEL = "America/New_York"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

TIME_RANGE_RE = re.compile(
    r"(?P<start>\d{1,2}(?::\d{2})?\s*(?:[AaPp]\.?[Mm]\.?)?)\s*(?:-|–|—|to)\s*"
    r"(?P<end>\d{1,2}(?::\d{2})?\s*(?:[AaPp]\.?[Mm]\.?)?)"
)
DATE_RE = re.compile(
    r"(?:"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}"
    r"|\d{1,2}/\d{1,2}/\d{2,4}"
    r")",
    re.I,
)


@dataclass
class ParseResult:
    start_dt: Optional[datetime]
    end_dt: Optional[datetime]
    source: Optional[str]
    note: str = ""


@dataclass
class RowContext:
    link: str
    sheet_start: Optional[datetime]
    sheet_end: Optional[datetime]
    current_start_time: str
    current_end_time: str


@dataclass
class RowOutcome:
    date: str
    start_time: str
    end_time: str
    tz: str
    time_source: str
    time_verified: str
    verify_notes: str


def extract_spreadsheet_id(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    if "docs.google.com/spreadsheets" in raw:
        m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", raw)
        if m:
            return m.group(1)
    return raw


def extract_gid(value: str) -> Optional[int]:
    m = re.search(r"[?#&]gid=(\d+)", value or "")
    return int(m.group(1)) if m else None


def get_google_credentials() -> service_account.Credentials:
    service_account_path = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")
    service_account_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON") or os.getenv("GDRIVE_SERVICE_ACCOUNT_JSON")
    scopes = [
        "https://www.googleapis.com/auth/drive",
        "https://www.googleapis.com/auth/spreadsheets",
    ]
    if service_account_path:
        return service_account.Credentials.from_service_account_file(service_account_path, scopes=scopes)
    if service_account_json:
        return service_account.Credentials.from_service_account_info(json.loads(service_account_json), scopes=scopes)
    raise RuntimeError("Missing GOOGLE_SERVICE_ACCOUNT_FILE or GOOGLE_SERVICE_ACCOUNT_JSON/GDRIVE_SERVICE_ACCOUNT_JSON")


def norm_header(h: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (h or "").strip().lower())


def col_to_a1(idx0: int) -> str:
    n = idx0 + 1
    out = ""
    while n:
        n, r = divmod(n - 1, 26)
        out = chr(65 + r) + out
    return out


def parse_iso(raw: str) -> Optional[datetime]:
    text = (raw or "").strip()
    if not text:
        return None
    try:
        dt = dtparser.isoparse(text)
    except Exception:
        try:
            dt = dtparser.parse(text)
        except Exception:
            return None
    return ensure_tz(dt)


def parse_clock_time(raw: str) -> Optional[Tuple[int, int]]:
    text = (raw or "").strip()
    if not text:
        return None
    try:
        dt = dtparser.parse(text, fuzzy=True)
    except Exception:
        return None
    return dt.hour, dt.minute


def ensure_tz(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=ET_TZ)
    return dt


def to_et(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    return ensure_tz(dt).astimezone(ET_TZ)


def to_date_str(dt: Optional[datetime]) -> str:
    et = to_et(dt)
    return et.strftime("%Y-%m-%d") if et else ""


def to_time_str(dt: Optional[datetime]) -> str:
    et = to_et(dt)
    return et.strftime("%I:%M %p").lstrip("0") if et else ""


def collect_jsonld_event_objs(payload: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            yield from collect_jsonld_event_objs(item)
        return
    if not isinstance(payload, dict):
        return

    typ = payload.get("@type")
    types = typ if isinstance(typ, list) else [typ]
    normalized = [str(t).strip().lower() for t in types if t is not None]
    if any("event" in t for t in normalized):
        yield payload

    for key in ("@graph", "mainEntity", "itemListElement", "subEvent"):
        if key in payload:
            yield from collect_jsonld_event_objs(payload.get(key))


def parse_dt_value(value: str) -> Optional[datetime]:
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        dt = dtparser.isoparse(raw)
    except Exception:
        try:
            dt = dtparser.parse(raw, fuzzy=True)
        except Exception:
            return None
    return ensure_tz(dt)


def parse_jsonld(soup: BeautifulSoup) -> ParseResult:
    scripts = soup.find_all("script", attrs={"type": re.compile("ld\\+json", re.I)})
    for script in scripts:
        raw = (script.string or script.get_text() or "").strip()
        if not raw:
            continue
        payloads: List[Any] = []
        try:
            payloads.append(json.loads(raw))
        except Exception:
            pass

        if not payloads:
            for m in re.finditer(r"\{[\s\S]*?\}", raw):
                try:
                    payloads.append(json.loads(m.group(0)))
                except Exception:
                    continue

        for payload in payloads:
            for event_obj in collect_jsonld_event_objs(payload):
                start_dt = parse_dt_value(str(event_obj.get("startDate", "")))
                end_dt = parse_dt_value(str(event_obj.get("endDate", "")))
                if start_dt:
                    return ParseResult(start_dt=start_dt, end_dt=end_dt, source="page_jsonld", note="jsonld")
    return ParseResult(None, None, None)


def parse_time_tags(soup: BeautifulSoup) -> ParseResult:
    start_dt: Optional[datetime] = None
    end_dt: Optional[datetime] = None
    for tag in soup.find_all("time"):
        raw = (tag.get("datetime") or tag.get_text(" ", strip=True) or "").strip()
        cand = parse_dt_value(raw)
        if not cand:
            continue

        attrs = " ".join([tag.get("itemprop", ""), tag.get("id", ""), " ".join(tag.get("class", []))]).lower()
        if any(tok in attrs for tok in ("start", "from")):
            start_dt = cand
        elif any(tok in attrs for tok in ("end", "to")):
            end_dt = cand
        elif start_dt is None:
            start_dt = cand
        elif end_dt is None:
            end_dt = cand

        if start_dt and end_dt:
            break

    if start_dt:
        return ParseResult(start_dt=start_dt, end_dt=end_dt, source="page_time_tag", note="time_tag")
    return ParseResult(None, None, None)


def parse_meta_tags(soup: BeautifulSoup) -> ParseResult:
    start_dt: Optional[datetime] = None
    end_dt: Optional[datetime] = None
    for tag in soup.find_all("meta"):
        key = " ".join([tag.get("property", ""), tag.get("name", ""), tag.get("itemprop", "")]).lower()
        content = (tag.get("content") or "").strip()
        if not content:
            continue

        if any(tok in key for tok in ("startdate", "event:start", "start_time", "starttime", "eventstart")):
            start_dt = start_dt or parse_dt_value(content)
        if any(tok in key for tok in ("enddate", "event:end", "end_time", "endtime", "eventend")):
            end_dt = end_dt or parse_dt_value(content)

    if start_dt:
        return ParseResult(start_dt=start_dt, end_dt=end_dt, source="page_meta", note="meta")
    return ParseResult(None, None, None)


def _normalize_ampm_token(token: str) -> str:
    t = token.strip().lower().replace(".", "")
    if re.search(r"\b\d{1,2}\s*[ap]m\b", t):
        t = re.sub(r"\b(\d{1,2})\s*([ap])m\b", r"\1:00 \2m", t, flags=re.I)
    return t


def parse_visible_text(soup: BeautifulSoup, fallback_start: Optional[datetime]) -> ParseResult:
    text = soup.get_text(" ", strip=True)
    date_match = DATE_RE.search(text)

    base_date: Optional[datetime] = None
    if date_match:
        try:
            base_date = dtparser.parse(date_match.group(0), fuzzy=True)
            base_date = ensure_tz(base_date)
        except Exception:
            base_date = None

    if base_date is None and fallback_start:
        base_date = to_et(fallback_start)

    if base_date is None:
        return ParseResult(None, None, None)

    tr = TIME_RANGE_RE.search(text)
    if not tr:
        return ParseResult(None, None, None)

    start_raw = _normalize_ampm_token(tr.group("start"))
    end_raw = _normalize_ampm_token(tr.group("end"))

    try:
        start_t = dtparser.parse(start_raw, fuzzy=True)
        end_t = dtparser.parse(end_raw, fuzzy=True)
    except Exception:
        return ParseResult(None, None, None)

    start_dt = datetime(base_date.year, base_date.month, base_date.day, start_t.hour, start_t.minute, tzinfo=base_date.tzinfo)
    end_dt = datetime(base_date.year, base_date.month, base_date.day, end_t.hour, end_t.minute, tzinfo=base_date.tzinfo)
    if end_dt <= start_dt:
        end_dt += timedelta(days=1)

    return ParseResult(start_dt=start_dt, end_dt=end_dt, source="page_text", note="text")


def fetch_with_retries(session: requests.Session, url: str, retries: int = 2, timeout: int = 15) -> Optional[str]:
    for attempt in range(retries + 1):
        try:
            resp = session.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp.text
        except Exception:
            if attempt >= retries:
                return None
    return None


def parse_page(url: str, fallback_start: Optional[datetime], session: requests.Session) -> ParseResult:
    html = fetch_with_retries(session=session, url=url, retries=2, timeout=15)
    if html is None:
        return ParseResult(None, None, None, "fetch_failed")

    soup = BeautifulSoup(html, "html.parser")

    for parser_fn in (parse_jsonld, parse_time_tags, parse_meta_tags):
        parsed = parser_fn(soup)
        if parsed.start_dt:
            return parsed

    parsed = parse_visible_text(soup, fallback_start=fallback_start)
    if parsed.start_dt:
        return parsed

    return ParseResult(None, None, None, "parse_failed")


def locate_headers(headers: List[str]) -> Dict[str, int]:
    nmap = {norm_header(h): i for i, h in enumerate(headers)}

    def pick(*cands: str) -> int:
        for cand in cands:
            if cand in nmap:
                return nmap[cand]
        return -1

    return {
        "title": pick("title", "eventtitle", "name"),
        "start_iso": pick("startiso", "start", "startdatetime", "iso"),
        "end_iso": pick("endiso", "end", "enddatetime"),
        "date": pick("date", "eventdate"),
        "start_time": pick("starttime", "time", "eventstarttime"),
        "end_time": pick("endtime", "eventendtime"),
        "link": pick("link", "url", "eventurl", "eventlink"),
        "location": pick("location", "venue", "address"),
        "categ": pick("categ", "category", "type"),
    }


def ensure_columns(headers: List[str], needed: List[str]) -> List[str]:
    existing = {norm_header(h) for h in headers}
    for col in needed:
        if norm_header(col) not in existing:
            headers.append(col)
            existing.add(norm_header(col))
    return headers


def duration_hours(start: Optional[datetime], end: Optional[datetime]) -> Optional[float]:
    if not (start and end):
        return None
    delta = to_et(end) - to_et(start)
    return delta.total_seconds() / 3600.0


def detect_default_pairs(rows: List[RowContext]) -> Counter:
    pair_counts: Counter = Counter()
    midnight_counts: Counter = Counter()
    for row in rows:
        st = (row.current_start_time or "").strip().upper()
        et = (row.current_end_time or "").strip().upper()
        if st and et:
            pair_counts[(st, et)] += 1
        if row.sheet_start and to_et(row.sheet_start).time().hour == 0 and to_et(row.sheet_start).time().minute == 0:
            midnight_counts[to_et(row.sheet_start).strftime("%H:%M:%S")] += 1
    return pair_counts + midnight_counts


def is_suspect_row(row: RowContext, repeated_default_pairs: Counter, repeated_midnight_iso: Counter) -> Tuple[bool, str]:
    start_et = to_et(row.sheet_start)
    end_et = to_et(row.sheet_end)

    # Existing exported times blank/unparseable.
    if not parse_clock_time(row.current_start_time) or (row.current_end_time and not parse_clock_time(row.current_end_time)):
        return True, "current_time_blank_or_unparseable"

    start_label = (row.current_start_time or "").strip().upper()
    end_label = (row.current_end_time or "").strip().upper()
    if start_label == "12:00 AM" and end_label == "2:00 AM":
        return True, "default_pair_1200_0200"

    if repeated_default_pairs[(start_label, end_label)] >= 6 and start_label and end_label:
        return True, "repeated_default_pair_detected"

    if row.sheet_start is None:
        return True, "start_iso_missing_or_unparseable"

    if start_et and start_et.hour == 0 and start_et.minute == 0 and start_et.second == 0:
        return True, "start_iso_midnight"

    if start_et and repeated_midnight_iso[start_et.strftime("%H:%M:%S")] >= 10:
        return True, "repeated_default_start_iso_time"

    if row.sheet_end is None:
        return True, "end_iso_missing"

    if start_et and end_et and end_et <= start_et:
        return True, "end_iso_le_start_iso"

    dur = duration_hours(row.sheet_start, row.sheet_end)
    if dur is not None and dur > 12.0:
        return True, "duration_gt_12h"

    return False, ""


def apply_page_result(
    parsed: ParseResult,
    sheet_start: Optional[datetime],
    sheet_end: Optional[datetime],
) -> Tuple[Optional[datetime], Optional[datetime], str, str, str, bool, bool]:
    """Return (start, end, source, verified, note, mismatch, parse_failed)."""
    if not parsed.start_dt:
        return sheet_start, sheet_end, "fallback_sheet", "FALLBACK", parsed.note or "parse_failed", False, True

    page_start = to_et(parsed.start_dt)
    page_end = to_et(parsed.end_dt) if parsed.end_dt else None

    if page_end is None and sheet_end and to_et(sheet_end) > page_start:
        final_end = to_et(sheet_end)
    else:
        final_end = page_end

    source = parsed.source or "fallback_sheet"
    verified = "YES"
    note = parsed.note or "page_time_parsed"
    mismatch = False

    if sheet_start:
        start_diff = abs((page_start - to_et(sheet_start)).total_seconds())
        end_diff = 0.0
        if final_end and sheet_end:
            end_diff = abs((final_end - to_et(sheet_end)).total_seconds())
        if start_diff > 300 or end_diff > 300:
            verified = "NO"
            note = "mismatch_sheet_vs_page"
            mismatch = True

    return page_start, final_end, source, verified, note, mismatch, False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--spreadsheet",
        default="https://docs.google.com/spreadsheets/d/1lVpqhmUOQDZywjGeYxgm7ILNXqP3l6Z74pVXw1oKSQ8/edit?gid=1268351023",
    )
    ap.add_argument("--tab", default="")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    spreadsheet_id = extract_spreadsheet_id(args.spreadsheet)
    gid = extract_gid(args.spreadsheet)

    creds = get_google_credentials()
    sheets = build("sheets", "v4", credentials=creds)

    meta = sheets.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    sheet_props = meta.get("sheets", [])

    tab_name = args.tab
    if not tab_name:
        if gid is not None:
            for s in sheet_props:
                if s.get("properties", {}).get("sheetId") == gid:
                    tab_name = s["properties"]["title"]
                    break
        if not tab_name and sheet_props:
            tab_name = sheet_props[0]["properties"]["title"]

    values_resp = sheets.spreadsheets().values().get(spreadsheetId=spreadsheet_id, range=tab_name).execute()
    rows: List[List[str]] = values_resp.get("values", [])
    if not rows:
        print("No rows found.")
        return 0

    headers = rows[0][:]
    idx = locate_headers(headers)
    for required in ("start_iso", "end_iso", "link"):
        if idx[required] < 0:
            raise RuntimeError(f"Required header not found: {required}")

    headers = ensure_columns(headers, ["date", "start_time", "end_time", "tz", "time_source", "time_verified", "verify_notes"])
    header_positions = {norm_header(h): i for i, h in enumerate(headers)}

    data_rows = rows[1:]
    for row in data_rows:
        if len(row) < len(headers):
            row.extend([""] * (len(headers) - len(row)))

    row_ctxs: List[RowContext] = []
    for row in data_rows:
        row_ctxs.append(
            RowContext(
                link=(row[idx["link"]] if idx["link"] < len(row) else "").strip(),
                sheet_start=parse_iso((row[idx["start_iso"]] if idx["start_iso"] < len(row) else "").strip()),
                sheet_end=parse_iso((row[idx["end_iso"]] if idx["end_iso"] < len(row) else "").strip()),
                current_start_time=(row[idx["start_time"]] if idx["start_time"] >= 0 and idx["start_time"] < len(row) else "").strip(),
                current_end_time=(row[idx["end_time"]] if idx["end_time"] >= 0 and idx["end_time"] < len(row) else "").strip(),
            )
        )

    repeated_default_pairs = Counter(
        (r.current_start_time.upper(), r.current_end_time.upper())
        for r in row_ctxs
        if r.current_start_time and r.current_end_time
    )
    repeated_midnight_iso = Counter(
        to_et(r.sheet_start).strftime("%H:%M:%S")
        for r in row_ctxs
        if r.sheet_start and to_et(r.sheet_start).hour == 0 and to_et(r.sheet_start).minute == 0 and to_et(r.sheet_start).second == 0
    )

    session = requests.Session()
    session.headers.update({"User-Agent": UA})
    adapter = requests.adapters.HTTPAdapter(max_retries=2, pool_connections=20, pool_maxsize=20)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    stats = Counter()
    time_source_counts = Counter()

    suspects: List[Tuple[int, RowContext, str]] = []
    for i, ctx in enumerate(row_ctxs):
        stats["total_rows"] += 1
        suspect, reason = is_suspect_row(ctx, repeated_default_pairs, repeated_midnight_iso)
        if suspect:
            suspects.append((i, ctx, reason or "suspect"))
            stats["suspect_rows"] += 1

    # Fetch suspect links once each.
    unique_jobs: Dict[str, Tuple[Optional[datetime], str]] = {}
    for _, ctx, reason in suspects:
        if ctx.link and ctx.link not in unique_jobs:
            unique_jobs[ctx.link] = (ctx.sheet_start, reason)

    cache: Dict[str, ParseResult] = {}
    if unique_jobs:
        with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 6))) as pool:
            futs = {
                pool.submit(parse_page, url, fallback_start, session): url
                for url, (fallback_start, _) in unique_jobs.items()
            }
            for fut in as_completed(futs):
                url = futs[fut]
                try:
                    cache[url] = fut.result()
                except Exception:
                    cache[url] = ParseResult(None, None, None, "fetch_failed")
                stats["links_fetched"] += 1

    stats["cache_hits"] = max(0, stats["suspect_rows"] - stats["links_fetched"])

    outcomes: List[RowOutcome] = []
    suspect_reason_by_index = {i: reason for i, _, reason in suspects}

    for i, ctx in enumerate(row_ctxs):
        reason = suspect_reason_by_index.get(i, "")

        if i not in suspect_reason_by_index:
            final_start = to_et(ctx.sheet_start)
            final_end = to_et(ctx.sheet_end)
            source = "sheet_iso"
            verified = "YES"
            note = "sheet_iso_used"
        else:
            if not ctx.link:
                final_start = to_et(ctx.sheet_start)
                final_end = to_et(ctx.sheet_end)
                source = "fallback_sheet"
                verified = "FALLBACK"
                note = f"{reason};missing_link"
                stats["fallback_count"] += 1
                stats["parse_failures"] += 1
            else:
                parsed = cache.get(ctx.link, ParseResult(None, None, None, "fetch_failed"))
                final_start, final_end, source, verified, detail_note, mismatch, parse_failed = apply_page_result(
                    parsed=parsed,
                    sheet_start=ctx.sheet_start,
                    sheet_end=ctx.sheet_end,
                )
                note = reason
                if detail_note:
                    note = f"{note};{detail_note}" if note else detail_note
                if mismatch:
                    stats["mismatch_count"] += 1
                if parse_failed:
                    stats["fallback_count"] += 1
                    stats["parse_failures"] += 1
                    if parsed.note == "fetch_failed":
                        note = f"{reason};fetch_failed"

        outcomes.append(
            RowOutcome(
                date=to_date_str(final_start),
                start_time=to_time_str(final_start),
                end_time=to_time_str(final_end),
                tz=TIMEZONE_LABEL,
                time_source=source,
                time_verified=verified,
                verify_notes=note,
            )
        )
        time_source_counts[source] += 1

    last_row = len(data_rows) + 1
    write_data = [
        {"range": f"{tab_name}!A1:{col_to_a1(len(headers)-1)}1", "values": [headers]},
        {
            "range": f"{tab_name}!{col_to_a1(header_positions['date'])}2:{col_to_a1(header_positions['date'])}{last_row}",
            "values": [[o.date] for o in outcomes],
        },
        {
            "range": f"{tab_name}!{col_to_a1(header_positions['starttime'])}2:{col_to_a1(header_positions['starttime'])}{last_row}",
            "values": [[o.start_time] for o in outcomes],
        },
        {
            "range": f"{tab_name}!{col_to_a1(header_positions['endtime'])}2:{col_to_a1(header_positions['endtime'])}{last_row}",
            "values": [[o.end_time] for o in outcomes],
        },
        {
            "range": f"{tab_name}!{col_to_a1(header_positions['tz'])}2:{col_to_a1(header_positions['tz'])}{last_row}",
            "values": [[o.tz] for o in outcomes],
        },
        {
            "range": f"{tab_name}!{col_to_a1(header_positions['timesource'])}2:{col_to_a1(header_positions['timesource'])}{last_row}",
            "values": [[o.time_source] for o in outcomes],
        },
        {
            "range": f"{tab_name}!{col_to_a1(header_positions['timeverified'])}2:{col_to_a1(header_positions['timeverified'])}{last_row}",
            "values": [[o.time_verified] for o in outcomes],
        },
        {
            "range": f"{tab_name}!{col_to_a1(header_positions['verifynotes'])}2:{col_to_a1(header_positions['verifynotes'])}{last_row}",
            "values": [[o.verify_notes] for o in outcomes],
        },
    ]

    sheets.spreadsheets().values().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body={"valueInputOption": "RAW", "data": write_data},
    ).execute()

    print("Event time normalization complete")
    print(f"total rows processed: {stats['total_rows']}")
    print(f"suspect rows count: {stats['suspect_rows']}")
    print(f"links fetched: {stats['links_fetched']}")
    print(f"cache hits: {stats['cache_hits']}")
    print(f"time_source breakdown counts: {dict(time_source_counts)}")
    print(f"fallback count: {stats['fallback_count']}")
    print(f"mismatch count: {stats['mismatch_count']}")
    print(f"parse failures count: {stats['parse_failures']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
