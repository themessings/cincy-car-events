#!/usr/bin/env python3
"""Verify and normalize event times in Google Sheet rows by parsing each event link."""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from dateutil import parser as dtparser
from dateutil import tz
from google.oauth2 import service_account
from googleapiclient.discovery import build

DEFAULT_TZ = tz.gettz("America/New_York")
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
    method: Optional[str]
    note: str


@dataclass
class RowOutcome:
    date: str
    start_time: str
    end_time: str
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
    raw = (raw or "").strip()
    if not raw:
        return None
    try:
        dt = dtparser.isoparse(raw)
    except Exception:
        try:
            dt = dtparser.parse(raw)
        except Exception:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=DEFAULT_TZ)
    return dt


def force_tz(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=DEFAULT_TZ)
    return dt


def to_date_str(dt: Optional[datetime]) -> str:
    if not dt:
        return ""
    return force_tz(dt).astimezone(DEFAULT_TZ).strftime("%Y-%m-%d")


def to_time_str(dt: Optional[datetime]) -> str:
    if not dt:
        return ""
    return force_tz(dt).astimezone(DEFAULT_TZ).strftime("%I:%M %p").lstrip("0")


def collect_jsonld_event_objs(payload: Any) -> List[Dict[str, Any]]:
    found: List[Dict[str, Any]] = []

    def _walk(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                _walk(item)
            return
        if not isinstance(node, dict):
            return

        typ = node.get("@type")
        types = typ if isinstance(typ, list) else [typ]
        normalized = [str(t).strip().lower() for t in types if t is not None]
        if any("event" in t for t in normalized):
            found.append(node)

        for key in ("@graph", "mainEntity", "itemListElement", "subEvent"):
            if key in node:
                _walk(node.get(key))

    _walk(payload)
    return found


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
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=DEFAULT_TZ)
    return dt


def parse_jsonld(soup: BeautifulSoup) -> ParseResult:
    scripts = soup.find_all("script", attrs={"type": re.compile("ld\\+json", re.I)})
    for script in scripts:
        raw = (script.string or script.get_text() or "").strip()
        if not raw:
            continue
        candidates: List[Any] = []
        try:
            candidates.append(json.loads(raw))
        except Exception:
            for m in re.finditer(r"\{[\s\S]*?\}", raw):
                chunk = m.group(0)
                try:
                    candidates.append(json.loads(chunk))
                except Exception:
                    continue

        for payload in candidates:
            for event_obj in collect_jsonld_event_objs(payload):
                start_dt = parse_dt_value(str(event_obj.get("startDate", "")))
                end_dt = parse_dt_value(str(event_obj.get("endDate", "")))
                if start_dt:
                    return ParseResult(start_dt, end_dt, "jsonld", "jsonld")
    return ParseResult(None, None, None, "")


def parse_time_tags(soup: BeautifulSoup) -> ParseResult:
    start_dt: Optional[datetime] = None
    end_dt: Optional[datetime] = None
    for tag in soup.find_all("time"):
        raw = (tag.get("datetime") or tag.get_text(" ", strip=True) or "").strip()
        cand = parse_dt_value(raw)
        if not cand:
            continue
        attrs = " ".join([tag.get("itemprop", ""), tag.get("id", ""), " ".join(tag.get("class", []))]).lower()
        if any(k in attrs for k in ["start", "from"]):
            start_dt = cand
        elif any(k in attrs for k in ["end", "to"]):
            end_dt = cand
        elif not start_dt:
            start_dt = cand
        elif not end_dt:
            end_dt = cand
        if start_dt and end_dt:
            break
    if start_dt:
        return ParseResult(start_dt, end_dt, "time_tag", "time_tag")
    return ParseResult(None, None, None, "")


def parse_meta_tags(soup: BeautifulSoup) -> ParseResult:
    start_dt: Optional[datetime] = None
    end_dt: Optional[datetime] = None
    for tag in soup.find_all("meta"):
        key = " ".join([tag.get("property", ""), tag.get("name", ""), tag.get("itemprop", "")]).lower()
        content = (tag.get("content") or "").strip()
        if not content:
            continue
        if any(k in key for k in ["startdate", "event:start", "start_time", "starttime", "eventstart"]):
            start_dt = start_dt or parse_dt_value(content)
        if any(k in key for k in ["enddate", "event:end", "end_time", "endtime", "eventend"]):
            end_dt = end_dt or parse_dt_value(content)
    if start_dt:
        return ParseResult(start_dt, end_dt, "meta_tag", "meta_tag")
    return ParseResult(None, None, None, "")


def _normalize_ampm_token(token: str) -> str:
    t = token.strip().lower().replace(".", "")
    if re.search(r"\b\d{1,2}\s*[ap]m\b", t):
        return re.sub(r"\b(\d{1,2})\s*([ap])m\b", r"\1:00 \2m", t, flags=re.I)
    return t


def parse_visible_text(soup: BeautifulSoup, fallback_start: Optional[datetime]) -> ParseResult:
    text = soup.get_text(" ", strip=True)
    date_match = DATE_RE.search(text)
    base_date: Optional[datetime] = None

    if date_match:
        try:
            base_date = dtparser.parse(date_match.group(0), fuzzy=True)
        except Exception:
            base_date = None

    if base_date is None and fallback_start is not None:
        base_date = force_tz(fallback_start)

    if base_date is None:
        return ParseResult(None, None, None, "")

    tr = TIME_RANGE_RE.search(text)
    if not tr:
        return ParseResult(None, None, None, "")

    start_raw = _normalize_ampm_token(tr.group("start"))
    end_raw = _normalize_ampm_token(tr.group("end"))

    try:
        start_t = dtparser.parse(start_raw, fuzzy=True)
        end_t = dtparser.parse(end_raw, fuzzy=True)
    except Exception:
        return ParseResult(None, None, None, "")

    tzinfo = base_date.tzinfo or DEFAULT_TZ
    start_dt = datetime(base_date.year, base_date.month, base_date.day, start_t.hour, start_t.minute, tzinfo=tzinfo)
    end_dt = datetime(base_date.year, base_date.month, base_date.day, end_t.hour, end_t.minute, tzinfo=tzinfo)
    if end_dt <= start_dt:
        end_dt += timedelta(days=1)
    return ParseResult(start_dt, end_dt, "page_text", "page_text")


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
    html = fetch_with_retries(session, url=url, retries=2, timeout=15)
    if html is None:
        return ParseResult(None, None, None, "fetch_failed")

    soup = BeautifulSoup(html, "html.parser")

    for parser_fn in (parse_jsonld, parse_time_tags, parse_meta_tags):
        res = parser_fn(soup)
        if res.start_dt:
            return res

    text_res = parse_visible_text(soup, fallback_start=fallback_start)
    if text_res.start_dt:
        return text_res

    return ParseResult(None, None, None, "parse_failed")


def matches_sheet(page_start: Optional[datetime], page_end: Optional[datetime], sheet_start: Optional[datetime], sheet_end: Optional[datetime]) -> bool:
    if not (page_start and page_end and sheet_start and sheet_end):
        return False

    ps = force_tz(page_start).astimezone(DEFAULT_TZ)
    pe = force_tz(page_end).astimezone(DEFAULT_TZ)
    ss = force_tz(sheet_start).astimezone(DEFAULT_TZ)
    se = force_tz(sheet_end).astimezone(DEFAULT_TZ)

    if ps.date() != ss.date() or pe.date() != se.date():
        return False

    tolerance = timedelta(minutes=5)
    return abs(ps - ss) <= tolerance and abs(pe - se) <= tolerance


def locate_headers(headers: List[str]) -> Dict[str, int]:
    nmap = {norm_header(h): i for i, h in enumerate(headers)}

    def pick(*cands: str) -> int:
        for c in cands:
            if c in nmap:
                return nmap[c]
        return -1

    return {
        "title": pick("title", "eventtitle", "name"),
        "start_iso": pick("startiso", "start", "startdatetime", "iso"),
        "end_iso": pick("endiso", "end", "enddatetime"),
        "categ": pick("categ", "category", "type"),
        "miles_from_c": pick("milesfromc", "miles", "distance", "mileage"),
        "location": pick("location", "venue", "address"),
        "link": pick("link", "url", "eventurl", "eventlink"),
    }


def ensure_columns(headers: List[str], needed: List[str]) -> List[str]:
    normalized = {norm_header(h) for h in headers}
    for col in needed:
        if norm_header(col) not in normalized:
            headers.append(col)
            normalized.add(norm_header(col))
    return headers


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spreadsheet", default="https://docs.google.com/spreadsheets/d/1lVpqhmUOQDZywjGeYxgm7ILNXqP3l6Z74pVXw1oKSQ8/edit?gid=1268351023")
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

    headers = ensure_columns(headers, ["date", "start_time", "end_time", "time_verified", "verify_notes"])
    header_positions = {norm_header(h): i for i, h in enumerate(headers)}

    data_rows = rows[1:]
    for r in data_rows:
        if len(r) < len(headers):
            r.extend([""] * (len(headers) - len(r)))

    session = requests.Session()
    session.headers.update({"User-Agent": UA})
    adapter = requests.adapters.HTTPAdapter(max_retries=2, pool_connections=20, pool_maxsize=20)
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    stats = Counter()
    parsed_method_counts = Counter()
    cache: Dict[str, ParseResult] = {}

    jobs: Dict[str, Tuple[Optional[datetime], int]] = {}
    row_parsed_inputs: List[Tuple[str, Optional[datetime], Optional[datetime]]] = []

    for row in data_rows:
        stats["total_rows"] += 1
        link = (row[idx["link"]] if idx["link"] < len(row) else "").strip()
        sheet_start = parse_iso((row[idx["start_iso"]] if idx["start_iso"] < len(row) else "").strip())
        sheet_end = parse_iso((row[idx["end_iso"]] if idx["end_iso"] < len(row) else "").strip())
        row_parsed_inputs.append((link, sheet_start, sheet_end))
        if link and link not in jobs:
            jobs[link] = (sheet_start, len(jobs))

    if jobs:
        with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 6))) as executor:
            future_map = {
                executor.submit(parse_page, url, fb_start, session): (url, order)
                for url, (fb_start, order) in jobs.items()
            }
            for fut in as_completed(future_map):
                url, _ = future_map[fut]
                try:
                    cache[url] = fut.result()
                except Exception:
                    cache[url] = ParseResult(None, None, None, "fetch_failed")
                stats["links_fetched"] += 1

    outcomes: List[RowOutcome] = []
    for link, sheet_start, sheet_end in row_parsed_inputs:
        if link and link in cache:
            stats["cache_hits"] += 1
            parsed = cache[link]
            page_start = parsed.start_dt
            page_end = parsed.end_dt
            if page_start and page_end:
                final_start = page_start
                final_end = page_end
                if matches_sheet(page_start, page_end, sheet_start, sheet_end):
                    time_verified = "YES"
                else:
                    time_verified = "NO"
                    stats["mismatches"] += 1
                verify_notes = parsed.note if time_verified == "YES" else "mismatch_sheet_vs_page"
                if parsed.method:
                    parsed_method_counts[parsed.method] += 1
            else:
                final_start = sheet_start
                final_end = sheet_end
                time_verified = "FALLBACK"
                verify_notes = parsed.note or "parse_failed"
                stats["fallbacks"] += 1
                if parsed.note == "fetch_failed":
                    stats["failures"] += 1
        else:
            final_start = sheet_start
            final_end = sheet_end
            time_verified = "FALLBACK"
            verify_notes = "fetch_failed" if link else "parse_failed"
            stats["fallbacks"] += 1
            stats["failures"] += 1 if link else 0

        outcomes.append(
            RowOutcome(
                date=to_date_str(final_start),
                start_time=to_time_str(final_start),
                end_time=to_time_str(final_end),
                time_verified=time_verified,
                verify_notes=verify_notes,
            )
        )

    out_date = [[o.date] for o in outcomes]
    out_start = [[o.start_time] for o in outcomes]
    out_end = [[o.end_time] for o in outcomes]
    out_verified = [[o.time_verified] for o in outcomes]
    out_notes = [[o.verify_notes] for o in outcomes]

    last_row = len(data_rows) + 1
    write_data = [
        {"range": f"{tab_name}!A1:{col_to_a1(len(headers)-1)}1", "values": [headers]},
        {
            "range": f"{tab_name}!{col_to_a1(header_positions['date'])}2:{col_to_a1(header_positions['date'])}{last_row}",
            "values": out_date,
        },
        {
            "range": f"{tab_name}!{col_to_a1(header_positions['starttime'])}2:{col_to_a1(header_positions['starttime'])}{last_row}",
            "values": out_start,
        },
        {
            "range": f"{tab_name}!{col_to_a1(header_positions['endtime'])}2:{col_to_a1(header_positions['endtime'])}{last_row}",
            "values": out_end,
        },
        {
            "range": f"{tab_name}!{col_to_a1(header_positions['timeverified'])}2:{col_to_a1(header_positions['timeverified'])}{last_row}",
            "values": out_verified,
        },
        {
            "range": f"{tab_name}!{col_to_a1(header_positions['verifynotes'])}2:{col_to_a1(header_positions['verifynotes'])}{last_row}",
            "values": out_notes,
        },
    ]

    sheets.spreadsheets().values().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body={"valueInputOption": "RAW", "data": write_data},
    ).execute()

    stats["cache_hits"] = max(0, stats["total_rows"] - stats["links_fetched"])

    print("Event time normalization complete")
    print(f"total rows: {stats['total_rows']}")
    print(f"links fetched: {stats['links_fetched']}")
    print(f"cache hits: {stats['cache_hits']}")
    print(f"parsed from page counts by method: {dict(parsed_method_counts)}")
    print(f"fallbacks count: {stats['fallbacks']}")
    print(f"mismatches count: {stats['mismatches']}")
    print(f"failures count: {stats['failures']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
