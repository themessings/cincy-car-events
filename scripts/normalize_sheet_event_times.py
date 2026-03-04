#!/usr/bin/env python3
"""Normalize/verify event date and time columns in a Google Sheet.

Creates/populates columns: date, start_time, end_time, verified_source, verify_status.
Preserves original iso/end_iso.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

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


def has_time_component(raw_iso: str) -> bool:
    return bool(raw_iso and "T" in raw_iso)


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


def to_date_str(dt: Optional[datetime]) -> str:
    return dt.strftime("%Y-%m-%d") if dt else ""


def to_time_str(dt: Optional[datetime]) -> str:
    if not dt:
        return ""
    return dt.strftime("%I:%M %p").lstrip("0")


def col_to_a1(idx0: int) -> str:
    n = idx0 + 1
    out = ""
    while n:
        n, r = divmod(n - 1, 26)
        out = chr(65 + r) + out
    return out


@dataclass
class VerifyResult:
    start_dt: Optional[datetime]
    end_dt: Optional[datetime]
    source: str
    status: str


class DomainRateLimiter:
    def __init__(self, every_n: int = 3, sleep_s: float = 0.6):
        self.every_n = every_n
        self.sleep_s = sleep_s
        self.lock = threading.Lock()
        self.counts: Dict[str, int] = defaultdict(int)

    def wait(self, domain: str) -> None:
        with self.lock:
            self.counts[domain] += 1
            if self.counts[domain] % self.every_n == 0:
                time.sleep(self.sleep_s)


def collect_jsonld_event_obj(obj):
    if isinstance(obj, list):
        for item in obj:
            found = collect_jsonld_event_obj(item)
            if found:
                return found
        return None
    if not isinstance(obj, dict):
        return None
    typ = obj.get("@type")
    if isinstance(typ, list):
        is_event = any(str(t).lower() == "event" for t in typ)
    else:
        is_event = str(typ).lower() == "event"
    if is_event:
        return obj
    if "@graph" in obj:
        return collect_jsonld_event_obj(obj.get("@graph"))
    return None


def _parse_dt(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = dtparser.isoparse(value)
    except Exception:
        try:
            dt = dtparser.parse(value, fuzzy=True)
        except Exception:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=DEFAULT_TZ)
    return dt


def parse_from_jsonld(soup: BeautifulSoup) -> Tuple[Optional[datetime], Optional[datetime]]:
    scripts = soup.find_all("script", attrs={"type": re.compile("ld\+json", re.I)})
    for s in scripts:
        raw = (s.string or s.get_text() or "").strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except Exception:
            continue
        event_obj = collect_jsonld_event_obj(payload)
        if not event_obj:
            continue
        start = _parse_dt(str(event_obj.get("startDate", "")))
        end = _parse_dt(str(event_obj.get("endDate", "")))
        if start:
            return start, end
    return None, None


def parse_from_time_tags(soup: BeautifulSoup) -> Tuple[Optional[datetime], Optional[datetime]]:
    times = soup.find_all("time")
    start = end = None
    for t in times:
        cand = _parse_dt((t.get("datetime") or "").strip())
        if not cand:
            continue
        attrs = " ".join(
            [t.get("itemprop", ""), " ".join(t.get("class", [])), t.get("id", "")]
        ).lower()
        if any(k in attrs for k in ["start", "from"]):
            start = cand
        elif any(k in attrs for k in ["end", "to"]):
            end = cand
        elif not start:
            start = cand
        elif not end:
            end = cand
    return start, end


def parse_from_meta(soup: BeautifulSoup) -> Tuple[Optional[datetime], Optional[datetime]]:
    start = end = None
    for m in soup.find_all("meta"):
        key = " ".join([m.get("property", ""), m.get("name", ""), m.get("itemprop", "")]).lower()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if any(k in key for k in ["start", "startdate"]):
            start = start or _parse_dt(content)
        if any(k in key for k in ["end", "enddate"]):
            end = end or _parse_dt(content)
    return start, end


def parse_from_text(soup: BeautifulSoup, fallback_date: Optional[datetime]) -> Tuple[Optional[datetime], Optional[datetime]]:
    text = soup.get_text(" ", strip=True)
    time_range = re.search(
        r"(\d{1,2}(?::\d{2})?\s*[AaPp]\.?[Mm]\.?)\s*(?:-|–|—|to)\s*(\d{1,2}(?::\d{2})?\s*[AaPp]\.?[Mm]\.?)",
        text,
    )
    date_match = re.search(
        r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s*\d{4}|\d{1,2}/\d{1,2}/\d{2,4})",
        text,
        flags=re.I,
    )

    base_date = None
    if date_match:
        try:
            base_date = dtparser.parse(date_match.group(1), fuzzy=True)
        except Exception:
            base_date = None
    if base_date is None and fallback_date is not None:
        base_date = fallback_date

    if not time_range or base_date is None:
        return None, None

    t1_raw, t2_raw = time_range.group(1), time_range.group(2)
    try:
        t1 = dtparser.parse(t1_raw, fuzzy=True)
        t2 = dtparser.parse(t2_raw, fuzzy=True)
    except Exception:
        return None, None

    start = datetime(
        base_date.year,
        base_date.month,
        base_date.day,
        t1.hour,
        t1.minute,
        tzinfo=base_date.tzinfo or DEFAULT_TZ,
    )
    end = datetime(
        base_date.year,
        base_date.month,
        base_date.day,
        t2.hour,
        t2.minute,
        tzinfo=base_date.tzinfo or DEFAULT_TZ,
    )
    if end <= start:
        end += timedelta(days=1)
    return start, end


def verify_from_page(url: str, fallback_start: Optional[datetime], fallback_end: Optional[datetime], session: requests.Session, limiter: DomainRateLimiter) -> VerifyResult:
    domain = (urlparse(url).hostname or "").lower()
    try:
        limiter.wait(domain)
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
    except Exception:
        return VerifyResult(fallback_start, fallback_end, "page_failed", "failed_fetch")

    soup = BeautifulSoup(resp.text, "html.parser")

    start, end = parse_from_jsonld(soup)
    if start:
        return VerifyResult(start, end or fallback_end, "page_jsonld", "verified_ok")

    start, end = parse_from_time_tags(soup)
    if start:
        return VerifyResult(start, end or fallback_end, "page_time_tag", "verified_ok")

    start, end = parse_from_meta(soup)
    if start:
        return VerifyResult(start, end or fallback_end, "page_meta", "verified_ok")

    start, end = parse_from_text(soup, fallback_start)
    if start:
        return VerifyResult(start, end or fallback_end, "page_text", "verified_ok")

    if fallback_start:
        return VerifyResult(fallback_start, fallback_end, "sheet_iso", "verified_fallback_sheet")
    return VerifyResult(None, None, "page_failed", "failed_parse")


def locate_headers(headers: List[str]) -> Dict[str, int]:
    nmap = {norm_header(h): i for i, h in enumerate(headers)}

    def pick(*cands):
        for c in cands:
            if c in nmap:
                return nmap[c]
        return -1

    return {
        "iso": pick("iso", "startiso", "startdatetime", "start"),
        "end_iso": pick("endiso", "enddatetime", "end"),
        "link": pick("link", "url", "eventurl", "eventlink"),
        "location": pick("location", "venue", "address"),
        "category": pick("category", "type"),
        "mileage": pick("mileage", "miles", "distance"),
    }


def parse_weekend(value: str) -> Optional[datetime]:
    if not value:
        return None
    dt = dtparser.parse(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=DEFAULT_TZ)
    return dt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spreadsheet", default="https://docs.google.com/spreadsheets/d/1lVpqhmUOQDZywjGeYxgm7ILNXqP3l6Z74pVXw1oKSQ8/edit?gid=1268351023")
    ap.add_argument("--tab", default="")
    ap.add_argument("--weekend-start", default="")
    ap.add_argument("--weekend-end", default="")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--unreliable-domains", default="")
    ap.add_argument("--small-mileage-threshold", type=float, default=None)
    ap.add_argument("--verify-local", action="store_true")
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
    for required in ["iso", "end_iso", "link"]:
        if idx[required] < 0:
            raise RuntimeError(f"Required header not found: {required}")

    new_cols = ["date", "start_time", "end_time", "verified_source", "verify_status"]
    for col in new_cols:
        if norm_header(col) not in [norm_header(h) for h in headers]:
            headers.append(col)

    header_positions = {norm_header(h): i for i, h in enumerate(headers)}

    data_rows = rows[1:]
    for r in data_rows:
        if len(r) < len(headers):
            r.extend([""] * (len(headers) - len(r)))

    weekend_start = parse_weekend(args.weekend_start)
    weekend_end = parse_weekend(args.weekend_end)
    unreliable_domains = {d.strip().lower() for d in args.unreliable_domains.split(",") if d.strip()}

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(max_retries=3, pool_connections=20, pool_maxsize=20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": UA})

    limiter = DomainRateLimiter()
    url_cache: Dict[str, VerifyResult] = {}
    to_verify_urls = set()
    suspect_link_counts = Counter()
    row_specs = []

    stats = Counter()
    by_method = Counter()

    for i, row in enumerate(data_rows):
        stats["rows_processed"] += 1
        raw_iso = row[idx["iso"]].strip() if idx["iso"] < len(row) else ""
        raw_end = row[idx["end_iso"]].strip() if idx["end_iso"] < len(row) else ""
        link = row[idx["link"]].strip() if idx["link"] < len(row) else ""

        start_dt = parse_iso(raw_iso)
        end_dt = parse_iso(raw_end)

        include_row = True
        if weekend_start and weekend_end and start_dt:
            include_row = weekend_start <= start_dt <= weekend_end

        suspect = False
        if not start_dt or not has_time_component(raw_iso):
            suspect = True
        if not end_dt:
            suspect = True
        if start_dt and end_dt:
            duration = end_dt - start_dt
            if end_dt <= start_dt or duration > timedelta(hours=18):
                suspect = True
        if include_row:
            suspect = True
        domain = (urlparse(link).hostname or "").lower()
        if domain in unreliable_domains:
            suspect = True

        if args.verify_local and idx["category"] >= 0:
            if "local" in (row[idx["category"]] or "").lower():
                suspect = True

        if args.small_mileage_threshold is not None and idx["mileage"] >= 0:
            m = re.search(r"\d+(?:\.\d+)?", row[idx["mileage"]] or "")
            if m and float(m.group(0)) <= args.small_mileage_threshold:
                suspect = True

        if idx["location"] >= 0 and re.search(r"\b(?:\d{1,2}/\d{1,2}/\d{2,4}|\d{1,2}:\d{2}\s*[AP]M?)\b", row[idx["location"]] or "", re.I):
            suspect = True

        row_specs.append((i, row, include_row, suspect, start_dt, end_dt, link))
        if include_row and suspect and link:
            to_verify_urls.add(link)
            suspect_link_counts[link] += 1
            stats["suspect_rows"] += 1

    def _fetch(url: str) -> Tuple[str, VerifyResult]:
        return url, verify_from_page(url, None, None, session, limiter)

    if to_verify_urls:
        futures = {}
        with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 6))) as ex:
            for url in to_verify_urls:
                futures[ex.submit(_fetch, url)] = url
            for fut in as_completed(futures):
                url, result = fut.result()
                url_cache[url] = result
                stats["links_fetched"] += 1

    stats["cache_hits"] = sum(max(0, count - 1) for count in suspect_link_counts.values())

    out_date, out_start, out_end, out_source, out_status = [], [], [], [], []

    for _, _, include_row, suspect, start_dt, end_dt, link in row_specs:
        source = "sheet_iso"
        status = "skipped_not_suspect"
        final_start = start_dt
        final_end = end_dt

        if include_row and suspect:
            if link in url_cache:
                res = url_cache[link]
                if res.start_dt:
                    final_start = res.start_dt
                if res.end_dt:
                    final_end = res.end_dt
                source = res.source if res.source != "page_failed" else "sheet_iso"
                status = res.status if res.status != "failed_parse" or final_start is None else "verified_fallback_sheet"
                if res.source.startswith("page_") and res.status == "verified_ok":
                    by_method[res.source] += 1
                    stats["verified_from_page"] += 1
                elif status == "verified_fallback_sheet":
                    stats["fallback_sheet"] += 1
                if res.status in {"failed_fetch", "failed_parse"}:
                    stats["parse_failures"] += 1
            else:
                status = "failed_fetch"
                source = "page_failed"

        out_date.append([to_date_str(final_start)])
        out_start.append([to_time_str(final_start)])
        out_end.append([to_time_str(final_end)])
        out_source.append([source])
        out_status.append([status])

    header_range = f"{tab_name}!A1:{col_to_a1(len(headers)-1)}1"
    write_data = [
        {"range": header_range, "values": [headers]},
        {
            "range": f"{tab_name}!{col_to_a1(header_positions['date'])}2:{col_to_a1(header_positions['date'])}{len(data_rows)+1}",
            "values": out_date,
        },
        {
            "range": f"{tab_name}!{col_to_a1(header_positions['starttime'])}2:{col_to_a1(header_positions['starttime'])}{len(data_rows)+1}",
            "values": out_start,
        },
        {
            "range": f"{tab_name}!{col_to_a1(header_positions['endtime'])}2:{col_to_a1(header_positions['endtime'])}{len(data_rows)+1}",
            "values": out_end,
        },
        {
            "range": f"{tab_name}!{col_to_a1(header_positions['verifiedsource'])}2:{col_to_a1(header_positions['verifiedsource'])}{len(data_rows)+1}",
            "values": out_source,
        },
        {
            "range": f"{tab_name}!{col_to_a1(header_positions['verifystatus'])}2:{col_to_a1(header_positions['verifystatus'])}{len(data_rows)+1}",
            "values": out_status,
        },
    ]

    sheets.spreadsheets().values().batchUpdate(
        spreadsheetId=spreadsheet_id,
        body={"valueInputOption": "RAW", "data": write_data},
    ).execute()

    print("Normalization complete")
    print(f"rows processed: {stats['rows_processed']}")
    print(f"suspect rows count: {stats['suspect_rows']}")
    print(f"links fetched: {stats['links_fetched']}")
    print(f"cache hits: {stats['cache_hits']}")
    print("verified from page count:", dict(by_method))
    print(f"fallbacks to sheet: {stats['fallback_sheet']}")
    print(f"parse failures: {stats['parse_failures']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
