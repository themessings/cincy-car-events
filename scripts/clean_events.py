#!/usr/bin/env python3
"""One-time cleanup of the existing Cincinnati car-events export.

Runs the same export-quality passes the collector now applies, but over the
already-collected ``data/events.csv`` so you get a cleaned sheet without a full
re-collection:

  1. drop non-car events (word-boundary, source-trust aware)
  2. open each event link and confirm/correct its date & time
  3. look up a full street address when only a venue/city is known
  4. merge duplicates (exact + same-day near-duplicates)
  5. add Size / Popularity / Attendance columns so big events surface

Outputs ``data/events.cleaned.csv`` and ``data/events.cleaned.json`` — the
tracked ``events.csv``/``events.json`` are left untouched. Facebook attendance
counts stay blank unless FACEBOOK_ACCESS_TOKEN is set (they fill in on the next
CI run).

Usage:
    python scripts/clean_events.py                 # full clean (slow: fetches links)
    python scripts/clean_events.py --no-verify-dates   # skip link date-checks
    python scripts/clean_events.py --no-address-lookup  # skip geocoded addresses
    python scripts/clean_events.py --limit 50 --in-place data/events.csv
"""
import argparse
import csv
import os
import sys
from datetime import timedelta

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import scripts.events_collector as EC  # noqa: E402

DATA_DIR = os.path.join(REPO_ROOT, "data")
DEFAULT_INPUT = os.path.join(DATA_DIR, "events.csv")
DEFAULT_OUT_CSV = os.path.join(DATA_DIR, "events.cleaned.csv")
DEFAULT_OUT_JSON = os.path.join(DATA_DIR, "events.cleaned.json")


def _row_get(row: dict, *keys: str) -> str:
    for key in keys:
        if key in row and EC.clean_ws(str(row[key])):
            return EC.clean_ws(str(row[key]))
    return ""


def row_to_internal(row: dict) -> dict:
    """Turn an export CSV row back into the collector's internal event shape."""
    title = _row_get(row, "Event Name", "title", "name")
    date = _row_get(row, "Date", "date")
    start_time = _row_get(row, "Start Time", "start_time")
    end_time = _row_get(row, "End Time", "end_time")
    location = _row_get(row, "Location", "location", "venue")
    address = _row_get(row, "Address", "address")
    url = _row_get(row, "Event URL", "url", "link")
    source = _row_get(row, "Source", "source")
    closest_city = _row_get(row, "Closest City", "closest_city")
    callout = _row_get(row, "Callout", "callout")

    start_dt = EC.parse_dt(f"{date} {start_time}".strip()) or EC.parse_dt(date)
    end_dt = EC.parse_dt(f"{date} {end_time}".strip()) if (date and end_time) else None
    if start_dt and (end_dt is None or end_dt <= start_dt):
        end_dt = start_dt + timedelta(hours=2)

    city_state = EC.guess_city_state(location) or closest_city
    # Existing exports duplicated Location into Address; treat that as "no real
    # street address yet" so the lookup pass tries to find one.
    real_address = address if (address and address != location and EC._has_full_street_address(address)) else ""

    return {
        "title": title,
        "location": location,
        "address": real_address,
        "url": url,
        "source": source,
        "closest_city": closest_city,
        "callout": callout,
        "city_state": city_state,
        "start_iso": start_dt.isoformat() if start_dt else "",
        "end_iso": end_dt.isoformat() if end_dt else "",
    }


def load_rows(path: str) -> list:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean the existing events export.")
    parser.add_argument("input", nargs="?", default=DEFAULT_INPUT, help="Input events CSV (default data/events.csv)")
    parser.add_argument("--out-csv", default=DEFAULT_OUT_CSV)
    parser.add_argument("--out-json", default=DEFAULT_OUT_JSON)
    parser.add_argument("--no-verify-dates", action="store_true", help="Skip opening links to re-check date/time")
    parser.add_argument("--no-address-lookup", action="store_true", help="Skip geocoded full-address lookup")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N rows (debugging)")
    parser.add_argument("--in-place", action="store_true", help="Also overwrite the input CSV/JSON with the cleaned data")
    args = parser.parse_args()

    cfg = EC.load_yaml(EC.CONFIG_PATH)
    raw_rows = load_rows(args.input)
    if args.limit:
        raw_rows = raw_rows[: args.limit]
    print(f"Loaded {len(raw_rows)} rows from {args.input}", flush=True)

    internal = [row_to_internal(r) for r in raw_rows]

    bypass_sources = {
        EC.clean_ws(str(src.get("name", "")))
        for src in (cfg.get("sources", []) or [])
        if EC.clean_ws(str(src.get("bypass_automotive_filter", ""))).lower() in {"1", "true", "yes", "y"}
    }

    geocache = EC.load_json(EC.GEOCODE_CACHE_PATH, {})
    url_cache = EC.load_json(EC.URL_CACHE_PATH, {})

    # Exact dedupe first (cheap), then the enrichment passes.
    deduped = EC.dedupe_export_rows(internal)
    print(f"After exact dedupe: {len(deduped)}", flush=True)

    fetch_fb = bool(EC.get_facebook_access_token())
    enriched = EC.enrich_events_for_export(
        deduped,
        geocache,
        url_cache,
        cfg,
        verify_dates=not args.no_verify_dates,
        lookup_addresses=not args.no_address_lookup,
        fetch_facebook=fetch_fb,
        revet_automotive=True,
        max_workers=args.workers,
        trusted_sources=bypass_sources,
    )
    print(f"After enrichment (non-car removed, near-dups merged): {len(enriched)}", flush=True)

    export_rows, headers = EC.normalize_export_schema(enriched)

    EC.write_csv(export_rows, args.out_csv)
    EC.save_json(args.out_json, {
        "generated_at_iso": EC.now_et_iso(),
        "count": len(export_rows),
        "events": export_rows,
    })
    EC.save_json(EC.GEOCODE_CACHE_PATH, geocache)
    EC.save_json(EC.URL_CACHE_PATH, url_cache)

    if args.in_place:
        EC.write_csv(export_rows, args.input)
        json_path = os.path.splitext(args.input)[0] + ".json"
        EC.save_json(json_path, {
            "generated_at_iso": EC.now_et_iso(),
            "count": len(export_rows),
            "events": export_rows,
        })

    # Summary
    major = sum(1 for r in export_rows if r.get("Size") == "Major")
    regional = sum(1 for r in export_rows if r.get("Size") == "Regional")
    with_addr = sum(1 for r in export_rows if EC._has_full_street_address(str(r.get("Address", ""))))
    with_att = sum(1 for r in export_rows if EC.clean_ws(str(r.get("Attendance", ""))))
    print("=" * 60, flush=True)
    print(f"IN:  {len(raw_rows)} rows", flush=True)
    print(f"OUT: {len(export_rows)} rows  (removed {len(raw_rows) - len(export_rows)})", flush=True)
    print(f"  Major={major}  Regional={regional}", flush=True)
    print(f"  full street address: {with_addr}/{len(export_rows)}", flush=True)
    print(f"  with attendance numbers: {with_att}", flush=True)
    print(f"Wrote {args.out_csv}", flush=True)
    print(f"Wrote {args.out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
