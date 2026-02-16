#!/usr/bin/env python3
"""Validate required/optional runtime env vars for the collector workflow."""

from __future__ import annotations

import os
import re
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.facebook_token_manager import TokenManager


def extract_spreadsheet_id(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    if "docs.google.com/spreadsheets" in raw:
        match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", raw)
        if match:
            return match.group(1)
    return raw


def main() -> int:
    missing_required = []

    if not os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON"):
        missing_required.append("❌ Missing GDRIVE_SERVICE_ACCOUNT_JSON secret; spreadsheet update cannot run.")

    if not os.getenv("APEX_SPREADSHEET_ID"):
        missing_required.append("❌ Missing APEX_SPREADSHEET_ID; spreadsheet destination is undefined.")

    token_env_name = "FACEBOOK_ACCESS_TOKEN"
    token_present = bool(os.getenv(token_env_name))
    print(f"ℹ️ FACEBOOK token env var name: {token_env_name}")
    print(f"ℹ️ FACEBOOK token env var non-empty: {'yes' if token_present else 'no'}")

    if not token_present:
        print("⚠️ FACEBOOK_ACCESS_TOKEN is empty; Facebook Graph API collection will be skipped.")

    app_id = bool((os.getenv("FACEBOOK_APP_ID") or "").strip())
    app_secret = bool((os.getenv("FACEBOOK_APP_SECRET") or "").strip())
    print(f"ℹ️ FACEBOOK_APP_ID set: {'yes' if app_id else 'no'}")
    print(f"ℹ️ FACEBOOK_APP_SECRET set: {'yes' if app_secret else 'no'}")
    if token_present and (not app_id or not app_secret):
        print("⚠️ FACEBOOK_APP_ID / FACEBOOK_APP_SECRET missing; token auto-refresh may not work reliably.")

    manager = TokenManager(print)
    token_status = manager.ensure_valid_user_token(refresh_days_threshold=7)
    print(
        "ℹ️ FACEBOOK_GRAPH token_valid: "
        f"{token_status.get('valid', 'no')} "
        f"(reason: {token_status.get('reason', 'unknown')}, "
        f"expires_at_et: {token_status.get('expires_at_et', 'unknown')})"
    )

    if token_status.get("refresh_attempted") and not token_status.get("refresh_succeeded"):
        print(f"⚠️ FACEBOOK token refresh attempt failed: {token_status.get('refresh_error')}")

    if not os.getenv("APEX_FACEBOOK_PAGES_SHEET_ID") and not os.getenv("FACEBOOK_PAGE_IDS"):
        print("⚠️ Neither APEX_FACEBOOK_PAGES_SHEET_ID nor FACEBOOK_PAGE_IDS is set; FB page events source has no IDs.")

    raw_pages_sheet = os.getenv("APEX_FACEBOOK_PAGES_SHEET_ID", "")
    normalized_pages_sheet = extract_spreadsheet_id(raw_pages_sheet)
    if raw_pages_sheet:
        if raw_pages_sheet != normalized_pages_sheet:
            print("ℹ️ APEX_FACEBOOK_PAGES_SHEET_ID provided as URL; " f"extracted spreadsheet ID: {normalized_pages_sheet}")
        if re.fullmatch(r"[a-zA-Z0-9-_]{20,}", normalized_pages_sheet):
            print(f"✅ APEX_FACEBOOK_PAGES_SHEET_ID will use spreadsheet ID: {normalized_pages_sheet}")
        else:
            print(
                "⚠️ APEX_FACEBOOK_PAGES_SHEET_ID is set but could not be parsed as a valid "
                f"spreadsheet ID after normalization: '{normalized_pages_sheet or raw_pages_sheet}'."
            )

    if os.getenv("COLLECTOR_DRY_RUN"):
        print("ℹ️ COLLECTOR_DRY_RUN is set; collector will skip Google Sheets writes.")

    if not os.getenv("SERPAPI_API_KEY"):
        print("ℹ️ SERPAPI_API_KEY is optional; discovery sources will self-skip when unset.")
    print(f"ℹ️ SERPAPI_LOCATION: {(os.getenv('SERPAPI_LOCATION') or 'Cincinnati, OH').strip() or 'Cincinnati, OH'}")
    print(f"ℹ️ SERPAPI_GL: {(os.getenv('SERPAPI_GL') or 'us').strip() or 'us'}")
    print(f"ℹ️ SERPAPI_HL: {(os.getenv('SERPAPI_HL') or 'en').strip() or 'en'}")
    print(f"ℹ️ SERPAPI_EVENTS_DATE_FILTER: {(os.getenv('SERPAPI_EVENTS_DATE_FILTER') or 'date:month').strip() or 'date:month'}")

    fb_discovery_enabled = os.getenv("ENABLE_FACEBOOK_SERP_DISCOVERY", "1").strip().lower() not in {"0", "false", "no"}
    print("ℹ️ ENABLE_FACEBOOK_SERP_DISCOVERY is " + ("enabled" if fb_discovery_enabled else "disabled") + " (default: enabled).")

    if missing_required:
        for msg in missing_required:
            print(msg)
        print("Failing early so required spreadsheet update inputs are fixed.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
