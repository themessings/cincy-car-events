#!/usr/bin/env python3
"""Validate required/optional runtime env vars for the collector workflow."""

from __future__ import annotations

import os
import re
import sys

import requests


def extract_spreadsheet_id(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return ""
    if "docs.google.com/spreadsheets" in raw:
        match = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", raw)
        if match:
            return match.group(1)
    return raw




def fb_graph_token_status() -> str:
    token = os.getenv("FACEBOOK_ACCESS_TOKEN", "").strip()
    if not token:
        return "no (missing)"
    try:
        r = requests.get(
            "https://graph.facebook.com/v18.0/me",
            params={"access_token": token, "fields": "id"},
            timeout=10,
        )
        if r.status_code == 200:
            return "yes"
        try:
            err = (r.json() or {}).get("error", {})
        except Exception:
            err = {}
        code = str(err.get("code", ""))
        subcode = str(err.get("error_subcode", ""))
        msg = str(err.get("message", "")).lower()
        if code == "190" or subcode == "463" or "session has expired" in msg:
            return "no (expired)"
        return f"no (http {r.status_code})"
    except Exception:
        return "no (unverified)"


def main() -> int:
    missing_required = []

    if not os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON"):
        missing_required.append(
            "❌ Missing GDRIVE_SERVICE_ACCOUNT_JSON secret; spreadsheet update cannot run."
        )

    if not os.getenv("APEX_SPREADSHEET_ID"):
        missing_required.append(
            "❌ Missing APEX_SPREADSHEET_ID; spreadsheet destination is undefined."
        )

    if not os.getenv("FACEBOOK_ACCESS_TOKEN"):
        print("⚠️ FACEBOOK_ACCESS_TOKEN is empty; Facebook Graph API collection will be skipped.")

    print(f"ℹ️ FACEBOOK_GRAPH token_valid: {fb_graph_token_status()}")

    if not os.getenv("APEX_FACEBOOK_PAGES_SHEET_ID") and not os.getenv("FACEBOOK_PAGE_IDS"):
        print("⚠️ Neither APEX_FACEBOOK_PAGES_SHEET_ID nor FACEBOOK_PAGE_IDS is set; FB page events source has no IDs.")

    raw_pages_sheet = os.getenv("APEX_FACEBOOK_PAGES_SHEET_ID", "")
    normalized_pages_sheet = extract_spreadsheet_id(raw_pages_sheet)
    if raw_pages_sheet:
        if raw_pages_sheet != normalized_pages_sheet:
            print(
                "ℹ️ APEX_FACEBOOK_PAGES_SHEET_ID provided as URL; "
                f"extracted spreadsheet ID: {normalized_pages_sheet}"
            )
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

    fb_discovery_enabled = os.getenv("ENABLE_FACEBOOK_SERP_DISCOVERY", "1").strip().lower() not in {"0", "false", "no"}
    print(
        "ℹ️ ENABLE_FACEBOOK_SERP_DISCOVERY is "
        + ("enabled" if fb_discovery_enabled else "disabled")
        + " (default: enabled)."
    )

    if missing_required:
        for msg in missing_required:
            print(msg)
        print("Failing early so required spreadsheet update inputs are fixed.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
