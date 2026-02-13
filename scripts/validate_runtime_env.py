#!/usr/bin/env python3
"""Validate required/optional runtime env vars for the collector workflow."""

from __future__ import annotations

import os
import sys


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

    if not os.getenv("APEX_FACEBOOK_PAGES_SHEET_ID") and not os.getenv("FACEBOOK_PAGE_IDS"):
        print("⚠️ Neither APEX_FACEBOOK_PAGES_SHEET_ID nor FACEBOOK_PAGE_IDS is set; FB page events source has no IDs.")

    if os.getenv("APEX_FACEBOOK_PAGES_SHEET_ID") and "docs.google.com" in os.getenv("APEX_FACEBOOK_PAGES_SHEET_ID", ""):
        print("⚠️ APEX_FACEBOOK_PAGES_SHEET_ID appears to be a URL; use only the spreadsheet ID.")

    if os.getenv("COLLECTOR_DRY_RUN"):
        print("ℹ️ COLLECTOR_DRY_RUN is set; collector will skip Google Sheets writes.")

    if not os.getenv("SERPAPI_API_KEY"):
        print("ℹ️ SERPAPI_API_KEY is optional; discovery sources will self-skip when unset.")

    if missing_required:
        for msg in missing_required:
            print(msg)
        print("Failing early so required spreadsheet update inputs are fixed.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
