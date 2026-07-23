# cincy-car-events

## Runtime setup notes

- `APEX_FACEBOOK_PAGES_SHEET_ID` should be set to the spreadsheet ID `1pol-GLdo3ylizOJ0DG3cDwo9TVClEJ23To7ber0waik`.
  - You may also provide a full Google Sheets URL; runtime validation/collector will auto-extract the spreadsheet ID.
- Share the Facebook Pages sheet with the Google service account email from `GDRIVE_SERVICE_ACCOUNT_JSON` / `GOOGLE_SERVICE_ACCOUNT_JSON`.
- `FACEBOOK_ACCESS_TOKEN` must be present for Graph API page-event collection.
- `FACEBOOK_APP_ID` and `FACEBOOK_APP_SECRET` should be set to enable automatic `/debug_token` validation and long-lived token refresh (`fb_exchange_token`).
- Optional: `ENABLE_PLAYWRIGHT_FB=1` enables JS-rendered Facebook event-page parsing fallback if static HTML is insufficient.
- `ENABLE_FACEBOOK_SERP_DISCOVERY` controls optional Facebook event URL discovery via SerpAPI. It is **enabled by default**; set to `0`, `false`, or `no` to disable.
- Optional SerpAPI Google Events tuning: `SERPAPI_LOCATION` (default `Cincinnati, OH`), `SERPAPI_GL` (default `us`), `SERPAPI_HL` (default `en`), `SERPAPI_EVENTS_DATE_FILTER` (default `date:month`, sent as `htichips`).

## Output columns

The Events export (CSV, `events.json`, and the Google Sheet) has these columns:

`Event Name`, `Date`, `Start Time`, `End Time`, `Location`, `Address`, `Closest City`, `Callout`, `Size`, `Popularity`, `Attendance`, `Source`, `Event URL`

- **Location** is the human venue label; **Address** is the full street address (looked up when the source only gave a venue/city).
- **Size** is a tier — `Major` / `Regional` / `Local` / `Small`.
- **Popularity** is a 0–100 score (higher = bigger). Sort this column **descending** to put the biggest events on top. It blends real Facebook engagement (when known) with a heuristic from multi-day span, marquee/venue signals (nationals, concours, speedway, fairgrounds…), and recurring/local cues.
- **Attendance** shows real Facebook `going / interested` counts when available (e.g. `600 going · 2.4k interested`); blank otherwise.

## Export enrichment

After events are collected, merged, and deduped, a final enrichment pass runs (all on by default):

1. **Non-car vetting** — a word-boundary, content-based automotive classifier. The event's title/description/location decides the automotive signal (never the source name — the source `Meetup Cincy Cars` used to auto-pass every unrelated meetup). Sources flagged `bypass_automotive_filter` and those in `filters.car_dedicated_sources` are trusted without a keyword. Keyword tiers live in `config/sources.yml` under `filters.automotive_strong_keywords` / `automotive_context_keywords` / `automotive_weak_keywords` / `non_automotive_exclude_keywords`.
2. **Date/time verification** — opens each event's link and corrects the start/end from schema.org data or the Facebook Graph API (cached 7 days).
3. **Full-address lookup** — geocodes a full street address via Nominatim when the source only had a venue/city (cached; state-checked to avoid wrong matches).
4. **Popularity/size scoring** and **same-day near-duplicate merge** (e.g. `57th NSRA Street Rod Nationals` vs `57th Annual Street Rod Nationals`).

Toggles (env vars, all default on): `ENABLE_EXPORT_ENRICHMENT`, `ENABLE_LINK_DATE_CHECK`, `ENABLE_ADDRESS_LOOKUP`, `EXPORT_ENRICH_WORKERS` (default 6). Optional USPS address verification is available via `verify_usps_address()` when `USPS_USER_ID` is set (off by default; no network call otherwise).

## One-time cleanup of the existing export

To re-run these quality passes over the already-collected `data/events.csv` without a full re-collection:

```bash
python scripts/clean_events.py
```

It writes `data/events.cleaned.csv` and `data/events.cleaned.json` (the tracked `events.csv`/`events.json` are left untouched; pass `--in-place` to overwrite them). Flags: `--no-verify-dates`, `--no-address-lookup`, `--limit N`, `--workers N`. Facebook attendance counts stay blank locally unless `FACEBOOK_ACCESS_TOKEN` is set.

## Facebook Pages sheet format

Use tab `Pages` (default; can be overridden by `APEX_FACEBOOK_PAGES_TAB`) with headers:

- `page_url` (required)
- `enabled` (optional, defaults to TRUE)
- `label` (optional)
- `notes` (optional)

Accepted `page_url` formats include (for direct Graph pulls):

- `https://www.facebook.com/<username>`
- `https://facebook.com/<username>`
- `https://www.facebook.com/profile.php?id=<numeric>`
- trailing slash/query variants


Non-Facebook `page_url` rows are no longer ignored. They are treated as organizer sources and used to seed SerpAPI discovery queries for `facebook.com/events/*` URLs (using `label`, domain, and URL context).

## Facebook feed scroll (optional)

The `facebook_events_scroll` source (config: `Facebook Feed Scroll`) scrolls your logged-in Facebook **"Most Recent" home feed** with a headless browser, harvests the `facebook.com/events/<id>` links that appear (event shares, "interested" cards, suggested events), and feeds them through the same automotive-focus, distance, and date filters as every other source. It is **opt-in and a no-op by default**, so the standard cron is unaffected unless you turn it on.

To enable it you need three things:

1. **`ENABLE_FACEBOOK_SCROLL=1`** — locally an env var; in CI a repository *variable* (Settings → Secrets and variables → Actions → Variables). The gated Playwright-install workflow step and the collector both key off this.
2. **Playwright + Chromium** — `pip install -r requirements-facebook.txt && python -m playwright install chromium` locally; installed automatically by the gated CI step.
3. **`FACEBOOK_SESSION_STATE`** — a Playwright `storageState` captured once from a browser already logged into Facebook. Store the JSON as a secret (raw or base64). Credentials-based login is intentionally unsupported — Facebook challenges datacenter logins with CAPTCHAs/checkpoints and can lock the account.

Export a session state locally (do this on a machine where you can log in interactively):

```bash
python - <<'PY'
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    page.goto("https://www.facebook.com/login")
    input("Log in fully in the opened browser, then press Enter here...")
    page.context.storage_state(path="fb_state.json")
    browser.close()
print("Wrote fb_state.json")
PY
```

Then point the collector at it:

- Local run: `FACEBOOK_SESSION_STATE_PATH=fb_state.json` (a file path), or `FACEBOOK_SESSION_STATE="$(cat fb_state.json)"` (raw JSON).
- CI secret: `base64 -i fb_state.json | pbcopy` and paste into the `FACEBOOK_SESSION_STATE` secret (the loader auto-detects base64).

The session eventually expires; when the run logs `session_invalid_or_expired`, re-export `fb_state.json` and update the secret. Optional tuning (source keys or env): `scrolls`/`FACEBOOK_SCROLL_COUNT` (default 20), `scroll_pause_ms`/`FACEBOOK_SCROLL_PAUSE_MS` (default 2000), `max_events`/`FACEBOOK_SCROLL_MAX_EVENTS` (default 40), and `feed_url`/`FACEBOOK_FEED_URL`.

Note: automated scrolling is against Facebook's Terms of Service and carries some account risk; a low-value/secondary account and modest `scrolls` are advisable.
