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
