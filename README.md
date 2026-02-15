# cincy-car-events

## Runtime setup notes

- `APEX_FACEBOOK_PAGES_SHEET_ID` should be set to the spreadsheet ID `1pol-GLdo3ylizOJ0DG3cDwo9TVClEJ23To7ber0waik`.
  - You may also provide a full Google Sheets URL; runtime validation/collector will auto-extract the spreadsheet ID.
- Share the Facebook Pages sheet with the Google service account email from `GDRIVE_SERVICE_ACCOUNT_JSON` / `GOOGLE_SERVICE_ACCOUNT_JSON`.
- `FACEBOOK_ACCESS_TOKEN` must be present for Graph API page-event collection.
- `ENABLE_FACEBOOK_SERP_DISCOVERY` controls optional Facebook event URL discovery via SerpAPI. It is **enabled by default**; set to `0`, `false`, or `no` to disable.

## Facebook Pages sheet format

Use tab `Pages` (default; can be overridden by `APEX_FACEBOOK_PAGES_TAB`) with headers:

- `page_url` (required)
- `enabled` (optional, defaults to TRUE)
- `label` (optional)
- `notes` (optional)

Accepted `page_url` formats include:

- `https://www.facebook.com/<username>`
- `https://facebook.com/<username>`
- `https://www.facebook.com/profile.php?id=<numeric>`
- trailing slash/query variants
