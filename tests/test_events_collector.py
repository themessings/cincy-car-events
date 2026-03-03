import json
import unittest
from collections import Counter
from datetime import datetime
from dateutil import tz
from pathlib import Path
from unittest.mock import patch

from scripts.events_collector import (
    EventItem,
    dedupe_merge,
    evaluate_automotive_focus_event,
    extract_facebook_page_identifier,
    extract_facebook_group_key,
    classify_facebook_pages_url,
    normalize_facebook_page_event,
    collect_facebook_events_from_pages,
    collect_facebook_events_serpapi_discovery,
    collect_facebook_group_events_serpapi,
    SourceDisabledError,
    derive_zero_yield_reason,
    main,
    load_facebook_targets,
    collect_ics,
    _parse_google_sheet_events_rows,
    to_event_items,
    simplify_location,
    parse_dt,
    extract_event_details_from_jsonld,
)


class CollectorTests(unittest.TestCase):
    def test_dedupe_signature_keeps_distinct_dates(self):
        base = EventItem(
            title="Cars and Coffee Cincinnati",
            start_iso="2026-03-01T09:00:00-05:00",
            end_iso="2026-03-01T11:00:00-05:00",
            location="Cincinnati OH",
            city_state="Cincinnati, OH",
            url="",
            source="A",
            category="local",
            miles_from_cincy=1.0,
            lat=39.1,
            lon=-84.5,
            last_seen_iso="2026-02-01T00:00:00-05:00",
        )
        incoming = [
            EventItem(**{**base.__dict__, "start_iso": "2026-03-08T09:00:00-05:00", "end_iso": "2026-03-08T11:00:00-05:00"}),
        ]
        merged = dedupe_merge([base], incoming)
        self.assertEqual(len(merged), 2)

    def test_filter_reason_non_automotive_keyword(self):
        cfg = {
            "filters": {
                "automotive_focus_keywords": ["car show", "cars and coffee"],
                "non_automotive_exclude_keywords": ["job fair"],
                "trusted_event_platforms": ["eventbrite.com"],
            }
        }
        ok, reason = evaluate_automotive_focus_event(
            title="Downtown Hiring Event",
            location="Cincinnati",
            source="Community Board",
            url="https://example.com/hiring",
            cfg=cfg,
        )
        self.assertFalse(ok)
        self.assertIn("hard_exclusion", reason)

    def test_normalize_facebook_event(self):
        raw = {
            "id": "12345",
            "name": "Cars & Coffee - Spring",
            "start_time": "2026-04-10T14:00:00+0000",
            "place": {"location": {"city": "Cincinnati", "state": "OH"}},
        }
        norm = normalize_facebook_page_event(raw, "Test Page")
        self.assertIsNotNone(norm)
        self.assertEqual(norm["title"], "Cars & Coffee - Spring")
        self.assertIn("facebook.com/events/12345", norm["url"])
        self.assertIsInstance(norm["start_dt"], datetime)


    def test_simplify_location_returns_address_only_in_us_format(self):
        location = "Circuit Cafe"
        address = "2726 Riverside Drive, Cincinnati, OH, United States, Ohio 45202"
        self.assertEqual(
            simplify_location(location, address),
            "2726 Riverside Drive, Cincinnati OH 45202",
        )

    def test_parse_dt_converts_utc_to_eastern(self):
        dt = parse_dt("2026-04-10T14:00:00+0000")
        self.assertIsNotNone(dt)
        self.assertEqual(dt.isoformat(), "2026-04-10T09:00:00-05:00")



    def test_extract_event_details_from_jsonld(self):
        html_doc = """
        <html><head><script type="application/ld+json">
        {
          "@context": "https://schema.org",
          "@type": "Event",
          "name": "Cincinnati Cars and Coffee",
          "startDate": "2026-03-07T08:00:00-05:00",
          "endDate": "2026-03-07T10:30:00-05:00",
          "location": {
            "@type": "Place",
            "name": "Crestview Hills Town Center",
            "address": {
              "@type": "PostalAddress",
              "streetAddress": "2791 Town Center Blvd",
              "addressLocality": "Crestview Hills",
              "addressRegion": "KY",
              "postalCode": "41017"
            }
          }
        }
        </script></head><body></body></html>
        """
        out = extract_event_details_from_jsonld(html_doc)
        self.assertIsNotNone(out)
        self.assertEqual(out["start_dt"].isoformat(), "2026-03-07T08:00:00-05:00")
        self.assertEqual(out["location"], "2791 Town Center Blvd, Crestview Hills KY 41017")

    def test_extract_facebook_page_identifier_variants(self):
        self.assertEqual(
            extract_facebook_page_identifier("https://www.facebook.com/Carscoffeewestchester1/"),
            "Carscoffeewestchester1",
        )
        self.assertEqual(
            extract_facebook_page_identifier("https://facebook.com/zakirasgarage?ref=bookmarks"),
            "zakirasgarage",
        )
        self.assertEqual(
            extract_facebook_page_identifier("https://www.facebook.com/profile.php?id=123456789012345"),
            "123456789012345",
        )

    def test_classify_facebook_pages_url_and_group_key(self):
        self.assertEqual(classify_facebook_pages_url("https://www.facebook.com/groups/cincycarsclub/"), "group")
        self.assertEqual(classify_facebook_pages_url("https://www.facebook.com/carsandcoffeecincy/"), "page")
        self.assertEqual(classify_facebook_pages_url("https://example.com/community"), "non_facebook")
        self.assertEqual(
            extract_facebook_group_key("https://www.facebook.com/groups/cincycarsclub/?ref=share"),
            "cincycarsclub",
        )


class AutomotiveBypassTests(unittest.TestCase):
    def test_to_event_items_allows_source_bypass_automotive_filter(self):
        cfg = {
            "home": {"lat": 39.1031, "lon": -84.512},
            "filters": {
                "lookahead_days": -1,
                "drop_past_days": 7,
                "local_max_miles": 200,
                "rally_max_miles": 600,
                "automotive_focus_keywords": ["car", "cars and coffee"],
                "non_automotive_exclude_keywords": ["job fair"],
                "trusted_event_platforms": ["facebook.com/events"],
            },
            "categorization": {
                "local_keywords": ["cars and coffee"],
                "rally_keywords": ["rally"],
            },
        }

        start_dt = datetime(2099, 3, 7, 14, 0, tzinfo=tz.gettz("America/New_York"))
        raw_events = [
            {
                "title": "Downtown runnaz meet #2",
                "start_dt": start_dt,
                "end_dt": start_dt.replace(hour=16),
                "location": "",
                "url": "https://p147-caldav.icloud.com/calendar",
                "source": "iCloud Published Calendar (ICS)",
                "bypass_automotive_filter": True,
            }
        ]

        out = to_event_items(raw_events, cfg, geocache={})
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].title, "Downtown runnaz meet #2")


if __name__ == "__main__":
    unittest.main()

class _FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class FacebookDiagnosticsTests(unittest.TestCase):
    def test_token_expired_disables_graph_source_without_spam(self):
        pages = [
            {"page_identifier": "carsandcoffeecincy", "page_url": "https://facebook.com/carsandcoffeecincy", "enabled": True},
            {"page_identifier": "anotherpage", "page_url": "https://facebook.com/anotherpage", "enabled": True},
        ]
        calls = {"count": 0}

        def fake_get(*args, **kwargs):
            calls["count"] += 1
            return _FakeResponse(400, {"error": {"code": 190, "error_subcode": 463, "message": "Session has expired"}})

        diagnostics = {}
        with patch("scripts.events_collector.FACEBOOK_ACCESS_TOKEN", "fake-token"), patch(
            "scripts.events_collector.FACEBOOK_GRAPH_RUNTIME", {"checked": True, "valid": True, "reason": "ok"}
        ), patch("scripts.events_collector.requests.get", side_effect=fake_get):
            with self.assertRaises(SourceDisabledError):
                collect_facebook_events_from_pages(pages, diagnostics=diagnostics)

        self.assertEqual(calls["count"], 1)
        self.assertEqual(diagnostics.get("reason"), "disabled_token_expired")

    def test_serpapi_facebook_discovery_parses_fixture_without_facebook_http_fetch(self):
        fixture = json.loads(Path("tests/fixtures/serpapi_facebook_events_sample.json").read_text())
        rows = fixture["rows"]

        with patch("scripts.events_collector.collect_facebook_event_urls_serpapi", return_value=rows), patch(
            "scripts.events_collector.fetch_facebook_event_via_graph", side_effect=AssertionError("Graph enrichment should be skipped")
        ):
            out = collect_facebook_events_serpapi_discovery(cfg={}, url_cache={}, diagnostics={})

        self.assertGreater(len(out), 0)
        self.assertTrue(any("facebook.com/events/" in e.get("url", "") for e in out))

    def test_zero_yield_reason_prefers_filter_diagnostics(self):
        reason = derive_zero_yield_reason(
            "Eventbrite Cincy",
            diagnostics={"raw_candidates": 5, "parse_failures": 0},
            source_drop_reasons=Counter({"outside_window_future": 3}),
        )
        self.assertEqual(reason, "filtered_out_by_window")

    def test_facebook_group_serpapi_collector_adds_group_metadata(self):
        pages = [
            {
                "page_url": "https://www.facebook.com/groups/cincycarsclub/",
                "enabled": True,
                "page_type": "group",
            }
        ]
        rows = [
            {
                "url": "https://www.facebook.com/events/1234567890/",
                "result": {
                    "link": "https://www.facebook.com/events/1234567890/",
                    "title": "Cars and Coffee Meetup",
                    "snippet": "Saturday, March 2 at 9 AM · Cincinnati",
                },
            }
        ]

        with patch("scripts.events_collector.SERPAPI_API_KEY", "fake-serpapi-key"), patch(
            "scripts.events_collector.load_facebook_targets", return_value={"group": pages, "page": [], "non_facebook": []}
        ), patch(
            "scripts.events_collector.collect_facebook_group_event_urls_serpapi", return_value=(rows, None, {"group_name": "Cincy Cars Club", "serp_urls": 5})
        ), patch(
            "scripts.events_collector.fetch_facebook_event_via_graph", side_effect=AssertionError("Graph enrichment should be skipped")
        ):
            out = collect_facebook_group_events_serpapi(source={}, cfg={}, url_cache={}, diagnostics={})

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].get("source"), "facebook_group_serpapi")
        self.assertEqual(out[0].get("source_group_key"), "cincycarsclub")
        self.assertIn("facebook.com/groups/cincycarsclub", out[0].get("source_group_url", ""))
        self.assertEqual(out[0].get("source_group_name"), "Cincy Cars Club")

    def test_main_calls_load_facebook_targets_with_force_reload(self):
        with patch("scripts.events_collector.load_facebook_targets", return_value={"page": [], "group": [], "non_facebook": []}) as mock_targets, patch(
            "scripts.events_collector.load_yaml", return_value={"sources": []}
        ), patch(
            "scripts.events_collector.load_json", return_value={"events": []}
        ), patch(
            "scripts.events_collector.is_automotive_event_safe", return_value=True
        ), patch(
            "scripts.events_collector.to_event_items", return_value=[]
        ), patch(
            "scripts.events_collector.dedupe_merge", return_value=[]
        ), patch(
            "scripts.events_collector.save_json"
        ), patch(
            "scripts.events_collector.write_csv"
        ), patch(
            "scripts.events_collector.update_apex_spreadsheet"
        ), patch(
            "scripts.events_collector.log"
        ):
            main()
        mock_targets.assert_called_once_with(force_reload=True)



    def test_main_preserves_bypass_source_events_through_final_filter(self):
        cfg = {
            "home": {"lat": 39.1031, "lon": -84.512},
            "filters": {
                "lookahead_days": -1,
                "drop_past_days": 7,
                "local_max_miles": 200,
                "rally_max_miles": 600,
                "automotive_focus_keywords": ["car"],
                "non_automotive_exclude_keywords": [],
                "trusted_event_platforms": [],
            },
            "categorization": {"local_keywords": ["cars and coffee"], "rally_keywords": ["rally"]},
            "sources": [{"type": "ics", "name": "iCloud Published Calendar (ICS)", "bypass_automotive_filter": True}],
        }

        kept_event = EventItem(
            title="Downtown runnaz meet #2",
            start_iso="2099-03-07T14:00:00-05:00",
            end_iso="2099-03-07T16:00:00-05:00",
            location="",
            city_state="",
            url="https://p147-caldav.icloud.com/calendar",
            source="iCloud Published Calendar (ICS)",
            category="local",
            miles_from_cincy=None,
            lat=None,
            lon=None,
            last_seen_iso="2099-03-01T00:00:00-05:00",
        )

        def fake_load_json(path, default=None):
            if str(path).endswith("events.json"):
                return {"events": []}
            return {}

        with patch("scripts.events_collector.load_facebook_targets", return_value={"page": [], "group": [], "non_facebook": []}), patch(
            "scripts.events_collector.load_yaml", return_value=cfg
        ), patch(
            "scripts.events_collector.load_json", side_effect=fake_load_json
        ), patch(
            "scripts.events_collector.to_event_items", return_value=[kept_event]
        ), patch(
            "scripts.events_collector.dedupe_merge", side_effect=lambda existing, incoming, metrics=None: list(incoming)
        ), patch(
            "scripts.events_collector.prune_past_events", side_effect=lambda rows, now: rows
        ), patch(
            "scripts.events_collector.save_json"
        ) as mock_save, patch(
            "scripts.events_collector.write_csv"
        ), patch(
            "scripts.events_collector.update_apex_spreadsheet"
        ), patch(
            "scripts.events_collector.log"
        ):
            main()

        payload_call = next(call for call in mock_save.call_args_list if str(call.args[0]).endswith("events.json"))
        payload = payload_call.args[1]
        self.assertEqual(payload["count"], 1)
        self.assertEqual(payload["events"][0]["title"], "Downtown runnaz meet #2")
    def test_load_facebook_targets_accepts_force_reload(self):
        with patch("scripts.events_collector.load_facebook_pages", return_value=[]):
            out = load_facebook_targets(force_reload=True)
        self.assertEqual(out, {"page": [], "group": [], "non_facebook": []})


class NewSourceIngestionTests(unittest.TestCase):
    def test_collect_ics_converts_webcal_and_applies_future_only(self):
        ics = """BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
SUMMARY:Past Cars Meetup
DTSTART:20200101T120000Z
DTEND:20200101T140000Z
END:VEVENT
BEGIN:VEVENT
SUMMARY:Future Cars Meetup
DTSTART:20990101T120000Z
DTEND:20990101T140000Z
END:VEVENT
END:VCALENDAR
"""

        class _Resp:
            status_code = 200
            content = ics.encode("utf-8")

            def raise_for_status(self):
                return None

        with patch("scripts.events_collector.requests.get", return_value=_Resp()) as mock_get:
            out = collect_ics({
                "name": "iCloud Published Calendar (ICS)",
                "url": "webcal://example.com/calendar.ics",
                "future_only": True,
            })

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["title"], "Future Cars Meetup")
        self.assertEqual(mock_get.call_args.args[0], "https://example.com/calendar.ics")

    def test_collect_ics_expands_rrule_occurrences(self):
        ics = """BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
SUMMARY:Recurring Cars Meetup
DTSTART:20990101T120000Z
DTEND:20990101T140000Z
RRULE:FREQ=DAILY;COUNT=3
END:VEVENT
END:VCALENDAR
"""

        class _Resp:
            status_code = 200
            content = ics.encode("utf-8")

            def raise_for_status(self):
                return None

        with patch("scripts.events_collector.requests.get", return_value=_Resp()):
            out = collect_ics(
                {
                    "name": "Recurring ICS",
                    "url": "https://example.com/recurring.ics",
                    "future_only": True,
                }
            )

        self.assertEqual(len(out), 3)
        self.assertEqual([e["start_dt"].day for e in out], [1, 2, 3])

    def test_collect_ics_rrule_honors_exdate_and_rdate(self):
        ics = """BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
SUMMARY:Cars Meetup with Exceptions
DTSTART:20990101T120000Z
DTEND:20990101T140000Z
RRULE:FREQ=DAILY;COUNT=2
EXDATE:20990102T120000Z
RDATE:20990105T120000Z
END:VEVENT
END:VCALENDAR
"""

        class _Resp:
            status_code = 200
            content = ics.encode("utf-8")

            def raise_for_status(self):
                return None

        with patch("scripts.events_collector.requests.get", return_value=_Resp()):
            out = collect_ics(
                {
                    "name": "Recurring ICS",
                    "url": "https://example.com/recurring.ics",
                    "future_only": True,
                }
            )

        self.assertEqual(len(out), 2)
        self.assertEqual([e["start_dt"].day for e in out], [1, 5])

    def test_parse_google_sheet_rows_maps_columns_and_defaults_date_only_time(self):
        rows = [
            ["Date", "Name", "City / Where You'd Be", "Link"],
            ["2099-05-01", "Cars & Coffee", "Cincinnati", "https://example.com/event"],
        ]
        out, stats = _parse_google_sheet_events_rows(rows, "Google Sheet Events Import", "Events")
        self.assertEqual(len(out), 1)
        self.assertEqual(stats["parsed_events"], 1)
        self.assertEqual(out[0]["title"], "Cars & Coffee")
        self.assertEqual(out[0]["start_dt"].hour, 9)
        self.assertEqual(out[0]["end_dt"].hour, 11)
        self.assertIn("Cincinnati", out[0]["location"])
