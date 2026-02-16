import json
import unittest
from collections import Counter
from datetime import datetime
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
        with patch("scripts.events_collector.FACEBOOK_ACCESS_TOKEN", "fake-token"), patch("scripts.events_collector.requests.get", side_effect=fake_get):
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
