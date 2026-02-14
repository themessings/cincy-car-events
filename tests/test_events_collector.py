import unittest
from datetime import datetime

from scripts.events_collector import (
    EventItem,
    dedupe_merge,
    evaluate_automotive_focus_event,
    extract_facebook_page_identifier,
    normalize_facebook_page_event,
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


if __name__ == "__main__":
    unittest.main()
