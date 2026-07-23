import os
import unittest
from unittest.mock import patch

import scripts.events_collector as ec
from scripts import facebook_feed_scraper as fb_scraper
from scripts.facebook_feed_scraper import (
    FeedScrapeResult,
    canonical_event_url,
    harvest_event_hrefs,
    is_login_wall,
    load_storage_state,
)

EVENT_HTML = """
<html><head>
<meta property="og:title" content="Cars &amp; Coffee Cincinnati"/>
<script type="application/ld+json">
{"@type":"Event","name":"Cars & Coffee Cincinnati",
 "startDate":"2026-08-01T09:00:00-04:00","endDate":"2026-08-01T11:00:00-04:00",
 "location":{"@type":"Place","name":"Newport, KY",
  "address":{"@type":"PostalAddress","streetAddress":"1 Levee Way",
   "addressLocality":"Newport","addressRegion":"KY","postalCode":"41071"}}}
</script>
</head><body></body></html>
"""


class PureHelperTests(unittest.TestCase):
    def test_canonical_event_url_variants(self):
        self.assertEqual(
            canonical_event_url("https://www.facebook.com/events/1234567890/?ref=feed"),
            "https://www.facebook.com/events/1234567890/",
        )
        self.assertEqual(
            canonical_event_url("https://m.facebook.com/events/999/permalink/1/"),
            "https://www.facebook.com/events/999/",
        )
        self.assertEqual(
            canonical_event_url("https://www.facebook.com/events/event.php?eid=555"),
            "https://www.facebook.com/events/555/",
        )

    def test_canonical_event_url_rejects_non_events(self):
        for href in [
            "https://www.facebook.com/events/create",
            "https://www.facebook.com/events/birthdays",
            "https://www.facebook.com/marketplace/item/42/",
            "",
            None,
        ]:
            self.assertEqual(canonical_event_url(href), "")

    def test_harvest_event_hrefs_dedupes_and_filters(self):
        hrefs = [
            "https://www.facebook.com/events/111/?ref=a",
            "https://www.facebook.com/events/111/?ref=b",  # dup id
            "https://www.facebook.com/events/222/",
            "https://www.facebook.com/events/create",       # noise
            "https://example.com/whatever",                 # noise
        ]
        self.assertEqual(
            harvest_event_hrefs(hrefs),
            [
                "https://www.facebook.com/events/111/",
                "https://www.facebook.com/events/222/",
            ],
        )

    def test_is_login_wall(self):
        self.assertTrue(is_login_wall("https://www.facebook.com/login/?next=/"))
        self.assertTrue(is_login_wall("https://www.facebook.com/checkpoint/12/"))
        self.assertFalse(is_login_wall("https://www.facebook.com/?sk=h_chr"))

    def test_load_storage_state_forms(self):
        # raw JSON
        self.assertEqual(load_storage_state('{"cookies": []}'), {"cookies": []})
        # base64 JSON
        import base64
        b64 = base64.b64encode(b'{"cookies": [1]}').decode()
        self.assertEqual(load_storage_state(b64), {"cookies": [1]})
        # nothing usable
        self.assertIsNone(load_storage_state(""))
        self.assertIsNone(load_storage_state("not json, not base64 !!!"))


class CollectorGatingTests(unittest.TestCase):
    @patch.dict(os.environ, {"ENABLE_FACEBOOK_SCROLL": "0"}, clear=False)
    def test_disabled_by_flag_is_noop(self):
        diag = {}
        out = ec.collect_facebook_events_scroll({"name": "FB"}, url_cache={}, diagnostics=diag)
        self.assertEqual(out, [])
        self.assertEqual(diag.get("reason"), "disabled_by_flag")

    @patch.dict(os.environ, {"ENABLE_FACEBOOK_SCROLL": "1", "FACEBOOK_SESSION_STATE": "", "FACEBOOK_SESSION_STATE_PATH": ""}, clear=False)
    def test_enabled_without_session_reports_reason(self):
        diag = {}
        out = ec.collect_facebook_events_scroll({"name": "FB"}, url_cache={}, diagnostics=diag)
        self.assertEqual(out, [])
        self.assertEqual(diag.get("reason"), "disabled_missing_session")

    @patch.dict(os.environ, {"ENABLE_FACEBOOK_SCROLL": "1"}, clear=False)
    def test_parses_events_from_scraped_pages(self):
        url = "https://www.facebook.com/events/1234567890/"
        fake = FeedScrapeResult(
            event_urls=[url],
            pages={url: EVENT_HTML},
            discovered=1,
            scrolls_done=3,
        )
        url_cache = {}
        diag = {}
        with patch.object(fb_scraper, "load_storage_state", return_value={"cookies": []}), \
             patch.object(fb_scraper, "scrape_feed", return_value=fake):
            out = ec.collect_facebook_events_scroll(
                {"name": "Facebook Feed Scroll", "scrolls": 3}, url_cache=url_cache, diagnostics=diag
            )
        self.assertEqual(len(out), 1)
        ev = out[0]
        self.assertEqual(ev["title"], "Cars & Coffee Cincinnati")
        self.assertEqual(ev["source"], "Facebook Feed Scroll")
        self.assertIsNotNone(ev["start_dt"])
        self.assertIn("Newport", ev["location"])
        # cached for next run
        self.assertIn(url, url_cache)
        self.assertTrue(url_cache[url]["event"]["title"])

    @patch.dict(os.environ, {"ENABLE_FACEBOOK_SCROLL": "1"}, clear=False)
    def test_empty_feed_sets_reason(self):
        fake = FeedScrapeResult(event_urls=[], pages={}, discovered=0, reason="no_event_links_in_feed")
        diag = {}
        with patch.object(fb_scraper, "load_storage_state", return_value={"cookies": []}), \
             patch.object(fb_scraper, "scrape_feed", return_value=fake):
            out = ec.collect_facebook_events_scroll({"name": "FB"}, url_cache={}, diagnostics=diag)
        self.assertEqual(out, [])
        self.assertEqual(diag.get("reason"), "no_event_links_in_feed")


if __name__ == "__main__":
    unittest.main()
