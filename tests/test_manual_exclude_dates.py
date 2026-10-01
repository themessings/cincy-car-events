import unittest

from scripts.events_collector import _matching_exclusion

RULE = {"title": "Cars at Madison Square", "dates": ["2026-10-03"], "reason": "private event"}


class ManualExcludeDatesTests(unittest.TestCase):
    def test_drops_only_the_cancelled_day(self):
        ev = {"title": "Cars at Madison Square", "url": "", "start_iso": "2026-10-03T09:00:00-04:00"}
        self.assertIs(_matching_exclusion(ev, [RULE]), RULE)

    def test_keeps_every_other_week(self):
        ev = {"title": "Cars at Madison Square", "url": "", "start_iso": "2026-10-10T09:00:00-04:00"}
        self.assertIsNone(_matching_exclusion(ev, [RULE]))

    def test_rule_without_dates_still_drops_every_day(self):
        rule = {"title": "No Limits 937"}
        ev = {"title": "No Limits 937 Cars & Coffee", "url": "", "start_iso": "2026-10-10T09:00:00-04:00"}
        self.assertIs(_matching_exclusion(ev, [rule]), rule)

    def test_keep_weekdays_spares_the_real_day(self):
        rule = {"title": "Starbucks in Westerville", "keep_weekdays": ["Sun"]}
        sat = {"title": "Columbus Cars & Coffee – Starbucks in Westerville", "url": "", "start_iso": "2026-10-03T08:00:00-04:00"}
        sun = {"title": "Columbus Cars & Coffee – Starbucks in Westerville", "url": "", "start_iso": "2026-10-04T08:00:00-04:00"}
        self.assertIs(_matching_exclusion(sat, [rule]), rule)
        self.assertIsNone(_matching_exclusion(sun, [rule]))


if __name__ == "__main__":
    unittest.main()
