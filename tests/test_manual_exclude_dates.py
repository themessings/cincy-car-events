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


class LineupCheck20261022Tests(unittest.TestCase):
    """The four listings pulled from the 2026-10-22 lineup, read from the real config."""

    def setUp(self):
        import yaml
        from pathlib import Path
        from scripts.events_collector import _manual_exclusions
        cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "config" / "sources.yml").read_text(encoding="utf-8"))
        self.rules = _manual_exclusions(cfg)

    def _hit(self, title, url, day):
        return _matching_exclusion({"title": title, "url": url, "start_iso": f"{day}T08:00:00-04:00"}, self.rules)

    def test_liberty_center_sunday_series_is_dropped(self):
        self.assertIsNotNone(self._hit("Liberty Center Cars & Coffee",
                                       "https://carsandcoffeeevents.com/events/liberty-center-cars-coffee", "2026-10-25"))

    def test_liberty_center_host_saturdays_stay(self):
        for url in ("https://carsandcoffeeevents.com/events/liberty-center-cars-and-coffee-2026-10-17",
                    "https://carsandcoffeeevents.com/events/cincy-cars-coffee-at-liberty-center"):
            self.assertIsNone(self._hit("Liberty Center Cars and Coffee", url, "2026-10-17"))

    def test_other_trunk_or_treats_stay(self):
        self.assertIsNotNone(self._hit("Trunk or Treat Car Show", "https://allevents.in/dayton/car-shows", "2026-10-24"))
        self.assertIsNone(self._hit("Graham Rahal Performance Trunk or Treat", "", "2026-10-24"))
        self.assertIsNone(self._hit("Indianapolis Cars and Coffee - Trunk-or-treat", "", "2026-10-24"))

    def test_four_bridges_only_loses_the_28th(self):
        self.assertIsNotNone(self._hit("Four Bridges Cruise-In", "", "2026-10-28"))
        self.assertIsNone(self._hit("Four Bridges Cruise-In", "", "2027-04-21"))


if __name__ == "__main__":
    unittest.main()
