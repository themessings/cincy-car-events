import unittest

from scripts.events_collector import closest_major_city, event_state


class ClosestCityMarketTests(unittest.TestCase):
    """Slide headings come from Closest City, so a heading must never hold a
    town its readers would not call part of that market (Joel, 2026-10-01:
    "liberty IN is not Dayton")."""

    def test_liberty_indiana_is_cincinnati_not_dayton(self):
        # 40mi from Dayton, 43mi from Cincinnati.
        self.assertEqual(closest_major_city(39.6356, -84.9311, "IN"), "Cincinnati, OH")

    def test_without_a_state_the_nearest_city_still_wins(self):
        self.assertEqual(closest_major_city(39.6356, -84.9311), "Dayton, OH")

    def test_river_metros_keep_their_cross_state_suburbs(self):
        self.assertEqual(closest_major_city(39.0909, -84.8494, "IN"), "Cincinnati, OH")  # Lawrenceburg
        self.assertEqual(closest_major_city(39.0273, -84.5849, "KY"), "Cincinnati, OH")  # Crestview Hills
        self.assertEqual(closest_major_city(38.2856, -85.8241, "IN"), "Louisville, KY")  # New Albany

    def test_ohio_border_town_goes_to_an_ohio_city(self):
        # Fort Recovery, OH: 50mi Fort Wayne, 55mi Dayton.
        self.assertEqual(closest_major_city(40.4131, -84.7766, "OH"), "Dayton, OH")

    def test_indiana_border_town_goes_to_an_indiana_city(self):
        # Winchester, IN: 51mi Dayton, 63mi Fort Wayne, 70mi Indianapolis.
        self.assertEqual(closest_major_city(40.1720, -84.9814, "IN"), "Fort Wayne, IN")

    def test_far_border_town_keeps_its_only_real_neighbour(self):
        # Steubenville, OH is a Pittsburgh suburb in all but name.
        self.assertEqual(closest_major_city(40.3698, -80.6340, "OH"), "Pittsburgh, PA")

    def test_event_state_reads_the_address_not_the_title(self):
        self.assertEqual(event_state("Rockin' Randall's, 5 Main St, Liberty, IN 47353"), "IN")
        self.assertEqual(event_state("", "Liberty, IN"), "IN")
        self.assertEqual(event_state("Cars, in the park"), "")


if __name__ == "__main__":
    unittest.main()
