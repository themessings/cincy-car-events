import unittest
from unittest.mock import patch

from scripts.events_collector import collect_google_sheet_events_import


class SheetImportPublicCsvFallbackTests(unittest.TestCase):
    def test_falls_back_to_extract_events_when_events_tab_unparseable_via_public_csv(self):
        source = {"name": "Google Sheet Events Import", "tab": "Events"}
        events_rows = [
            ["Date", "Name", "City / Where You'd Be", "Link"],
            ["2099-06-01", "Cars & Coffee", "Cincinnati", "https://example.com/event"],
        ]

        def _csv_rows(_sheet_id, tab_name):
            if tab_name == "Events":
                return [["title", "date", "start_time"]]
            if tab_name == "Extracted Events":
                return events_rows
            return []

        with patch("scripts.events_collector.APEX_IMPORT_SPREADSHEET_ID", "sheet-id"), patch(
            "scripts.events_collector._fetch_public_sheet_rows_csv", side_effect=_csv_rows
        ), patch("scripts.events_collector._fetch_sheet_rows_via_api", return_value=[]):
            out = collect_google_sheet_events_import(source, diagnostics={})

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["title"], "Cars & Coffee")


if __name__ == "__main__":
    unittest.main()
