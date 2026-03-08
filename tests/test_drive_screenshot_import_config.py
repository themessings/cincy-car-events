import os
import unittest
from unittest.mock import patch

from scripts import import_drive_event_screenshots as screenshot_import
from scripts.events_collector import get_google_credentials


class DriveScreenshotImportConfigTests(unittest.TestCase):
    def test_parse_args_prefers_apex_import_spreadsheet_id(self):
        with patch.dict(
            os.environ,
            {"APEX_IMPORT_SPREADSHEET_ID": "apex-sheet", "SHEET_ID": "legacy-sheet"},
            clear=False,
        ), patch("sys.argv", ["prog"]):
            args = screenshot_import.parse_args()

        self.assertEqual(args.sheet_id, "apex-sheet")

    def test_get_google_credentials_uses_gdrive_alias(self):
        with patch("scripts.events_collector.os.getenv") as mock_getenv, patch(
            "scripts.events_collector.service_account.Credentials.from_service_account_info"
        ) as mock_from_info:
            env_map = {
                "GOOGLE_SERVICE_ACCOUNT_FILE": "",
                "GOOGLE_SERVICE_ACCOUNT_JSON": "",
                "GDRIVE_SERVICE_ACCOUNT_JSON": '{"type":"service_account"}',
            }
            mock_getenv.side_effect = lambda k, d=None: env_map.get(k, d)
            mock_from_info.return_value = object()

            creds = get_google_credentials()

        self.assertIsNotNone(creds)
        mock_from_info.assert_called_once()


if __name__ == "__main__":
    unittest.main()
