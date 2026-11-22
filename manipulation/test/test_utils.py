import unittest
import warnings
from datetime import date
from unittest import mock

from manipulation.mustard_depth_camera_example import MustardExampleSystem
from manipulation.utils import DrakeVersionGreaterThan, SystemHtml


class TestUtils(unittest.TestCase):
    def test_drake_version(self):
        # The installed Drake must be recognized and new enough.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            DrakeVersionGreaterThan(date(2023, 12, 1))

    def _check(self, version: str, minimum_date: date) -> None:
        with mock.patch("manipulation.utils.GetDrakeVersion", return_value=version):
            DrakeVersionGreaterThan(minimum_date)

    def test_drake_version_dates(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self._check("1.58.0", date(2026, 10, 15))
            self._check("0.0.20261120", date(2026, 11, 1))
            self._check("source", date(2100, 1, 1))
            # Releases newer than the table count as at least its last date.
            self._check("99.0.0", date(2026, 1, 1))
        with self.assertRaisesRegex(RuntimeError, "update your Drake"):
            self._check("1.50.0", date(2026, 10, 1))
        with self.assertRaisesRegex(RuntimeError, "update your Drake"):
            self._check("0.0.20250101", date(2026, 1, 1))
        with self.assertWarnsRegex(UserWarning, "Cannot tell"):
            self._check("99.0.0", date(2100, 1, 1))
        with self.assertWarnsRegex(UserWarning, "Unrecognized"):
            self._check("1.2.3rc1", date(2023, 1, 1))

    def test_system_html(self):
        system = MustardExampleSystem()
        # Test that this doesn't raise an exception
        SystemHtml(system)


if __name__ == "__main__":
    unittest.main()
