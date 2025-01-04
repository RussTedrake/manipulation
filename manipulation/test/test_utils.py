import unittest
from datetime import date

from manipulation.mustard_depth_camera_example import MustardExampleSystem
from manipulation.utils import DrakeVersionGreaterThan, SystemHtml


class TestUtils(unittest.TestCase):
    def test_drake_version(self):
        # Test that this doesn't raise an exception
        DrakeVersionGreaterThan(date(2023, 12, 1))

    def test_system_html(self):
        system = MustardExampleSystem()
        # Test that this doesn't raise an exception
        SystemHtml(system)


if __name__ == "__main__":
    unittest.main()
