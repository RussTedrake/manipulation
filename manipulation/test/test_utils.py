from datetime import date

from manipulation.mustard_depth_camera_example import MustardExampleSystem
from manipulation.utils import DrakeVersionGreaterThan, SystemHtml

# Test DrakeVersionGreaterThan
DrakeVersionGreaterThan(date(2023, 12, 1))

# Test SystemHtml
system = MustardExampleSystem()
SystemHtml(system)
