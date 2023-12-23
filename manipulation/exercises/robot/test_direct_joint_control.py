import unittest

import numpy as np
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight


class TestDirectJointControl(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(5)
    @timeout_decorator.timeout(1.0)
    def test_position(self):
        """Test iiwa_position_measured"""
        f = self.notebook_locals["teleop_2d_direct"]

        q_cmd = np.array([0.1, 0.2, 0.3])
        station, context = f(interactive=False, q_cmd=q_cmd)
        station.ValidateContext(context)

        commanded_pos = station.GetOutputPort("iiwa.position_commanded").Eval(context)

        self.assertLessEqual(
            np.linalg.norm(commanded_pos - q_cmd),
            1e-3,
            "wrong commanded position",
        )
