import unittest

import numpy as np
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight


class TestHybrid(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(2)
    @timeout_decorator.timeout(30.0)
    def test_antipodal_points(self):
        """Test compute_ctrl"""
        log = self.notebook_locals["log"]
        plant = self.notebook_locals["plant"]
        plant_context = self.notebook_locals["plant_context"]

        # Get final pose of the book.
        X_WB = plant.GetFreeBodyPose(plant_context, plant.GetBodyByName("book"))

        p_WB = X_WB.translation()

        # 1. Check upper bound on book pose.
        self.assertLessEqual(
            p_WB[0],
            0.55,
            "Edge of the book is not between the gap. " "It is too far.",
        )

        # 2. Check lower bound on book pose.
        self.assertGreaterEqual(
            p_WB[0],
            0.35,
            "Edge of the book is not between the gap. " "It is too close.",
        )

        time = log.sample_times()
        traj = log.data()

        finger_length = 0.12

        p_x = traj[1, :] - finger_length * np.sin(traj[0, :])
        p_y = traj[2, :] - finger_length * np.cos(traj[0, :])

        for t in range(len(time)):
            self.assertGreaterEqual(
                p_y[t],
                0.035,
                "The gripper tip cannot be lower than the " "surface of the book.",
            )
