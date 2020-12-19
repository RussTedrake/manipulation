import unittest
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np
import os


class TestHybrid(unittest.TestCase):

    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(4)
    @timeout_decorator.timeout(30.)
    def test_antipodal_points(self):
        """Test compute_ctrl"""
        log = self.notebook_locals["log"]
        plant = self.notebook_locals["plant"]
        plant_context = self.notebook_locals["plant_context"]

        # Get final pose of the book.
        X_WB = plant.GetFreeBodyPose(plant_context,
                                     plant.GetBodyByName("book_body"))

        p_WB = X_WB.translation()

        # 1. Check upper bound on book pose.
        self.assertLessEqual(
            p_WB[0], 0.55, "Edge of the book is not between the gap. "
            "It is too far.")

        # 2. Check lower bound on book pose.
        self.assertLessEqual(
            -p_WB[0], -0.5, "Edge of the book is not betwee the gap. "
            "It is too close.")

        time = log.sample_times()
        traj = log.data()

        finger_length = 0.12

        p_x = traj[1, :] - finger_length * np.sin(traj[0, :])
        p_y = traj[2, :] - finger_length * np.cos(traj[0, :])

        for t in range(len(time)):
            self.assertLessEqual(
                -p_y[t], -0.045, "The gripper tip cannot be lower than the "
                "surface of the book.")
