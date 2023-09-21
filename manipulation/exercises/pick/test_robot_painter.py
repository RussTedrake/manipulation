import unittest
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np


class TestRobotPainter(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(4)
    @timeout_decorator.timeout(1.0)
    def test_key_frames(self):
        """compose_circular_key_frames"""
        f = self.notebook_locals["compose_circular_key_frames"]
        X_WC = self.notebook_locals["X_WCenter"]
        X_WG = self.notebook_locals["X_WG"]
        radius = self.notebook_locals["radius"]
        thetas = self.notebook_locals["thetas"]
        # carry out computation
        output_frames = f(thetas, X_WC, X_WG)

        # test all key positions match radius
        for i, frame_i in enumerate(output_frames):
            if i == 0:
                # check if the first frame is the gripper's current frame
                dist = np.linalg.norm(
                    frame_i.translation() - X_WG.translation()
                )
                self.assertLessEqual(
                    dist, 1e-6, "first key frame position incorrect!"
                )
            else:
                # check if the radius is correct
                pos_cur = frame_i.translation()
                r_cur = np.linalg.norm(pos_cur - X_WC.translation())
                self.assertLessEqual(
                    abs(radius - r_cur),
                    1e-6,
                    "key frame positions incorrect!",
                )

                # check if +z of each frame points toward the center
                z_cur = frame_i.rotation().matrix()[0:3, 2]
                test_center = z_cur * radius + pos_cur
                center_err = np.linalg.norm(test_center - X_WC.translation())
                self.assertLessEqual(
                    center_err, 1e-6, "key frame orientations incorrect!"
                )
