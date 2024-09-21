import unittest

import numpy as np
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight


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
        radius = self.notebook_locals["radius"]
        thetas = self.notebook_locals["thetas"]
        num_key_frames = self.notebook_locals["num_key_frames"]
        # carry out computation
        output_frames = f(thetas, X_WC, radius)

        # check number of key frames
        self.assertEqual(
            len(output_frames), num_key_frames, "wrong number of key frames"
        )

        # test all key positions match radius
        for i, frame_i in enumerate(output_frames):
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
            self.assertLessEqual(center_err, 1e-6, "key frame orientations incorrect!")

            # check if we're rotating counterclockwise
            if i > 0:
                X_G1W = output_frames[i - 1].inverse()
                X_WG2 = frame_i
                X_G1G2 = X_G1W @ X_WG2
                rotation_angle = X_G1G2.rotation().ToAngleAxis().angle()
                self.assertAlmostEqual(
                    thetas[i] - thetas[i - 1],
                    rotation_angle,
                    2,
                    "rotation angle incorrect",
                )

                rotation_direction = X_G1G2.rotation().ToAngleAxis().axis()[1]
                self.assertAlmostEqual(
                    rotation_direction, -1, 2, "rotating in the wrong direction"
                )
