import unittest

import numpy as np
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight


def normalize(arr):
    rng = arr.max() - arr.min()
    amin = arr.min()
    return (arr - amin) * 255 / rng


class TestMask(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(3)
    @timeout_decorator.timeout(60.0)
    def test_mask(self):
        """Testing deproject_pW_to_image"""
        env = self.notebook_locals["env"]

        # compute the student solution
        cx, cy, fx, fy = env.get_intrinsics()
        f = self.notebook_locals["deproject_pW_to_image"]
        p_W_mustard = self.notebook_locals["p_W_mustard"]
        X_WC = self.notebook_locals["X_WC"]
        mask_student = f(p_W_mustard, cx, cy, fx, fy, X_WC)
        mask_sol = env.mask

        self.assertTrue(
            mask_student.shape == mask_sol.shape,
            "the shape of the generated mask is incorrect",
        )

        # normalize student's mask
        mask_student = normalize(mask_student)
        mask_sol = normalize(mask_sol.astype(np.uint8))
        # quantify image difference
        difference = np.sum(np.abs(mask_student - mask_sol))
        self.assertTrue(difference < 0.1 * 480 * 640, "Computed mask is incorrect")
