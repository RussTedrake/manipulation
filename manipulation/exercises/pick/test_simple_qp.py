import unittest

import numpy as np
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight


class TestSimpleQP(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(4)
    @timeout_decorator.timeout(1.0)
    def test_simple_qp(self):
        """Test Simple QP problem"""
        f = self.notebook_locals["result_submission"]

        # To test, prepare a random matrix so that we don't show the
        # answers directly.
        np.random.seed(7)
        crypt_mat = np.random.rand(8, 3)
        f_eval = crypt_mat.dot(f).squeeze()

        f_target = np.array(
            [
                0.08640122,
                0.20073211,
                0.16338357,
                0.28908734,
                0.15244127,
                0.27599083,
                0.35008611,
                0.31416134,
            ]
        )

        self.assertLessEqual(
            np.linalg.norm(f_target - np.stack(f_eval)),
            1e-6,
            "The answer to the QP is not correct.",
        )
