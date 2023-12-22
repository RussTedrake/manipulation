import unittest

import numpy as np
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight


class TestPlanarManipulator(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(2)
    @timeout_decorator.timeout(1.0)
    def test_forward_kinematics(self):
        """Test forward kinematics"""
        f = self.notebook_locals["forward_kinematics"]

        # yapf: disable
        f_target = np.array(  # noqa
            [[1.50622542e+00, -3.24106720e-01],
             [-4.00204670e-01, 1.22797901e+00],
             [-4.18796913e-03, -2.41251153e-01],
             [-1.89613836e+00, -4.50764733e-01],
             [-7.33351352e-04, -8.50706389e-05],
             [-1.42443194e+00, -7.95951898e-01],
             [-1.67781495e+00, 1.00784000e+00],
             [8.51078060e-02, 1.91798594e+00],
             [-2.78251693e-01, 1.11286876e-01],
             [1.87023541e+00, -6.91226373e-01],
             [-1.75671560e+00, -9.03640182e-01],
             [3.03370245e-01, 8.67113852e-03],
             [1.80627262e+00, -2.77795237e-01],
             [-1.13832198e+00, 8.42243133e-01],
             [1.65770062e-01, -1.15899848e-01],
             [-5.51982846e-02, 1.78460814e-02],
             [-1.08696195e-01, -9.11768635e-02],
             [-3.79324860e-01, 1.70457760e+00],
             [9.85086393e-01, -4.96909466e-01],
             [-3.16654300e-01, -1.25453082e+00]])
        # yapf: enable

        np.random.seed(7)
        n_rands = 20
        f_eval = []
        for i in range(n_rands):
            q = 2.0 * np.pi * np.random.rand(2)
            f_eval.append(f(q))

        f_eval = np.array(f_eval).squeeze()

        self.assertLessEqual(
            np.linalg.norm(f_target - np.stack(f_eval)),
            1e-6,
            "The forward kinematics implementation is not correct",
        )

    @weight(2)
    @timeout_decorator.timeout(1.0)
    def test_jacobian(self):
        """Test jacobian"""
        f = self.notebook_locals["Jacobian"]

        # yapf: disable
        f_target = np.array(  # noqa
            [[[3.24106720e-01, 7.85406073e-01],
              [1.50622542e+00, 6.18980857e-01]],
             [[-1.22797901e+00, -8.50579796e-01],
              [-4.00204670e-01, 5.25845994e-01]],
             [[2.41251153e-01, 1.03395593e-01],
              [-4.18796913e-03, -9.94640313e-01]],
             [[4.50764733e-01, 4.43724710e-01],
              [-1.89613836e+00, -8.96163144e-01]],
             [[8.50706389e-05, 9.93381322e-01],
              [-7.33351352e-04, 1.14863173e-01]],
             [[7.95951898e-01, -1.06804620e-01],
              [-1.42443194e+00, -9.94280028e-01]],
             [[-1.00784000e+00, -3.27615457e-01],
              [-1.67781495e+00, -9.44811152e-01]],
             [[-1.91798594e+00, -9.46570970e-01],
              [8.51078060e-02, 3.22495579e-01]],
             [[-1.11286876e-01, 8.62366612e-01],
              [-2.78251693e-01, -5.06284334e-01]],
             [[6.91226373e-01, 2.72315699e-01],
              [1.87023541e+00, 9.62207961e-01]]])
        # yapf: enable

        np.random.seed(7)
        n_rands = 10
        f_eval = []
        for i in range(n_rands):
            q = 2.0 * np.pi * np.random.rand(2)
            f_eval.append(f(q))

        f_eval = np.array(f_eval).squeeze()

        self.assertLessEqual(
            np.linalg.norm(f_target - np.stack(f_eval)),
            1e-6,
            "The Jacobian implementation is not correct",
        )
