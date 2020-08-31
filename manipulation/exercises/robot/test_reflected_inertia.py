import unittest
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight
import pydrake.symbolic as ps
import numpy as np


class TestSimplePendulumWithGearbox(unittest.TestCase):

    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(5)
    @timeout_decorator.timeout(1.)
    def test_dynamics(self):
        """Test state space dynamics"""
        f = self.notebook_locals['pendulum_with_motor_dynamics']

        # Don't use local p since students might have different params.
        p = {"m": 1.0, "l": 0.5, "g": 9.81, "N": 160, "I": 3.46e-4}

        f_target = np.array([[7.79918792e-01, 7.66080504e+00],
                             [9.77989512e-01, 9.10363857e+00],
                             [7.20511334e-02, 4.45713832e+00],
                             [6.79229996e-01, 1.38617385e+01],
                             [6.59363469e-02, 4.86183460e+00],
                             [2.13385354e-01, 7.51773706e+00],
                             [2.48992276e-02, 1.01181812e+01],
                             [2.30302879e-01, 9.19761678e+00],
                             [1.33169446e-01, 8.77027126e+00],
                             [6.69013241e-01, 7.85009758e+00],
                             [4.90765889e-01, 6.43240569e+00],
                             [3.65890386e-01, 1.44728761e+01],
                             [3.13994677e-01, 9.68534802e+00],
                             [4.52842933e-01, 6.05424583e+00],
                             [3.70351083e-01, 7.73613524e+00],
                             [4.12991829e-01, 1.55689682e+01],
                             [7.41118873e-01, 7.32350209e+00],
                             [6.34379869e-01, 8.96351019e+00],
                             [1.42688056e-03, 1.40375517e+00],
                             [5.24345597e-01, 1.18791650e+01]])

        np.random.seed(7)
        n_rands = 20
        f_eval = []
        for i in range(n_rands):
            x = np.random.rand(2)
            u = np.random.rand(1)
            f_eval.append(ps.Evaluate(f(x, u, p)))

        f_eval = np.array(f_eval).squeeze()

        self.assertLessEqual(
            np.linalg.norm(f_target - np.stack(f_eval)), 1e-6,
            'The pendulum with motor dynamics are not corret.')
