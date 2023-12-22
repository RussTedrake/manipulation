import unittest

import numpy as np
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight


class TestStochasticOptimization(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals
        self.gradient_descent = self.notebook_locals["gradient_descent"]

    @weight(2)
    @timeout_decorator.timeout(15.0)
    def test_exact_gradient(self):
        """Test exact gradient function"""
        np.random.seed(7)
        exact_gradient = self.notebook_locals["exact_gradient"]
        f_eval = self.gradient_descent(
            0.1, exact_gradient, initial_x=np.array([2.0, 2.0]), iter=5
        )
        f_eval = np.array(f_eval)

        f_target = np.array(
            [
                [2.000000, 2.000000, 2.466667],
                [1.790000, 1.850000, 1.961246],
                [1.707546, 1.712750, 1.723588],
                [1.653825, 1.584424, 1.539064],
                [1.614487, 1.463857, 1.384269],
                [1.584082, 1.350302, 1.250706],
            ]
        )

        self.assertLessEqual(
            np.linalg.norm(f_target - f_eval),
            1e-2,
            "You have wrong exact gradients.",
        )

    @weight(2)
    @timeout_decorator.timeout(15.0)
    def test_approximated_gradient(self):
        """Test approximated gradient function"""
        np.random.seed(7)
        approximated_gradient = self.notebook_locals["approximated_gradient"]
        f_eval = self.gradient_descent(
            0.1, approximated_gradient, initial_x=np.array([2.0, 2.0]), iter=5
        )
        f_eval = np.array(f_eval)

        f_target = np.array(
            [
                [2.000000, 2.000000, 2.466667],
                [1.279693, 2.198529, 2.209571],
                [1.279187, 2.192247, 2.200421],
                [1.268846, 2.192274, 2.195393],
                [1.268799, 2.100353, 2.067566],
                [1.232492, 2.078929, 2.020906],
            ]
        )

        self.assertLessEqual(
            np.linalg.norm(f_target - f_eval),
            1e-2,
            "You have wrong approximated gradients.",
        )

    @weight(1)
    @timeout_decorator.timeout(15.0)
    def test_approximated_gradient_with_baseline(self):
        """Test approximated gradient with baseline function"""
        np.random.seed(7)
        approximated_gradient_with_baseline = self.notebook_locals[
            "approximated_gradient_with_baseline"
        ]

        def baseline(x):
            return 5

        def reduced_function(x, rate):
            return approximated_gradient_with_baseline(x, rate, baseline)

        f_eval = self.gradient_descent(
            0.1, reduced_function, initial_x=np.array([2.0, 2.0]), iter=5
        )
        f_eval = np.array(f_eval)

        f_target = np.array(
            [
                [2.000000, 2.000000, 2.466667],
                [1.493827, 2.139510, 2.214994],
                [1.497885, 2.189906, 2.289990],
                [1.383463, 2.190205, 2.244283],
                [1.383293, 1.854385, 1.788513],
                [1.509619, 1.928927, 1.927494],
            ]
        )

        self.assertLessEqual(
            np.linalg.norm(f_target - f_eval),
            1e-2,
            "You have wrong approixmated gradients" "with baseline",
        )
