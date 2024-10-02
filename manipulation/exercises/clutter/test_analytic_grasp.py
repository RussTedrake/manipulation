import unittest

import numpy as np
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight
from pydrake.all import Variable


class TestAnalyticGrasp(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(4)
    @timeout_decorator.timeout(20.0)
    def test_antipodal_points(self):
        """Test find_antipodal_pts"""
        shape = self.notebook_locals["shape"]
        f = self.notebook_locals["find_antipodal_pts"]

        # Test if points are antipodal by comparing normal vectors.
        for i in range(20):
            result, H_eig = f(shape)

            t = Variable("t")
            surface = shape(t)
            J_x = surface[0].Differentiate(t)
            J_y = surface[1].Differentiate(t)

            dx_1 = J_x.Evaluate({t: result[0]})
            dy_1 = J_y.Evaluate({t: result[0]})

            dx_2 = J_x.Evaluate({t: result[1]})
            dy_2 = J_y.Evaluate({t: result[1]})

            pts = np.array([dx_1, dy_1]), np.array([dx_2, dy_2])

            v_1 = pts[0] / np.linalg.norm(pts[0])
            v_2 = pts[1] / np.linalg.norm(pts[1])

            n_1 = np.array([-v_1[1], v_1[0]])
            n_2 = -np.array([-v_2[1], v_2[0]])

            self.assertLessEqual(
                np.linalg.norm(n_1 - n_2),
                1e-4,
                "Tested points are not antipodal.",
            )
