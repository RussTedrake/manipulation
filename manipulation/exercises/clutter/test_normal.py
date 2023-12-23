import unittest

import numpy as np
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight

from manipulation.utils import FindResource


class TestNormal(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(4)
    @timeout_decorator.timeout(60.0)
    def test_normal(self):
        """Testing the normal vectors"""
        env = self.notebook_locals["env"]
        f = self.notebook_locals["estimate_normal_by_nearest_pixels"]
        pC = self.notebook_locals["pC"]

        student_sol = f(env.X_WC, pC, uv_step=10)
        reference_sol = np.load(FindResource("exercises/clutter/normal_solution.npy"))

        self.assertTrue(
            len(student_sol) == len(reference_sol),
            "the number of the normals is incorrect",
        )

        for X, X_sol in zip(student_sol, reference_sol):
            # check only points within the bounding box
            pos_x, pos_y, pos_z = X_sol[0:3, 3]
            if abs(pos_x) < 1.0 and abs(pos_y) < 1.0:
                sol_v = X_sol[0:3, 2]
                # check only the normal vector, not the entire frame
                student_v = X.GetAsMatrix4()[0:3, 2]
                # normalize the vector in case it is not normalized
                student_v = student_v / np.linalg.norm(student_v)
                test = np.dot(student_v, sol_v)
                self.assertTrue(
                    test > 0.7,
                    "error at the point: \n {} \
                    \n normal is: {} \
                    \n solution normal is: {}".format(
                        [pos_x, pos_y, pos_z], student_v, sol_v
                    ),
                )
