import unittest
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np
import os


class TestAnalyticGrasp(unittest.TestCase):

    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(4)
    @timeout_decorator.timeout(10.)
    def test_antipodal_points(self):
        """Test find_antipodal_pts"""
        shape = self.notebook_locals["shape"]
        f = self.notebook_locals["find_antipodal_pts"]

        # Use a seed that can test for different Hessians
        np.random.seed(45)

        result_lst = []
        H_eig_lst = []

        for i in range(4):
            result, H_eig = f(shape)
            result_lst.append(result)
            H_eig_lst.append(H_eig)

        result_lst_eval = np.array(result_lst)
        H_eig_lst_eval = np.array(H_eig_lst)

        # yapf: disable
        result_lst_target = np.array([  # noqa
            [6.276351, 3.497493],
            [3.948776, 1.636278],
            [6.104002, 4.219046],
            [3.223685, 0.710411]])

        H_eig_lst_target = np.array([  # noqa
            [29612.82676, 35855.23483],
            [-1025.65590, -1312.08498],
            [-643.00117, 6819.32722],
            [-1249.67350, 27654.86062]])
        # yapf: enable

        self.assertLessEqual(
            np.linalg.norm(result_lst_target - result_lst_eval), 1e-4,
            "The optimal values are not returned correctly.")

        self.assertLessEqual(np.linalg.norm(H_eig_lst_target - H_eig_lst_eval),
                             1e-4,
                             "The Hessian values are not returned correctly.")
