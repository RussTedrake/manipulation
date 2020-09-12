import unittest
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight
import pydrake.symbolic as ps
import numpy as np


class TestSimpleQP(unittest.TestCase):

    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(2)
    @timeout_decorator.timeout(1.)
    def test_simple_qp(self):
        """Test Simple QP problem"""
        f = self.notebook_locals["result_submission"]

        # To test, prepare a random matrix so that we don't show the
        # answers directly.
        np.random.seed(7)
        crypt_mat = np.random.rand(8, 3)
        f_eval = crypt_mat.dot(f).squeeze()

        f_target = np.array([
            0.09174558, 0.18417568, 0.14758899, 0.28134268, 0.14121386,
            0.24749159, 0.32098331, 0.28508303
        ])

        self.assertLessEqual(np.linalg.norm(f_target - np.stack(f_eval)), 1e-6,
                             'The answer to the QP is not correct.')

    @weight(4)
    @timeout_decorator.timeout(1.)
    def test_solve_qp(self):
        """Test solve_qp problem"""
        f = self.notebook_locals["solve_qp"]

        # Test 1. Test input-output correctness for succeeding cases.
        seed_lst_pass = [0, 15, 29, 64, 75, 96]

        f_eval = []
        for i in range(len(seed_lst_pass)):
            np.random.seed(seed_lst_pass[i])

            Q = np.random.rand(3, 3)
            A_eq = np.random.rand(2, 3)
            b_eq = np.random.rand(2)
            A_ineq = np.random.rand(3, 3)
            b_ineq = np.random.rand(3)
            b_bb = np.random.rand(3)

            f_eval.append(f(Q, A_eq, b_eq, A_ineq, b_ineq, b_bb))

        f_eval = np.array(f_eval).squeeze()

        f_target = np.array([  # noqa
            [-0.40310909, 0.26455561, 0.0609625],
            [0.04465451, 0.24459557, -0.09485914],
            [-0.16886317, -0.12903977, 0.57050779],
            [-0.02712253, 0.20608795, 0.130202],
            [0.07784826, 0.19994872, 0.07972953],
            [-0.21492276, 0.32833558, 0.15242427]
        ])

        self.assertLessEqual(
            np.linalg.norm(f_target - np.stack(f_eval)), 1e-6,
            'solve_qp does not have correct input-output responses')

        # Test 2. If the QP fails, then it should return a ValueError

        seed_lst_fail = [5, 7, 10, 23, 35, 44, 59]

        for i in range(len(seed_lst_fail)):
            np.random.seed(seed_lst_fail[i])

            Q = np.random.rand(3, 3)
            A_eq = np.random.rand(2, 3)
            b_eq = np.random.rand(2)
            A_ineq = np.random.rand(3, 3)
            b_ineq = np.random.rand(3)
            b_bb = np.random.rand(3)

            with self.assertRaises(ValueError):
                f(Q, A_eq, b_eq, A_ineq, b_ineq, b_bb)

        # Test 3. Try another dimension of x to see if it passes.
        seed_lst_pass = [57, 74]

        f_eval = []
        for i in range(len(seed_lst_pass)):
            np.random.seed(seed_lst_pass[i])

            Q = np.random.rand(4, 4)
            A_eq = np.random.rand(3, 4)
            b_eq = np.random.rand(3)
            A_ineq = np.random.rand(4, 4)
            b_ineq = np.random.rand(4)
            b_bb = np.random.rand(4)

            f_eval.append(f(Q, A_eq, b_eq, A_ineq, b_ineq, b_bb))

        f_eval = np.array(f_eval).squeeze()

        f_target = np.array([  # noqa
            [-0.06737614, -0.19623779, 0.07046111, 0.170199],
            [0.1581591, 0.32354142, 0.48367475, -0.21180427]
        ])

        self.assertLessEqual(
            np.linalg.norm(f_target - np.stack(f_eval)), 1e-6,
            'solve_qp does not have correct input-output responses '
            'or does not generalize across dimensions')
