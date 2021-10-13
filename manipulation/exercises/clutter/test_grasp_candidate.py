import unittest
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np
import os
import open3d as o3d

from pydrake.all import RigidTransform

# Store X_lst_target as global for testing all the functions
# yapf: disable
X_lst_target = np.array([  # noqa
    [[-0.209311, -0.977586, +0.022690, +0.022568],
     [+0.966542, -0.210354, -0.146791, +0.023823],
     [+0.148274, -0.008794, +0.988907, +0.082323],
     [+0.000000, +0.000000, +0.000000, +1.000000]],
    [[-0.731169, +0.159814, +0.663214, -0.007744],
     [-0.580146, -0.657145, -0.481239, +0.032702],
     [+0.358919, -0.736627, +0.573199, +0.144046],
     [+0.000000, +0.000000, +0.000000, +1.000000]],
    [[-0.350573, -0.936270, +0.022282, +0.018658],
     [+0.931311, -0.351029, -0.097156, +0.034710],
     [+0.098786, -0.013308, +0.995020, +0.106885],
     [+0.000000, +0.000000, +0.000000, +1.000000]],
    [[-0.843675, +0.525630, -0.109206, -0.015267],
     [-0.468279, -0.820000, -0.329111, +0.043170],
     [-0.262540, -0.226524, +0.937955, +0.045414],
     [+0.000000, +0.000000, +0.000000, +1.000000]]])
# yapf: enable

test_indices = [10137, 21584, 7259, 32081]


class TestGraspCandidate(unittest.TestCase):

    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(4)
    @timeout_decorator.timeout(10.)
    def test_darboux_frame(self):
        """Test compute_darboux_frame"""
        pcd = self.notebook_locals["pcd"]
        kdtree = o3d.geometry.KDTreeFlann(pcd)
        f = self.notebook_locals["compute_darboux_frame"]

        X_lst_eval = []

        np.random.seed(11)
        for i in range(4):
            index = test_indices[i]
            RT = f(index, pcd, kdtree)
            X_lst_eval.append(RT.GetAsMatrix4())

        X_lst_eval = np.asarray(X_lst_eval)

        self.assertLessEqual(np.linalg.norm(X_lst_target - X_lst_eval), 1e-4,
                             "The Darboux frame is not correct")

        index = 5
        RT = f(index, pcd, kdtree)

        X_lst_order_eval = RT.GetAsMatrix4()

        # yapf: disable
        X_lst_order_target = np.array([  # noqa
            [+0.036684, -0.880547, -0.472537, +0.008844],
            [+0.937533, +0.194023, -0.288768, -0.002408],
            [+0.345957, -0.432426, +0.832659, +0.191187],
            [+0.000000, +0.000000, +0.000000, +1.000000]])
        # yapf: enable

        self.assertLessEqual(
            np.linalg.norm(X_lst_order_eval - X_lst_order_target), 1e-4,
            "Did you forget to sort the eigenvalues, "
            "or handle improper rotations?")

    @weight(4)
    @timeout_decorator.timeout(10.)
    def test_minimum_distance(self):
        """Test find_minimum_distance"""
        pcd = self.notebook_locals["pcd"]
        f = self.notebook_locals["find_minimum_distance"]

        # The following should return nan
        for i in [0, 2]:
            dist, X_new = f(pcd, RigidTransform(X_lst_target[i]))
            self.assertTrue(
                np.isnan(dist), "There is no value of y that results in "
                "no collision in the grid, but dist is not nan")
            self.assertTrue(
                isinstance(X_new, type(None)),
                "There is no value of y that results in "
                "no collision in the grid, but X_WGnew is"
                "not None.")

        # yapf: disable
        dist_new_target = np.array([  # noqa
            0.0035799752,
            0.0008069168])

        X_new_target = np.array([  # noqa
           [[-0.73117, +0.15981, +0.66321, -0.01573],
            [-0.58015, -0.65715, -0.48124, +0.06556],
            [+0.35892, -0.73663, +0.57320, +0.18088],
            [+0.00000, +0.00000, +0.00000, +1.00000]],
           [[-0.84368, +0.52563, -0.10921, -0.03571],
            [-0.46828, -0.82000, -0.32911, +0.07506],
            [-0.26254, -0.22652, +0.93796, +0.05422],
            [+0.00000, +0.00000, +0.00000, +1.00000]]])
        # yapf: enable

        dist_new_eval = []
        X_new_eval = []
        # The following should return numbers.
        for i in [1, 3]:
            dist, X_new = f(pcd, RigidTransform(X_lst_target[i]))
            self.assertTrue(
                not np.isnan(dist),
                "There is a valid value of y that results in "
                "no collision in the grid, but dist is nan")
            self.assertTrue(
                not isinstance(X_new, type(None)),
                "There is a valid value of y that results in no "
                "collision in the grid, but X_WGnew is None.")
            dist_new_eval.append(dist)
            X_new_eval.append(X_new.GetAsMatrix4())

        dist_new_eval = np.array(dist_new_eval)
        X_new_eval = np.array(X_new_eval)

        self.assertLessEqual(np.linalg.norm(dist_new_target - dist_new_eval),
                             1e-5, "The returned distance is not correct.")
        self.assertLessEqual(np.linalg.norm(X_new_target - X_new_eval), 1e-4,
                             "The returned transform is not correct.")

    @weight(4)
    @timeout_decorator.timeout(60.)
    def test_candidate_grasps(self):
        """Test compute_candidate_grasps"""
        pcd = self.notebook_locals["pcd"]
        f = self.notebook_locals["compute_candidate_grasps"]

        # yapf: disable
        X_lst_target = np.array([  # noqa
            [[-0.86670, +0.49867, -0.01296, -0.04684],
             [-0.49881, -0.86662, +0.01232, +0.07370],
             [-0.00508, +0.01714, +0.99984, +0.01943],
             [+0.00000, +0.00000, +0.00000, +1.00000]],
            [[+0.52811, -0.84916, +0.00468, +0.06930],
             [+0.83829, +0.52222, +0.15671, -0.04796],
             [-0.13552, -0.07884, +0.98763, +0.10482],
             [+0.00000, +0.00000, +0.00000, +1.00000]],
            [[-0.90546, +0.38488, +0.17889, -0.03838],
             [-0.40438, -0.65434, -0.63899, +0.05335],
             [-0.12889, -0.65092, +0.74812, +0.18382],
             [+0.00000, +0.00000, +0.00000, +1.00000]]])
        # yapf: enable

        grasp_candidates = f(pcd, candidate_num=3, random_seed=5)

        self.assertTrue(
            len(grasp_candidates) == 3,
            "Length of returned array is not correct.")

        X_lst_eval = []
        for i in range(len(grasp_candidates)):
            X_lst_eval.append(grasp_candidates[i].GetAsMatrix4())
        X_lst_eval = np.array(X_lst_eval)

        self.assertLessEqual(np.linalg.norm(X_lst_target - X_lst_eval), 1e-4,
                             "The returned grasp candidates are not correct.")
