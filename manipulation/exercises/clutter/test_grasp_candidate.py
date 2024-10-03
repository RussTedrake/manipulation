import unittest

import numpy as np
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight
from pydrake.all import RigidTransform
from scipy.spatial import KDTree

# Store X_lst_target as global for testing all the functions
# yapf: disable
X_lst_target = np.array([  # noqa
    [[-0.20932715, -0.97757709, +0.02291735, +0.02256790],
     [+0.96633077, -0.21039143, -0.14812255, +0.02382296],
     [+0.14962283, -0.00886034, +0.98870343, +0.08232251]],
    [[-0.74793190, +0.15975761, +0.64426345, -0.00774401],
     [-0.56756133, -0.65722209, -0.49591660, +0.03270167],
     [+0.34419772, -0.73657084, +0.58222961, +0.14404616]],
    [[-0.35055721, -0.93627107, +0.02249645, +0.01865858],
     [+0.93124354, -0.35102496, -0.09781062, +0.03471036],
     [+0.09947407, -0.01333854, +0.99495077, +0.10688476]],
    [[-0.84367001, +0.52562118, -0.10928528, -0.01526701],
     [-0.46823874, -0.82000375, -0.32916006, +0.04316992],
     [-0.26262784, -0.22653088, +0.93792874, +0.04541413]]])
# yapf: enable

test_indices = [10137, 21584, 7259, 32081]


class TestGraspCandidate(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(4)
    @timeout_decorator.timeout(10.0)
    def test_darboux_frame(self):
        """Test compute_darboux_frame"""
        pcd = self.notebook_locals["pcd"]
        kdtree = KDTree(pcd.xyzs().T)
        f = self.notebook_locals["compute_darboux_frame"]

        # NOTE!
        # You can rotate about the y-axis by 180 degrees and still
        # get the correct answer.

        for i in range(4):
            index = test_indices[i]
            RT = f(index, pcd, kdtree)
            RT_desired = RigidTransform(X_lst_target[i])
            alt = X_lst_target[i].copy()
            alt[:, [0, 2]] = -alt[:, [0, 2]]
            RT_desired_alt = RigidTransform(alt)
            self.assertTrue(
                RT.IsNearlyEqualTo(RT_desired, 0.1)
                or RT.IsNearlyEqualTo(RT_desired_alt, 0.1),
                "Your transform doesn't match the expected transform",
            )

        index = 5
        RT = f(index, pcd, kdtree)

        # yapf: disable
        RT_desired = RigidTransform(np.array([  # noqa
            [+0.03674694, -0.88056642, -0.47249594, +0.00884530],
            [+0.93758428, +0.19399482, -0.28862041, -0.00240727],
            [+0.34581122, -0.43239886, +0.83273393, +0.19118717]]))
        RT_desired_alt = RigidTransform(np.array([  # noqa
            [-0.03674694, -0.88056642, +0.47249594, +0.00884530],
            [-0.93758428, +0.19399482, +0.28862041, -0.00240727],
            [-0.34581122, -0.43239886, -0.83273393, +0.19118717]]))
        # yapf: enable

        self.assertTrue(
            RT.IsNearlyEqualTo(RT_desired, 0.01)
            or RT.IsNearlyEqualTo(RT_desired_alt, 0.01),
            "Did you forget to sort the eigenvalues, " "or handle improper rotations?",
        )

    @weight(4)
    @timeout_decorator.timeout(60.0)
    def test_minimum_distance(self):
        """Test find_minimum_distance"""
        pcd = self.notebook_locals["pcd"]
        f = self.notebook_locals["find_minimum_distance"]

        # The following should return nan
        for i in [0, 2]:
            dist, X_new = f(pcd, RigidTransform(X_lst_target[i]))
            self.assertTrue(
                np.isnan(dist),
                "There is no value of y that results in "
                "no collision in the grid, but dist is not nan",
            )
            self.assertTrue(
                isinstance(X_new, type(None)),
                "There is no value of y that results in "
                "no collision in the grid, but X_WGnew is"
                "not None.",
            )

        # yapf: disable
        dist_new_target = np.array([  # noqa
            0.0035799752,
            0.0008069168])

        X_new_target = np.array([  # noqa
           [[-0.74793190, +0.15975761, +0.64426345, -0.01573189],
            [-0.56756133, -0.65722209, -0.49591660, +0.06556277],
            [+0.34419772, -0.73657084, +0.58222961, +0.18087470]],
           [[-0.84367001, +0.52562118, -0.10928528, -0.03570783],
            [-0.46823874, -0.82000375, -0.32916006, +0.07505895],
            [-0.26262784, -0.22653088, +0.93792874, +0.05422366]]])
        # yapf: enable

        dist_new_eval = []
        X_new_eval = []
        # The following should return numbers.
        for i in [1, 3]:
            dist, X_new = f(pcd, RigidTransform(X_lst_target[i]))
            self.assertTrue(
                not np.isnan(dist),
                "There is a valid value of y that results in "
                "no collision in the grid, but dist is nan",
            )
            self.assertTrue(
                not isinstance(X_new, type(None)),
                "There is a valid value of y that results in no "
                "collision in the grid, but X_WGnew is None.",
            )
            dist_new_eval.append(dist)
            X_new_eval.append(X_new.GetAsMatrix34())

        dist_new_eval = np.array(dist_new_eval)
        X_new_eval = np.array(X_new_eval)

        self.assertLessEqual(
            np.linalg.norm(dist_new_target - dist_new_eval),
            1e-5,
            "The returned distance is not correct.",
        )
        self.assertLessEqual(
            np.linalg.norm(X_new_target - X_new_eval),
            1e-4,
            "The returned transform is not correct.",
        )

    @weight(4)
    @timeout_decorator.timeout(60.0)
    def test_candidate_grasps(self):
        """Test compute_candidate_grasps"""
        pcd = self.notebook_locals["pcd_downsampled"]
        compute_candidate_grasps = self.notebook_locals["compute_candidate_grasps"]
        find_minimum_distance = self.notebook_locals["find_minimum_distance"]
        check_collision = self.notebook_locals["check_collision"]
        check_nonempty = self.notebook_locals["check_nonempty"]

        grasp_candidates = compute_candidate_grasps(pcd, candidate_num=3, random_seed=5)

        self.assertTrue(
            len(grasp_candidates) == 3,
            "Length of returned array is not correct.",
        )

        for X_WP in grasp_candidates:
            distance, X_WP_new = find_minimum_distance(pcd, X_WP)
            self.assertLessEqual(
                distance, 1e-2
            ), "The returned grasp candidates are not minimum distance"
            self.assertTrue(
                check_collision(pcd, X_WP)
            ), "The returned grasp candidates have collisions"
            self.assertTrue(
                check_nonempty(pcd, X_WP)
            ), "The returned grasp candidates are not empty"
