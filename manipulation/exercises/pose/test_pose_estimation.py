import unittest
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np
import os

from manipulation import FindResource


class TestPoseEstimation(unittest.TestCase):

    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(4)
    @timeout_decorator.timeout(2.)
    def test_filtered_pointcloud(self):
        """Test filtered pointcloud"""
        # Round to 3 decimal places to avoid floating point sensitivity
        f = np.around(self.notebook_locals["scene_pcl_np_filtered"].T,
                      decimals=3).tolist()
        f_target = np.load(FindResource("exercises/pose/pcl_filtered.npy"))
        f_target = np.around(f_target, decimals=3).tolist()

        f = set(tuple(row) for row in f)
        f_target = set(tuple(row) for row in f_target)

        inliers = len(f.intersection(f_target))
        outliers = len(f) - inliers

        self.assertLessEqual(
            outliers, 80,
            'You have too many false positives (too many outliers)')
        self.assertLessEqual(
            len(f_target) - inliers, 80,
            'You have too many false negatives (not enough inliers)')
