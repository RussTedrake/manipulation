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
        f = self.notebook_locals["scene_pcl_np_filtered"].tolist()
        f_target = np.load(
            FindResource("exercises/pose/pcl_filtered.npy")).tolist()

        f = set(tuple(row) for row in f)
        f_target = set(tuple(row) for row in f_target)

        inliers = len(set(f).intersection(f_target))
        outliers = len(f) - inliers

        self.assertLessEqual(
            outliers, 80,
            'You have too many false positives (too many outliers)')
        self.assertLessEqual(
            len(f_target) - inliers, 80,
            'You have too many false negatives (not enough inliers)')

    @weight(1)
    @timeout_decorator.timeout(1.)
    def test_xyz_loose(self):
        """Test XYZ with loose bounds"""
        f = self.notebook_locals["xyz"]
        target = 0.02

        for i in range(3):
            self.assertLessEqual(abs(f[i]), target,
                                 "XYZ"[i] + " translation is too off")

    @weight(1)
    @timeout_decorator.timeout(1.)
    def test_xyz_tight(self):
        """Test XYZ with tight bounds"""
        f = self.notebook_locals["xyz"]
        target = 0.005

        for i in range(3):
            self.assertLessEqual(abs(f[i]), target,
                                 "XYZ"[i] + " translation is too off")

    @weight(1)
    @timeout_decorator.timeout(1.)
    def test_rpy_loose(self):
        """Test RPY with loose bounds"""
        f = self.notebook_locals["rpy"]
        target = 0.5

        for i in range(3):
            self.assertLessEqual(abs(f[i]), target,
                                 "RPY"[i] + " translation is too off")

    @weight(1)
    @timeout_decorator.timeout(1.)
    def test_rpy_tight(self):
        """Test RPY with tight bounds"""
        f = self.notebook_locals["rpy"]
        target = 0.05

        for i in range(3):
            self.assertLessEqual(abs(f[i]), target,
                                 "RPY"[i] + " translation is too off")
