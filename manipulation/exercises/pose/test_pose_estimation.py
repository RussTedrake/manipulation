import unittest

import numpy as np
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight


class TestPoseEstimation(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(4)
    @timeout_decorator.timeout(2.0)
    def test_filtered_pointcloud(self):
        """Test filtered pointcloud"""
        X_WO = self.notebook_locals["X_WO"]
        p_WA = self.notebook_locals["scene_pcl_np_filtered"]
        num_points = p_WA.shape[1]
        p_OA = X_WO.inverse().multiply(p_WA)
        # bounding box from collision geometry:
        # bbox_min = np.array([-0.0375, -0.025, 0])
        # bbox_max = np.array([0.0375, 0.025, 0.05])
        # relax this a bit (the visual geometry is slightly bigger)
        bbox_min = np.array([-0.04, -0.03, -0.01])
        bbox_max = np.array([0.04, 0.03, 0.06])

        in_bbox = np.all(
            (bbox_min[:, None] <= p_OA) & (p_OA <= bbox_max[:, None]), axis=0
        )

        inliers = np.sum(in_bbox)
        outliers = num_points - inliers

        self.assertLessEqual(
            outliers,
            80,
            "You have too many false positives (too many outliers)",
        )
        self.assertGreaterEqual(
            inliers,
            num_points - 170,
            "You have too many false negatives (not enough inliers)",
        )
