import unittest
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np
from manipulation.utils import LoadDataResource


def chamfer_dist(pc_a, pc_b):
    """
    pc_a of Size(N, 3)
    pc_b of Size(M, 3)
    """
    diff = np.linalg.norm(pc_a[:, None] - pc_b[None], axis=2) ** 2
    dist = np.mean(np.min(diff, axis=0)) + np.mean(np.min(diff, axis=1))
    return dist


class TestSegmentationAndGrasp(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(4)
    @timeout_decorator.timeout(10.0)
    def test_get_merged_masked_pcd(self):
        """Test find_antipodal_pts"""
        predictions = self.notebook_locals["predictions"]
        cameras = self.notebook_locals["cameras"]
        get_merged_masked_pcd = self.notebook_locals["get_merged_masked_pcd"]

        rgb_ims = [c.rgb_im for c in cameras]
        depth_ims = [c.depth_im for c in cameras]
        project_depth_to_pC_funcs = [c.project_depth_to_pC for c in cameras]
        X_WCs = [c.X_WC for c in cameras]

        pcd_eval = get_merged_masked_pcd(
            predictions, rgb_ims, depth_ims, project_depth_to_pC_funcs, X_WCs
        )
        pcd_pts_eval = np.asarray(pcd_eval.xyzs()[:])
        pcd_colors_eval = np.asarray(pcd_eval.rgbs()[:])
        pcd_pts_eval = pcd_pts_eval.T
        num_points_eval = pcd_pts_eval.shape[0]

        data_target = np.load(
            LoadDataResource("segmentation_and_grasp_soln.npz")
        )
        pcd_pts_target = data_target["points"]
        data_target["colors"]
        num_points_target = pcd_pts_target.shape[0]

        # Allow some deviation in the number of points
        self.assertLessEqual(
            np.linalg.norm(num_points_target - num_points_eval),
            200,
            "Wrong number of points returned.",
        )

        # Make sure the sizes match
        min_num_pts = min(num_points_eval, num_points_target)

        self.assertLessEqual(
            chamfer_dist(
                pcd_pts_target[:min_num_pts, :], pcd_pts_eval[:min_num_pts, :]
            ),
            1e-4,
            "Point cloud points are not close enough to the solution values.",
        )
