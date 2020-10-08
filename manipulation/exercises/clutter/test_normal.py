import unittest
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np
from pydrake.all import (RigidTransform, RotationMatrix, RandomGenerator)
from copy import deepcopy


def estimate_normal_by_nearest_pixels(X_WC, pC, bbox, mask, uv_step=10):
    """
    Compute the surface normals from the nearest pixels (by a sliding window)
    Input:
        X_WC: RigidTransform of the camera in world frame
        pC: 3D points computed from the depth image in the camera frame
        uv_step: recommended step size for the sliding window (see codes below)
    Output:
        normals: a list of RigidTransforms of the normal frames in world frame.
                 The +z axis of the normal frame is the normal vector, it should
                 points outward (towards the camera)
    """
    normals = []
    u_bound, v_bound = bbox(mask)
    for u in range(u_bound[0], u_bound[1], uv_step):
        for v in range(v_bound[0], v_bound[1], uv_step):
            # center of the window
            center = [u, v]
            u_length = 3
            v_length = 3
            # side of the window
            u_range = np.arange(center[0] - u_length, center[0] + u_length + 1)
            v_range = np.arange(center[1] - v_length, center[1] + v_length + 1)

            # collect nearest pixels from the image
            def pC_uv(u, v):
                return pC[v + u * 640]

            pC_near = []
            for ui in u_range:
                for vi in v_range:
                    pC_near.append(pC_uv(ui, vi))
            pC_star = pC_uv(u, v)
            prel = pC_near - pC_star
            W = np.matmul(prel.T, prel)
            w, V = np.linalg.eigh(W)
            # local fit
            X_CN = RigidTransform(R=RotationMatrix(np.fliplr(V)), p=pC_star)
            X_WN = X_WC.multiply(X_CN)
            X_WN = X_WN.GetAsMatrix4()
            # flip normals
            cam_vec = X_WC.GetAsMatrix4()[0:3, 3] - X_WN[0:3, 3]
            test = np.dot(X_WN[0:3, 2], cam_vec)
            if test < 0:
                X_WN[0:3, 0:3] = X_WN[0:3, 0:3] * (-1)
            normals.append(RigidTransform(X_WN))
    return normals


class TestNormal(unittest.TestCase):

    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(4)
    @timeout_decorator.timeout(60.0)
    def test_normal(self):
        """Testing the normal vectors"""
        env = self.notebook_locals['env']
        f = self.notebook_locals['estimate_normal_by_nearest_pixels']
        pC = self.notebook_locals['pC']
        mask = self.notebook_locals['mask']
        bbox = self.notebook_locals['bbox']

        student_sol = f(env.X_WC, pC, uv_step=10)
        reference_sol = estimate_normal_by_nearest_pixels(env.X_WC,
                                                          pC,
                                                          bbox,
                                                          mask,
                                                          uv_step=10)

        for X1, X2 in zip(student_sol, reference_sol):
            test = X1.multiply(X2.inverse())
            self.assertTrue(
                np.allclose(test.GetAsMatrix4(), np.eye(4), atol=1e-3),
                'normal vector is incorrect')
