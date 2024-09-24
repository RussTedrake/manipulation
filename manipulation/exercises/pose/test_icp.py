import unittest

import numpy as np
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight
from pydrake.all import RigidTransform, RollPitchYaw, RotationMatrix


def least_squares_transform(scene, model):
    X_BA = RigidTransform()
    mu_m = np.mean(model, axis=1)
    mu_s = np.mean(scene, axis=1)

    W = (scene.T - mu_s).T @ (model.T - mu_m)
    U, Sigma, Vh = np.linalg.svd(W)
    R_star = U.dot(Vh)

    if np.linalg.det(R_star) < 0:
        Vh[-1] *= -1
        R_star = U.dot(Vh)

    t_star = mu_s - R_star.dot(mu_m)

    X_BA.set_rotation(RotationMatrix(R_star))
    X_BA.set_translation(t_star)

    return X_BA


def generate_arbitrary_transform(seed):
    R_BA = RollPitchYaw(seed * 7.363, seed * 1.35, seed * 5.47)
    p_BA = np.mod([seed * 2.13, seed * 3.3, seed * 5.225], 0.1) - 0.05
    X_BA = RigidTransform(R_BA, p_BA)
    return X_BA


class TestICP(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

        self.model = self.notebook_locals["pointcloud_model"]

    @weight(3)
    @timeout_decorator.timeout(5.0)
    def test_least_square(self):
        """Test least square transform"""
        f = self.notebook_locals["least_squares_transform"]
        nearest_neighbors = self.notebook_locals["nearest_neighbors"]

        # Run a battery of tests deterministically, making sure that the
        # condition number of the data matrix (the singular values of W) are
        # not too small.
        for i in range(10):
            X_BA = generate_arbitrary_transform(i)
            self.scene = X_BA.multiply(self.model)
            distances, indices = nearest_neighbors(self.scene, self.model)

            X_BA_test = f(self.scene, self.model[:, indices])
            X_BA_true = least_squares_transform(self.scene, self.model[:, indices])

            # check answer
            result = X_BA_true.inverse().multiply(X_BA_test)

            self.assertTrue(
                np.allclose(result.GetAsMatrix4(), np.eye(4)),
                "least square transform is incorrect",
            )

        # Test improper matrix
        result = f(self.model, np.diag([1, 1, -1]) @ self.model)
        self.assertTrue(
            result.rotation().IsValid(),
            "Your method returned an improper rotation matrix",
        )

    @weight(3)
    @timeout_decorator.timeout(10.0)
    def test_icp(self):
        """Test icp implementation"""
        f = self.notebook_locals["icp"]
        nearest_neighbors = self.notebook_locals["nearest_neighbors"]

        # It should be sufficient to test only one for ICP.
        X_BA = generate_arbitrary_transform(7)
        self.scene = X_BA.multiply(self.model)
        X_BA_test, mean_error_test, num_iters_test = f(self.scene, self.model)

        distances, indices = nearest_neighbors(
            self.scene, X_BA_test.multiply(self.model)
        )

        mean_error = np.mean(distances)
        assert mean_error < 1.3e-2, "ICP test failed with mean error {}".format(
            mean_error
        )
