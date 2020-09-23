import unittest
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np
from pydrake.all import RigidTransform, RotationMatrix
from sklearn.neighbors import NearestNeighbors


class TestICP(unittest.TestCase):

    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals
        # random rotation
        theta_z, theta_y, theta_x = np.random.rand(3) * np.pi / 4
        rot_z = RotationMatrix.MakeZRotation(theta_z)
        rot_y = RotationMatrix.MakeZRotation(theta_y)
        rot_x = RotationMatrix.MakeZRotation(theta_x)
        R_BA = rot_z.multiply(rot_y).multiply(rot_x)
        # random translation
        dx, dy, dz = np.random.rand(3) * 1.0
        p_BA = [dx, dy, dz]

        X_BA = RigidTransform(R_BA, p_BA)

        self.model = self.notebook_locals["pointcloud_model"]
        self.scene = X_BA.multiply(self.model.T).T

    @weight(3)
    @timeout_decorator.timeout(1.)
    def test_least_square(self):
        """Test least square transform"""
        f = self.notebook_locals["least_squares_transform"]
        X_BA_test = f(self.scene, self.model)

        # correct implementation
        mu_m = np.mean(self.model, axis=0)
        mu_s = np.mean(self.scene, axis=0)

        W = (self.scene - mu_s).T.dot(self.model - mu_m)
        U, Sigma, Vh = np.linalg.svd(W)
        R_star = U.dot(Vh)

        if np.linalg.det(R_star) < 0:
            Vh[-1] *= -1
            R_star = U.dot(Vh)

        t_star = mu_s - R_star.dot(mu_m)
        X_BA_true = RigidTransform(RotationMatrix(R_star), t_star)

        # check answer
        result = X_BA_true.inverse().multiply(X_BA_test)

        self.assertTrue(np.allclose(result.GetAsMatrix4(), np.eye(4)),
                        'least square transform is incorrect')

    @weight(3)
    @timeout_decorator.timeout(1.)
    def test_icp(self):
        """Test icp implementation"""
        f = self.notebook_locals["icp"]
        nearest_neighbors = self.notebook_locals["nearest_neighbors"]
        least_squares_transform = self.notebook_locals[
            "least_squares_transform"]

        X_BA_test, mean_error_test, num_iters_test = f(self.scene, self.model)

        num_iters = 0
        mean_error = 0.0
        max_iterations = 20
        tolerance = 1e-3
        prev_error = 0
        X_BA = RigidTransform()

        while True:
            num_iters += 1
            distances, indices = nearest_neighbors(
                self.scene,
                X_BA.multiply(self.model.T).T)
            X_BA = RigidTransform(
                least_squares_transform(self.scene, self.model[indices]))
            mean_error = np.mean(distances)
            if abs(mean_error -
                   prev_error) < tolerance or num_iters >= max_iterations:
                break
            prev_error = mean_error

        result = X_BA.multiply(X_BA_test.inverse())
        self.assertTrue(np.allclose(result.GetAsMatrix4(), np.eye(4)),
                        'least square transform is incorrect')
