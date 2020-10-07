import unittest
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np
from pydrake.all import (RigidTransform, RotationMatrix, RandomGenerator,
                         UniformlyRandomRotationMatrix)


def least_squares_transform(scene, model):
    X_BA = RigidTransform()
    mu_m = np.mean(model, axis=0)
    mu_s = np.mean(scene, axis=0)

    W = (scene - mu_s).T.dot(model - mu_m)
    U, Sigma, Vh = np.linalg.svd(W)
    R_star = U.dot(Vh)

    if np.linalg.det(R_star) < 0:
        Vh[-1] *= -1
        R_star = U.dot(Vh)

    t_star = mu_s - R_star.dot(mu_m)

    X_BA.set_rotation(RotationMatrix(R_star))
    X_BA.set_translation(t_star)

    return X_BA


def generate_random_transform(seed_num):
    generator = RandomGenerator(seed_num)
    R_BA = UniformlyRandomRotationMatrix(generator)
    p_BA = np.random.RandomState(generator()).rand(3)
    X_BA = RigidTransform(R_BA, p_BA)
    return X_BA


class TestICP(unittest.TestCase):

    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

        self.model = self.notebook_locals["pointcloud_model"]

    @weight(3)
    @timeout_decorator.timeout(5.)
    def test_least_square(self):
        """Test least square transform"""
        f = self.notebook_locals["least_squares_transform"]
        nearest_neighbors = self.notebook_locals["nearest_neighbors"]

        # Run a batter of tests determinisitcally, making sure the test cases
        # include an impropoer rotation case.
        for i in range(10):
            X_BA = generate_random_transform(i)
            self.scene = X_BA.multiply(self.model.T).T
            distances, indices = nearest_neighbors(self.scene, self.model)

            X_BA_test = f(self.scene, self.model[indices])
            X_BA_true = least_squares_transform(self.scene, self.model[indices])

            # check answer
            result = X_BA_true.inverse().multiply(X_BA_test)

            self.assertTrue(np.allclose(result.GetAsMatrix4(), np.eye(4)),
                            'least square transform is incorrect')

    @weight(3)
    @timeout_decorator.timeout(5.)
    def test_icp(self):
        """Test icp implementation"""
        f = self.notebook_locals["icp"]
        nearest_neighbors = self.notebook_locals["nearest_neighbors"]

        # It should be sufficient to test only one for ICP.
        X_BA = generate_random_transform(7)
        self.scene = X_BA.multiply(self.model.T).T
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
            X_BA = least_squares_transform(self.scene, self.model[indices])
            mean_error = np.mean(distances)
            if abs(mean_error
                   - prev_error) < tolerance or num_iters >= max_iterations:
                break
            prev_error = mean_error

        result = X_BA.multiply(X_BA_test.inverse())
        self.assertTrue(np.allclose(result.GetAsMatrix4(), np.eye(4)),
                        'icp implementation is incorrect')
