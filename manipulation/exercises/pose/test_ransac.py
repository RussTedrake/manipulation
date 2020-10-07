import unittest
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np
from pydrake.all import (RigidTransform, RotationMatrix, RandomGenerator)
from copy import deepcopy


def ransac_solution(point_cloud,
                    model_fit_func,
                    tolerance=1e-3,
                    max_iterations=500):
    """
    Args:
      point_cloud is (N, 3) numpy array
      tolerance is a float
      max_iterations is a (small) integer
      model_fit_func: the function to fit the model (point clouds)

    Returns:
      (4,) numpy array
    """
    best_ic = 0  # inlier count
    best_model = None  # plane equation ((4,) array)
    N = point_cloud.shape[0]  # number of points

    sample_size = 3

    point_cloud_1 = np.ones((N, 4))
    point_cloud_1[:, :3] = point_cloud

    generator = RandomGenerator(5)

    for i in range(max_iterations):
        s = point_cloud[np.random.RandomState(generator()).randint(
            N, size=sample_size)]
        m = model_fit_func(s)
        abs_distances = np.abs(np.dot(m, point_cloud_1.T))  # 1 x N
        inliner_count = np.sum(abs_distances < tolerance)

        if inliner_count > best_ic:
            best_ic = inliner_count
            best_model = m

    return best_ic, best_model


class TestRANSAC(unittest.TestCase):

    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals
        self.ransac_student = self.notebook_locals['ransac']
        self.fit_plane = self.notebook_locals['fit_plane']

    @weight(4)
    @timeout_decorator.timeout(2.0)
    def test_ransac(self):
        """Test ransac method"""
        simple_cloud = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 1, 0],
                                 [0.2, 0.5, 0], [2, 4, 1]])

        _, plane_student = self.ransac_student(simple_cloud, self.fit_plane,
                                               1e-5, 100)
        _, plane_solution = ransac_solution(simple_cloud, self.fit_plane, 1e-5,
                                            100)
        self.assertTrue(np.array_equal(plane_solution, plane_student),
                        "ransac implementation incorrect")

    @weight(2)
    @timeout_decorator.timeout(2.0)
    def test_outlier_removal(self):
        """Test outlier removal"""
        # check whether outlier is in student's answer
        remove_plane = self.notebook_locals['remove_plane']
        bunny_w_plane = self.notebook_locals['bunny_w_plane'].T
        bunny_w_plane_copy = deepcopy(bunny_w_plane)

        student_answer = remove_plane(bunny_w_plane,
                                      self.ransac_student,
                                      tol=1e-4)

        outliers = []
        plane_equation = np.array([0.0, 0.0, 1.0, 0.0])
        dst = np.dot(bunny_w_plane_copy,
                     plane_equation[0:3]) + plane_equation[3]
        for i in range(len(dst)):
            if abs(dst[i]) < 1e-6:
                outliers.append(bunny_w_plane_copy[i])
        outliers = np.asarray(outliers)
        for pnt in student_answer:
            self.assertTrue(not (pnt in outliers),
                            "found unremoved points on the planar surface")
