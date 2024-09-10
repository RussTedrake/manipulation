import unittest

import numpy as np
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight
from pydrake.all import RigidTransform, RotationMatrix


class TestRigidTransforms(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(1)
    @timeout_decorator.timeout(1.0)
    def test_X_WB(self):
        """Testing X_WB"""
        f = self.notebook_locals["compute_X_WB"]

        # construct a test case
        theta1, theta2, theta3 = np.pi / 3.0, np.pi / 6.0, np.pi / 4.0
        R_WA = RotationMatrix.MakeXRotation(theta1)
        R_AB = RotationMatrix.MakeZRotation(theta2)
        R_CB = RotationMatrix.MakeYRotation(theta3)

        X_WA = RigidTransform(R_WA, [0.1, 0.2, 0.5])
        X_AB = RigidTransform(R_AB, [0.3, 0.4, 0.1])
        X_CB = RigidTransform(R_CB, [0.5, 0.9, 0.7])

        test_X_WB = f(X_WA, X_AB, X_CB)
        true_X_WB = X_WA.multiply(X_AB)

        test_result = test_X_WB.multiply(true_X_WB.inverse())
        test_result = test_result.GetAsMatrix4()
        self.assertTrue(np.allclose(test_result, np.eye(4)))

    @weight(1)
    @timeout_decorator.timeout(1.0)
    def test_X_CW(self):
        """Testing X_CW"""
        f = self.notebook_locals["compute_X_CW"]

        # construct a test case
        theta1, theta2, theta3 = np.pi / 3.0, np.pi / 6.0, np.pi / 4.0
        R_WA = RotationMatrix.MakeXRotation(theta1)
        R_AB = RotationMatrix.MakeZRotation(theta2)
        R_CB = RotationMatrix.MakeYRotation(theta3)

        X_WA = RigidTransform(R_WA, [0.1, 0.2, 0.5])
        X_AB = RigidTransform(R_AB, [0.3, 0.4, 0.1])
        X_CB = RigidTransform(R_CB, [0.5, 0.9, 0.7])

        true_X_WC = X_WA.multiply(X_AB).multiply(X_CB.inverse())
        true_X_CW = true_X_WC.inverse()

        test_X_CW = f(X_WA, X_AB, X_CB)

        test_result = true_X_CW.multiply(test_X_CW.inverse())
        test_result = test_result.GetAsMatrix4()
        self.assertTrue(np.allclose(test_result, np.eye(4)))

    @weight(2)
    @timeout_decorator.timeout(1.0)
    def test_grasp_pose(self):
        """Testing grasp pose"""
        f = self.notebook_locals["design_grasp_pose"]
        X_WO = self.notebook_locals["X_WO"]

        test_X_OG, test_X_WG = f(X_WO)

        R_OG = RotationMatrix(np.array([[0, 0, 1], [0, -1, 0], [1, 0, 0]]).T)
        p_OG = [0, 0.02, 0]
        X_OG = RigidTransform(R_OG, p_OG)
        X_WG = X_WO.multiply(X_OG)

        f(X_WO)

        test_result = test_X_OG.multiply(X_OG.inverse())
        test_result = test_result.GetAsMatrix4()
        self.assertTrue(np.allclose(test_result, np.eye(4)))

        test_result = test_X_WG.multiply(X_WG.inverse())
        test_result = test_result.GetAsMatrix4()
        self.assertTrue(np.allclose(test_result, np.eye(4)))
