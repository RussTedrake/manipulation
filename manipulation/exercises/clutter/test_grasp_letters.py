import unittest

import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight


class TestLetterGrasp(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(1)
    @timeout_decorator.timeout(15.0)
    def test_final_pose(self):
        context = self.notebook_locals["context"]
        plant = self.notebook_locals["plant"]
        initials = self.notebook_locals["initials"]
        plant_context = plant.GetMyContextFromRoot(context)

        model_instance1 = plant.GetModelInstanceByName(f"{initials[0]}_letter")
        X_WO1 = plant.EvalBodyPoseInWorld(
            plant_context,
            plant.GetBodyByName(f"{initials[0]}_body_link", model_instance1),
        )
        R_WO1 = X_WO1.rotation()

        model_instance2 = plant.GetModelInstanceByName(f"{initials[1]}_letter")
        X_WO2 = plant.EvalBodyPoseInWorld(
            plant_context,
            plant.GetBodyByName(f"{initials[1]}_body_link", model_instance2),
        )
        R_WO2 = X_WO2.rotation()
        self.assertTrue(R_WO1.IsNearlyIdentity(tolerance=0.5))
        self.assertTrue(R_WO2.IsNearlyIdentity(tolerance=0.05))
