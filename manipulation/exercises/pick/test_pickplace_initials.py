import unittest

import numpy as np
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight


class TestPickPlacePoses(unittest.TestCase):
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

        model_instance1 = plant.GetModelInstanceByName("first_initial")
        X_WO1 = plant.EvalBodyPoseInWorld(
            plant_context,
            plant.GetBodyByName(f"{initials[0]}_body_link", model_instance1),
        )
        p_WO1 = X_WO1.translation()

        model_instance2 = plant.GetModelInstanceByName("last_initial")
        X_WO2 = plant.EvalBodyPoseInWorld(
            plant_context,
            plant.GetBodyByName(f"{initials[1]}_body_link", model_instance2),
        )
        p_WO2 = X_WO2.translation()

        # check initials sit on a line in the yz plane
        initials_aligned = np.linalg.norm(p_WO1[1:] - p_WO2[1:]) < 6e-2
        self.assertTrue(
            initials_aligned,
            "Verification failed, intials are not aligned in the yz plane.",
        )

        # check initials are oriented so the first intial is read before the second
        initials_readable = p_WO1[0].item() < p_WO2[0].item()
        self.assertTrue(
            initials_readable,
            "Verification failed, first initial not to the left of the second initial.",
        )
