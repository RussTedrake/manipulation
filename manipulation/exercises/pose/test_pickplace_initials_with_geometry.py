import unittest

import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight


class TestPickPlacePosesWithGeometry(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(1)
    @timeout_decorator.timeout(5.0)
    def test_final_pose(self):
        context = self.notebook_locals["context"]
        plant = self.notebook_locals["plant"]
        initial = "B"
        plant_context = plant.GetMyContextFromRoot(context)

        X_WOinitial = plant.EvalBodyPoseInWorld(
            plant_context, plant.GetBodyByName(f"{initial}_body_link")
        )
        p_WOinitial = X_WOinitial.translation()

        # check initial is above the table
        self.assertTrue(p_WOinitial[2].item() > -0.01)
