import unittest

import numpy as np
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight


class TestSimulationTuning(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @timeout_decorator.timeout(2.0)
    @weight(2)
    def test_set_timestep(self):
        """Test test_set_timestep"""
        simulator = self.notebook_locals["simulator_b"]
        diagram = self.notebook_locals["diagram_b"]

        context = simulator.get_context()
        plant = diagram.GetSubsystemByName("plant")
        plant_context = plant.GetMyMutableContextFromRoot(context)
        block_poses = plant.GetPositions(plant_context)
        block1_pos = block_poses[:7][4:]
        block2_pos = block_poses[7:][4:]

        block_pos_range = np.array([[0.0, 1.0], [-0.1, 0.1], [0.03, 0.07]])

        in_range_list1 = []
        in_range_list2 = []
        for i in range(3):
            in_range_1 = block_pos_range[i, 0] < block1_pos[i] < block_pos_range[i, 1]
            in_range_2 = block_pos_range[i, 0] < block2_pos[i] < block_pos_range[i, 1]
            in_range_list1.append(in_range_1)
            in_range_list2.append(in_range_2)

        in_range_all = np.asarray(in_range_list1 + in_range_list2)
        self.assertTrue(
            in_range_all.all(),
            "Final block positions are not in the correct range.",
        )

    @timeout_decorator.timeout(2.0)
    @weight(2)
    def test_set_masses(self):
        """Test test_set_masses"""
        simulator = self.notebook_locals["simulator_c"]
        diagram = self.notebook_locals["diagram_c"]

        context = simulator.get_context()
        plant = diagram.GetSubsystemByName("plant")
        plant_context = plant.GetMyMutableContextFromRoot(context)
        block_poses = plant.GetPositions(plant_context)
        block1_pos = block_poses[:7][4:]
        block2_pos = block_poses[7:][4:]

        block1_pos_range = np.array([[0.07, 0.09], [-0.0004, 0.000], [0.0475, 0.0675]])
        block2_pos_range = np.array([[0.25, 0.30], [-0.0009, -0.0005], [0.0275, 0.05]])

        in_range_list1 = []
        in_range_list2 = []
        for i in range(3):
            in_range_1 = block1_pos_range[i, 0] < block1_pos[i] < block1_pos_range[i, 1]
            in_range_2 = block2_pos_range[i, 0] < block2_pos[i] < block2_pos_range[i, 1]
            in_range_list1.append(in_range_1)
            in_range_list2.append(in_range_2)

        in_range_all = np.asarray(in_range_list1 + in_range_list2)
        self.assertTrue(
            in_range_all.all(),
            "Final block positions are not in the correct range.",
        )

    @timeout_decorator.timeout(2.0)
    @weight(3)
    def test_set_friction(self):
        """Test test_set_friction"""
        simulator = self.notebook_locals["simulator_e"]
        diagram = self.notebook_locals["diagram_e"]

        context = simulator.get_context()
        plant = diagram.GetSubsystemByName("plant")
        plant_context = plant.GetMyMutableContextFromRoot(context)
        block_poses = plant.GetPositions(plant_context)
        block1_pos = block_poses[:7][4:]
        block2_pos = block_poses[7:][4:]

        block1_pos_range = np.array([[0.2, 0.3], [-0.025, 0.025], [0.025, 0.0375]])
        block2_pos_range = np.array([[0.2, 0.3], [-0.025, 0.025], [0.0475, 0.06]])

        in_range_list1 = []
        in_range_list2 = []
        for i in range(3):
            in_range_1 = block1_pos_range[i, 0] < block1_pos[i] < block1_pos_range[i, 1]
            in_range_2 = block2_pos_range[i, 0] < block2_pos[i] < block2_pos_range[i, 1]
            in_range_list1.append(in_range_1)
            in_range_list2.append(in_range_2)

        in_range_all = np.asarray(in_range_list1 + in_range_list2)
        self.assertTrue(
            in_range_all.all(),
            "Final block positions are not in the correct range.",
        )
