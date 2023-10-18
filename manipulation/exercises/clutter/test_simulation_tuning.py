import unittest
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np
from pydrake.all import RotationMatrix


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
        box_poses = plant.GetPositions(plant_context)
        box1_pos = box_poses[:7][4:]
        box2_pos = box_poses[7:][4:]

        box_pos_range = np.array(
            [
                [0.0, 1.05],
                [-0.15, 0.15],
                [0.035, 0.075],
            ]
        )

        in_range_list1 = []
        in_range_list2 = []
        for i in range(3):
            in_range_1 = (
                box_pos_range[i, 0] < box1_pos[i] < box_pos_range[i, 1]
            )
            in_range_2 = (
                box_pos_range[i, 0] < box2_pos[i] < box_pos_range[i, 1]
            )
            in_range_list1.append(in_range_1)
            in_range_list2.append(in_range_2)

        in_range_all = np.asarray(in_range_list1 + in_range_list2)
        self.assertTrue(
            in_range_all.all(), "Final box positions are not in correct range."
        )

    @timeout_decorator.timeout(2.0)
    @weight(2)
    def test_set_masses(self):
        """Test test_set_masses"""
        simulator = self.notebook_locals["simulator_d"]
        diagram = self.notebook_locals["diagram_d"]

        context = simulator.get_context()
        plant = diagram.GetSubsystemByName("plant")
        plant_context = plant.GetMyMutableContextFromRoot(context)
        box_poses = plant.GetPositions(plant_context)
        box1_pos = box_poses[:7][4:]
        box2_pos = box_poses[7:][4:]

        box_pos_range = np.array(
            [
                [0.0, 1.05],
                [-0.15, 0.15],
                [0.035, 0.075],
            ]
        )

        in_range_list1 = []
        in_range_list2 = []
        for i in range(3):
            in_range_1 = (
                box_pos_range[i, 0] < box1_pos[i] < box_pos_range[i, 1]
            )
            in_range_2 = (
                box_pos_range[i, 0] < box2_pos[i] < box_pos_range[i, 1]
            )
            in_range_list1.append(in_range_1)
            in_range_list2.append(in_range_2)

        in_range_all = np.asarray(in_range_list1 + in_range_list2)
        self.assertTrue(
            in_range_all.all(),
            "Final box positions are not in the correct range.",
        )

    @timeout_decorator.timeout(2.0)
    @weight(3)
    def test_set_friction(self):
        """Test test_set_friction"""
        simulator = self.notebook_locals["simulator_f"]
        diagram = self.notebook_locals["diagram_f"]

        context = simulator.get_context()
        plant = diagram.GetSubsystemByName("plant")
        plant_context = plant.GetMyMutableContextFromRoot(context)
        box_poses = plant.GetPositions(plant_context)
        box1_pos = box_poses[:7][4:]
        box2_pos = box_poses[7:][4:]

        box1_pos_range = np.array(
            [[0.25, 0.35], [-0.035, 0.035], [0.035, 0.045]]
        )
        box2_pos_range = np.array(
            [[0.25, 0.35], [-0.035, 0.035], [0.055, 0.065]]
        )

        in_range_list1 = []
        in_range_list2 = []
        for i in range(3):
            in_range_1 = (
                box1_pos_range[i, 0] < box1_pos[i] < box1_pos_range[i, 1]
            )
            in_range_2 = (
                box2_pos_range[i, 0] < box2_pos[i] < box2_pos_range[i, 1]
            )
            in_range_list1.append(in_range_1)
            in_range_list2.append(in_range_2)

        in_range_all = np.asarray(in_range_list1 + in_range_list2)
        self.assertTrue(
            in_range_all.all(),
            "Final box positions are not in the correct range.",
        )

    @timeout_decorator.timeout(2.0)
    @weight(2)
    def test_multi_contact(self):
        """Test test_multi_contact"""
        simulator = self.notebook_locals["simulator_c"]
        diagram = self.notebook_locals["diagram_c"]

        context = simulator.get_context()
        plant = diagram.GetSubsystemByName("plant")
        plant_context = plant.GetMyMutableContextFromRoot(context)

        plant.get_contact_results_output_port()
        contact_results = plant.get_contact_results_output_port().Eval(
            plant_context
        )
        n_contacts = contact_results.num_point_pair_contacts()

        self.assertGreater(
            n_contacts,
            2,
            "AddBox not implemented correctly. Not enough contact points have beendetected.",
        )

    @timeout_decorator.timeout(5.0)
    @weight(2)
    def test_minimal_rotation(self):
        """Test test_minimal_rotation"""
        sim_maker = self.notebook_locals["sim_maker"]
        f = sim_maker.make_simulation
        simulator, diagram = f(0.001, (1, 1), (0.1, 0.1))

        context = simulator.get_context()
        plant = diagram.GetSubsystemByName("plant")
        plant_context = plant.GetMyMutableContextFromRoot(context)

        box2_frame = plant.GetBodyByName("box_2").body_frame()

        final_rot = box2_frame.CalcPoseInWorld(plant_context).rotation()
        expected_rot = RotationMatrix.MakeYRotation(0.1 + np.pi / 2)

        rot_diff = final_rot.inverse() @ expected_rot
        diff_norm = np.linalg.norm(rot_diff.matrix() - np.eye(3))

        self.assertLessEqual(
            diff_norm,
            0.15,
            "AddBox not implemented correctly. Box 2 moved more than expected.",
        )
