import unittest
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np
import os
from pydrake.all import RotationMatrix


class TestSimulationTuning(unittest.TestCase):

    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @timeout_decorator.timeout(2.)
    @weight(2)
    def test_on_slope(self):
        """Test test_on_slope"""
        simulator = self.notebook_locals["simulator_d"]
        diagram = self.notebook_locals["diagram_d"]

        context = simulator.get_context()
        plant = diagram.GetSubsystemByName('plant')
        plant_context = plant.GetMyMutableContextFromRoot(context)
        box_poses = plant.GetPositions(plant_context)
        box1_pos = box_poses[:7][4:]
        box2_pos = box_poses[7:][4:]

        box_pos_range = np.array([
            [0.0, 1.0],
            [-0.1, 0.1],
            [0.03, 0.07],
        ])

        in_range_list1 = []
        in_range_list2 = []
        for i in range(3):
            in_range_1 = box_pos_range[i, 0] < box1_pos[i] < box_pos_range[i, 1]
            in_range_2 = box_pos_range[i, 0] < box2_pos[i] < box_pos_range[i, 1]
            in_range_list1.append(in_range_1)
            in_range_list2.append(in_range_2)

        in_range_all = np.asarray(in_range_list1 + in_range_list2)
        self.assertTrue(in_range_all.all(),
                        'Final box positions not in correct range!')

    @timeout_decorator.timeout(2.)
    @weight(2)
    def test_make_simulation(self):
        """Test test_make_simulation"""
        simulator = self.notebook_locals["simulator_f"]
        diagram = self.notebook_locals["diagram_f"]

        context = simulator.get_context()
        plant = diagram.GetSubsystemByName('plant')
        plant_context = plant.GetMyMutableContextFromRoot(context)
        box_poses = plant.GetPositions(plant_context)
        box1_pos = box_poses[:7][4:]
        box2_pos = box_poses[7:][4:]

        # box1_frame = plant.GetBodyByName('box').body_frame()
        # box2_frame = plant.GetBodyByName('box_2').body_frame()

        # box1_pose = box1_frame.CalcPoseInWorld(plant_context).GetAsMatrix4()
        # box2_pose = box2_frame.CalcPoseInWorld(plant_context).GetAsMatrix4()
        # print(f'Box 1 pose: {box1_pose}')
        # print(f'Box 2 pose: {box2_pose}')

        box_pos_range = np.array([
            [0.0, 1.0],
            [-0.1, 0.1],
            [0.03, 0.07],
        ])

        in_range_list1 = []
        in_range_list2 = []
        for i in range(3):
            in_range_1 = box_pos_range[i, 0] < box1_pos[i] < box_pos_range[i, 1]
            in_range_2 = box_pos_range[i, 0] < box2_pos[i] < box_pos_range[i, 1]
            in_range_list1.append(in_range_1)
            in_range_list2.append(in_range_2)

        in_range_all = np.asarray(in_range_list1 + in_range_list2)
        self.assertTrue(in_range_all.all(),
                        'Final box positions not in correct range!')

    @timeout_decorator.timeout(2.)
    @weight(3)
    def test_make_stacking_simulation(self):
        """Test test_make_stacking_simulation"""
        simulator = self.notebook_locals["simulator_h"]
        diagram = self.notebook_locals["diagram_h"]

        context = simulator.get_context()
        plant = diagram.GetSubsystemByName('plant')
        plant_context = plant.GetMyMutableContextFromRoot(context)
        box_poses = plant.GetPositions(plant_context)
        box1_pos = box_poses[:7][4:]
        box2_pos = box_poses[7:][4:]

        # box1_frame = plant.GetBodyByName('box').body_frame()
        # box2_frame = plant.GetBodyByName('box_2').body_frame()

        # box1_pose = box1_frame.CalcPoseInWorld(plant_context).GetAsMatrix4()
        # box2_pose = box2_frame.CalcPoseInWorld(plant_context).GetAsMatrix4()
        # print(f'Box 1 pose: {box1_pose}')
        # print(f'Box 2 pose: {box2_pose}')

        box1_pos_range = np.array([[0.2, 0.3], [-0.025, 0.025], [0.025,
                                                                 0.0375]])
        box2_pos_range = np.array([[0.2, 0.3], [-0.025, 0.025], [0.0475, 0.06]])

        in_range_list1 = []
        in_range_list2 = []
        for i in range(3):
            in_range_1 = box1_pos_range[i, 0] < box1_pos[i] < box1_pos_range[i,
                                                                             1]
            in_range_2 = box2_pos_range[i, 0] < box2_pos[i] < box2_pos_range[i,
                                                                             1]
            in_range_list1.append(in_range_1)
            in_range_list2.append(in_range_2)

        in_range_all = np.asarray(in_range_list1 + in_range_list2)
        self.assertTrue(in_range_all.all(),
                        'Final box positions not in correct range!')

    @timeout_decorator.timeout(2.)
    @weight(2)
    def test_matching_coll_shape(self):
        """Test test_matching_coll_shape"""
        plant = self.notebook_locals["plant_a"]
        diagram = self.notebook_locals["diagram_a"]

        context = diagram.CreateDefaultContext()
        plant_context = plant.GetMyContextFromRoot(context)
        contact_results = plant.get_contact_results_output_port().Eval(
            plant_context)

        box_q_list = np.array([[0., 0.15, 0.], [0., 0.2, 0.], [0., 0.2, 0.],
                               [0., 0.13, 0.]])

        box2_q_list = np.array([[-0.049, 0.198, 0.], [-0.049, 0.155, 0.],
                                [0.046, 0.168, 0.], [0.046, 0.168, 0.]])

        n_contacts_list = []
        for i in range(4):
            plant.SetPositions(plant_context,
                               np.concatenate([box_q_list[i], box2_q_list[i]]))

            contact_results = plant.get_contact_results_output_port().Eval(
                plant_context)
            n_contacts = contact_results.num_point_pair_contacts()

            n_contacts_list.append(n_contacts)

        self.assertTrue((np.array(n_contacts) > 0).all(),
                        'Objects not contacting!')

    @timeout_decorator.timeout(2.)
    @weight(2)
    def test_force_discontinuity(self):
        """Test test_force_discontinuity"""
        plant = self.notebook_locals["plant_a"]
        diagram = self.notebook_locals["diagram_a"]
        f = self.notebook_locals["set_block_2d_poses"]

        box1_pos1, box2_pos1, box1_pos2, box2_pos2 = f()

        context = diagram.CreateDefaultContext()
        plant_context = plant.GetMyContextFromRoot(context)

        plant.SetPositions(plant_context,
                           np.concatenate([box1_pos1 + box2_pos1]))

        contact_port = plant.get_contact_results_output_port()
        contact_results = plant.get_contact_results_output_port().Eval(
            plant_context)
        info = contact_results.point_pair_contact_info(0)
        force_1 = info.contact_force()
        fn1 = force_1 / np.linalg.norm(force_1)
        point_1 = info.contact_point()

        plant.SetPositions(plant_context,
                           np.concatenate([box1_pos2 + box2_pos2]))

        contact_port = plant.get_contact_results_output_port()
        contact_results = plant.get_contact_results_output_port().Eval(
            plant_context)
        info = contact_results.point_pair_contact_info(0)
        force_2 = info.contact_force()
        fn2 = force_2 / np.linalg.norm(force_2)
        point_2 = info.contact_point()

        contact_pt_dist = np.linalg.norm(
            np.array([point_1[0], point_1[2]])
            - np.array([point_2[0], point_2[2]]))
        force_angle = np.abs(np.arccos(np.dot(fn1, fn2)))

        self.assertLessEqual(
            contact_pt_dist, 0.01,
            'Contact points not close enough, discontinuity not detected')

        self.assertGreaterEqual(
            force_angle, np.deg2rad(60),
            'Angle between force vectors too small, discontinuity not detected')

    @timeout_decorator.timeout(2.)
    @weight(2)
    def test_multi_contact(self):
        """Test test_multi_contact"""
        simulator = self.notebook_locals["simulator_e"]
        diagram = self.notebook_locals["diagram_e"]

        context = simulator.get_context()
        plant = diagram.GetSubsystemByName('plant')
        plant_context = plant.GetMyMutableContextFromRoot(context)

        contact_port = plant.get_contact_results_output_port()
        contact_results = plant.get_contact_results_output_port().Eval(
            plant_context)
        n_contacts = contact_results.num_point_pair_contacts()

        self.assertGreater(n_contacts, 2, 'Not enough contacts detected!')

    @timeout_decorator.timeout(5.)
    @weight(2)
    def test_minimal_rotation(self):
        """Test test_minimal_rotation"""
        sim_maker = self.notebook_locals["sim_maker"]
        f = sim_maker.make_simulation
        simulator, diagram = f(0.001, (1, 1), (0.1, 0.1))

        context = simulator.get_context()
        plant = diagram.GetSubsystemByName('plant')
        plant_context = plant.GetMyMutableContextFromRoot(context)

        box2_frame = plant.GetBodyByName('box_2').body_frame()

        final_rot = box2_frame.CalcPoseInWorld(plant_context).rotation()
        expected_rot = RotationMatrix.MakeYRotation(0.1 + np.pi / 2)

        rot_diff = final_rot.inverse() @ expected_rot
        diff_norm = np.linalg.norm(rot_diff.matrix() - np.eye(3))

        self.assertLessEqual(diff_norm, 0.1, 'Box 2 moved more than expected!')
