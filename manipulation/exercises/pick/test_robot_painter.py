import unittest
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np
from pydrake.all import PiecewiseQuaternionSlerp, PiecewisePolynomial


class TestRobotPainter(unittest.TestCase):

    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(4)
    @timeout_decorator.timeout(1.)
    def test_key_frames(self):
        """compose_circualr_key_frames"""
        f = self.notebook_locals['compose_circular_key_frames']
        X_WC = self.notebook_locals['X_WorldCenter']
        X_WG = self.notebook_locals['X_WG']
        radius = self.notebook_locals['radius']
        thetas = self.notebook_locals['thetas']
        # carry out computation
        output_frames = f(thetas, X_WC, X_WG)

        # test all key positions match radius
        for i, frame_i in enumerate(output_frames):
            if i == 0:
                # check if the first frame is the gripper's current frame
                dist = np.linalg.norm(frame_i.translation() -
                                      X_WG.translation())
                self.assertLessEqual(dist, 1e-6,
                                     'first key frame position incorrenct!')
            else:
                # check if the radius is correct
                pos_cur = frame_i.translation()
                r_cur = np.linalg.norm(pos_cur - X_WC.translation())
                self.assertLessEqual(abs(radius - r_cur), 1e-6,
                                     'key frame positions incorrenct!')

                # check if +z of each frame points toward the center
                z_cur = frame_i.rotation().matrix()[0:3, 2]
                test_center = z_cur * radius + pos_cur
                center_err = np.linalg.norm(test_center - X_WC.translation())
                self.assertLessEqual(center_err, 1e-6,
                                     'key frame orientations incorrect!')

    @weight(2)
    @timeout_decorator.timeout(1.)
    def test_trajectories(self):
        """construct_v_w_trajectories"""
        f = self.notebook_locals['construct_v_w_trajectories']
        key_frames = self.notebook_locals['test_key_frames']
        times = self.notebook_locals['times']

        # Make the trajectories
        traj_v_G_test, traj_w_G_test = f(times, key_frames)

        self.assertTrue(isinstance(traj_v_G_test, PiecewisePolynomial))
        self.assertTrue(isinstance(traj_w_G_test, PiecewisePolynomial))

        # use student's answer on first problem to construct trajectories
        key_frame_pos = []
        for kf in key_frames:
            key_frame_pos.append(kf.translation())
        key_frame_pos = np.asarray(key_frame_pos)
        key_frame_ori = [pose.rotation().matrix() for pose in key_frames]

        traj_position = PiecewisePolynomial.FirstOrderHold(
            times, key_frame_pos.T)
        traj_rotation = PiecewiseQuaternionSlerp(times, key_frame_ori)
        traj_vG_true = traj_position.MakeDerivative()
        traj_wG_true = traj_rotation.MakeDerivative()

        self.assertTrue(traj_wG_true.isApprox(traj_w_G_test, tol=1e-3))
        self.assertTrue(traj_vG_true.isApprox(traj_v_G_test, tol=1e-3))
