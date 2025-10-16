import unittest

import numpy as np
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight


class TestPIDController(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(1)
    @timeout_decorator.timeout(3.0)
    def test_pd_controller_output(self):
        """Test PD controller produces expected output"""
        simulate_IIWA14_with_pd_controller = self.notebook_locals[
            "simulate_IIWA14_with_pd_controller"
        ]

        # Standard test conditions
        q_initial = np.array([0.5, 0.3, 0, -0.5, 0, 0.8, 0])
        q_desired = np.array([0, 0, 0, 0, 0, 0, 0])

        q_final, _, _ = simulate_IIWA14_with_pd_controller(
            kp=150, kd=50, q_initial=q_initial, q_desired=q_desired, simulation_time=2.0
        )

        q_expected = np.array([0, 0, 0, 0, 0, 0, 0])

        self.assertLessEqual(
            np.linalg.norm(q_expected - q_final),
            0.05,
            "PD controller output not within expected range",
        )

    @weight(1)
    @timeout_decorator.timeout(3.0)
    def test_pid_controller_output(self):
        """Test PID controller produces expected output"""
        simulate_IIWA14_with_individual_pid_controller = self.notebook_locals[
            "simulate_IIWA14_with_individual_pid_controller"
        ]

        # Standard test conditions
        q_initial = np.array([0.8, -0.5, 0.3, -1.2, 0.2, 0.7, -0.3])
        q_desired = np.array([0, 0, 0, 0, 0, 0, 0])

        q_final, _, _ = simulate_IIWA14_with_individual_pid_controller(
            kp=150,
            kd=30,
            ki=5,
            q_initial=q_initial,
            q_desired=q_desired,
            simulation_time=2.0,
        )

        q_expected = np.array([0, 0, 0, 0, 0, 0, 0])

        self.assertLessEqual(
            np.linalg.norm(q_expected - q_final),
            0.05,
            "PID controller output not within expected range",
        )
