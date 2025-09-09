import unittest

import numpy as np
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight
from pydrake.all import BasicVector


class TestIntroFundamentalsPendulumImplementation(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(10)
    @timeout_decorator.timeout(5.0)
    def test_inverted_pendulum_structure(self):
        InvertedPendulum = self.notebook_locals["InvertedPendulum"]

        # Test class creation
        pendulum = InvertedPendulum()

        # Test system structure
        self.assertEqual(
            pendulum.num_continuous_states(), 2, "Should have 2 continuous states"
        )
        self.assertEqual(pendulum.num_input_ports(), 1, "Should have 1 input port")
        self.assertEqual(pendulum.num_output_ports(), 1, "Should have 1 output port")

    @weight(10)
    @timeout_decorator.timeout(5.0)
    def test_pendulum_dynamics(self):
        InvertedPendulum = self.notebook_locals["InvertedPendulum"]

        # Test methods with known values (extracted from notebook)
        pendulum = InvertedPendulum()
        pendulum_context = pendulum.CreateDefaultContext()
        theta, theta_dot, torque = np.pi / 6, 0.5, 1.0

        pendulum.get_input_port(0).FixValue(pendulum_context, [torque])
        pendulum_context.get_mutable_continuous_state().SetFromVector(
            [theta, theta_dot]
        )

        # Test DoCalcTimeDerivatives
        derivatives = pendulum.AllocateTimeDerivatives()
        pendulum.DoCalcTimeDerivatives(pendulum_context, derivatives)
        computed_derivatives = derivatives.get_vector().CopyToVector()
        expected_derivatives = [theta_dot, float((-9.81 * np.sin(theta) + torque))]

        # Verify results
        self.assertTrue(
            np.allclose(computed_derivatives, expected_derivatives, rtol=1e-10),
            f"Derivatives mismatch: got {computed_derivatives}, expected {expected_derivatives}",
        )

    @weight(10)
    @timeout_decorator.timeout(5.0)
    def test_pendulum_output(self):
        """Test pendulum output computation"""
        InvertedPendulum = self.notebook_locals["InvertedPendulum"]

        pendulum = InvertedPendulum()
        pendulum_context = pendulum.CreateDefaultContext()
        theta, theta_dot = np.pi / 6, 0.5
        pendulum_context.get_mutable_continuous_state().SetFromVector(
            [theta, theta_dot]
        )

        # Test OutputTheta
        output = BasicVector(1)
        pendulum.OutputTheta(pendulum_context, output)
        computed_output = output[0]

        self.assertAlmostEqual(
            computed_output,
            theta,
            places=10,
            msg=f"Output mismatch: got {computed_output}, expected {theta}",
        )


class TestIntroFundamentalsSimulationExercises(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(20)
    @timeout_decorator.timeout(10.0)
    def test_verification_1(self):
        """Test Verification 1: Basic simulation"""
        simulate_pendulum = self.notebook_locals["simulate_pendulum"]

        times, outputs = simulate_pendulum([0.15, 0.0], simulation_time=2.5, torque=0.0)
        final_angle = outputs[0, -1]

        # Expected value from running the solution
        expected_final_angle = 0.005206

        self.assertAlmostEqual(
            final_angle,
            expected_final_angle,
            places=4,
            msg=f"Verification 1 failed: got {final_angle:.4f}, expected {expected_final_angle:.4f}",
        )

    @weight(20)
    @timeout_decorator.timeout(10.0)
    def test_verification_2(self):
        """Test Verification 2: With applied torque"""
        simulate_pendulum = self.notebook_locals["simulate_pendulum"]

        times, outputs = simulate_pendulum([-0.1, 0.2], simulation_time=1.8, torque=0.5)
        final_angle = outputs[0, -1]

        # Expected value from running the solution
        expected_final_angle = -0.10743

        self.assertAlmostEqual(
            final_angle,
            expected_final_angle,
            places=4,
            msg=f"Verification 2 failed: got {final_angle:.4f}, expected {expected_final_angle:.4f}",
        )
