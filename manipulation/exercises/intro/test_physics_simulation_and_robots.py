import unittest

import numpy as np
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight
from pydrake.all import BasicVector, Diagram, ModelInstanceIndex, MultibodyPlant


class TestPhysicsSimulationDiagramStructure(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(10)
    @timeout_decorator.timeout(10.0)
    def test_create_IIWA14_diagram_structure(self):
        """Test IIWA14 diagram creation and structure"""
        create_IIWA14_diagram = self.notebook_locals["create_IIWA14_diagram"]

        diagram, plant, iiwa = create_IIWA14_diagram()

        # Test return types
        self.assertIsInstance(diagram, Diagram, "Should return a Diagram")
        self.assertIsInstance(plant, MultibodyPlant, "Should return a MultibodyPlant")
        self.assertIsInstance(
            iiwa, ModelInstanceIndex, "Should return a ModelInstanceIndex"
        )

        # Test robot structure
        self.assertEqual(plant.num_positions(), 7, "IIWA14 should have 7 joints")
        self.assertEqual(plant.num_actuators(), 7, "IIWA14 should have 7 actuators")


class TestPhysicsSimulationSimpleController(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(10)
    @timeout_decorator.timeout(5.0)
    def test_simple_controller_computation(self):
        """Test SimpleController torque computation"""
        SimpleController = self.notebook_locals["SimpleController"]

        # Create controller with known parameters
        controller = SimpleController(
            gain=10.0, q_desired=np.array([1, 1, 0, 0, 0, 0, 0])
        )
        context = controller.CreateDefaultContext()

        # Set known input state
        q = np.array([-1, -1, 0, 0, 0, 3, 2])
        v = np.array([-1, -1, 0, 0, 0, 3, 2])
        controller.input_port.FixValue(context, np.concatenate([q, v]))

        # Compute output
        output = BasicVector(7)
        controller.ComputeTorque(context, output)

        # Expected: tau = -gain * (q - q_desired) = -10 * ([-1,-1,0,0,0,3,2] - [1,1,0,0,0,0,0])
        # = -10 * [-2, -2, 0, 0, 0, 3, 2] = [20, 20, 0, 0, 0, -30, -20]
        expected_output = 10 * np.array([2, 2, -0, -0, -0, -3, -2])

        self.assertTrue(
            np.allclose(output.get_value(), expected_output, atol=1e-2),
            f"Controller output mismatch: got {output.get_value()}, expected {expected_output}",
        )


class TestPhysicsSimulationVerification(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(20)
    @timeout_decorator.timeout(15.0)
    def test_gradescope_verification(self):
        """Test the Gradescope verification exercise"""
        simulate_IIWA14_with_controller = self.notebook_locals[
            "simulate_IIWA14_with_controller"
        ]

        # Run simulation with specified parameters
        q_final = simulate_IIWA14_with_controller(
            q0=np.array([0.2, 0.2, 0.2, 0, 0, 0, 0]),
            controller_gain=120.0,
            q_desired=np.array([0, 0, 0, 0, 0, 0, 0]),
            simulation_time=10.0,
            set_target_realtime_rate=False,
        )

        # Expected values from running the solution
        expected_q_final = np.array(
            [
                1.26317171e-02,
                8.13040566e-03,
                -5.72823579e-02,
                -2.43876159e-02,
                -1.37802794e-04,
                4.71369969e-04,
                -2.97584254e-05,
            ]
        )

        self.assertEqual(len(q_final), 7, "Should return 7 joint positions")
        self.assertTrue(
            np.all(np.isfinite(q_final)), "All joint positions should be finite"
        )
        self.assertTrue(
            np.allclose(q_final, expected_q_final, atol=1e-4),
            f"Final positions mismatch: got {q_final}, expected {expected_q_final}",
        )
