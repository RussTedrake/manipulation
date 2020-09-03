import unittest
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np


class TestManipulationIO(unittest.TestCase):

    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(5)
    @timeout_decorator.timeout(1.)
    def test_dynamics(self):
        """Test state space dynamics"""
        f = self.notebook_locals['get_velocity']
        station = self.notebook_locals['station']
        station_context = self.notebook_locals['station_context']

        velocities = f(station, station_context)
        self.assertLessEqual(
            np.linalg.norm(velocities), 1e-6,
            'The measured velocities are not correct.')