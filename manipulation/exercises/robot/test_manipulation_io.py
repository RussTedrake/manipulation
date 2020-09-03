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
        """Test get_velocity implementation"""
        f = self.notebook_locals['get_velocity']
        station = self.notebook_locals['station']
        station_context = self.notebook_locals['station_context']

        np.random.seed(7)
        for i in range(10):
            test_vel = np.random.rand(7)  # draw 7 random numbers
            station.SetIiwaVelocity(station_context, test_vel)
            eval_vel = f(station, station_context)
            self.assertLessEqual(np.linalg.norm(test_vel - eval_vel), 1e-6,
                                 'get_velocity implementation is not correct!')
