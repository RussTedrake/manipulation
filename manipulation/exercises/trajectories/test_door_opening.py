import unittest

import numpy as np
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight


class TestDoorOpening(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

        simulator = self.notebook_locals["simulator"]
        station_plant = self.notebook_locals["station_plant"]

        context = simulator.get_context()
        door_angle = station_plant.GetPositions(
            station_plant.GetMyContextFromRoot(context),
            station_plant.GetModelInstanceByName("cupboard"),
        )

        self.door_angle_deg = np.rad2deg(door_angle[1])

    @weight(2)
    @timeout_decorator.timeout(15.0)
    def test_ten_bound(self):
        """Test 10 degree bound for door angle"""
        self.assertGreaterEqual(
            self.door_angle_deg, 10, "Door is not opened more than 10 degs."
        )

    @weight(2)
    @timeout_decorator.timeout(15.0)
    def test_forty_bound(self):
        """Test 40 degree bound for door angle"""
        self.assertGreaterEqual(
            self.door_angle_deg, 40, "Door is not opened more than 40 degs."
        )

    @weight(1)
    @timeout_decorator.timeout(15.0)
    def test_seventy_bound(self):
        """Test 70 degree bound for door angle"""
        self.assertGreaterEqual(
            self.door_angle_deg, 70, "Door is not opened more than 70 degs."
        )
