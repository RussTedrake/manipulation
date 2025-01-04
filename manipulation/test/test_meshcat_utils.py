import unittest

import numpy as np
from pydrake.all import MathematicalProgram, Meshcat, Solve

import manipulation.meshcat_utils as dut


class TestMeshcatUtils(unittest.TestCase):
    def test_plot_mathematical_program(self):
        prog = MathematicalProgram()
        x = prog.NewContinuousVariables(2)
        prog.AddCost(x.dot(x))
        prog.AddBoundingBoxConstraint(-2, 2, x)
        result = Solve(prog)

        meshcat = Meshcat()
        X, Y = np.meshgrid(np.linspace(-3, 3, 35), np.linspace(-3, 3, 31))
        # Test that plotting doesn't raise any exceptions
        dut.plot_mathematical_program(meshcat, "test", prog, X, Y, result)


if __name__ == "__main__":
    unittest.main()
