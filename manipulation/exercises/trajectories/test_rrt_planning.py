import unittest

import numpy as np
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight


class TestRRT(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(5)
    @timeout_decorator.timeout(60.0)
    def test_rrt(self):
        """Test RRT Planner"""
        # construct a new problem
        self.notebook_locals["IKSolver"]
        IiwaProblem = self.notebook_locals["IiwaProblem"]
        env = self.notebook_locals["env"]

        np.random.seed(0)
        q_init = env.q0
        q_goal = self.notebook_locals["q_goal"]

        gripper_setpoint = 0.1
        left_door_angle = -np.pi / 2
        right_door_angle = np.pi / 2

        iiwa_problem_test = IiwaProblem(
            q_start=q_init,
            q_goal=q_goal,
            gripper_setpoint=gripper_setpoint,
            left_door_angle=left_door_angle,
            right_door_angle=right_door_angle,
            is_visualizing=False,
        )
        print("Constructed problem object")
        # load student RRT method
        student_rrt = self.notebook_locals["rrt_planning"]
        student_path = None
        print("Started to solve the problem using RRT planner ...")
        for i in range(3):  # allow 3 attempts
            student_path = student_rrt(iiwa_problem_test, 500, 0.5)
            if student_path is not None:
                print("run 1 found a solution")
                break
            print("run {} fails to find a solution".format(i))
        self.assertTrue(
            expr=(student_path is not None),
            msg="computed path is None. This may imply that "
            "the RRT planner failse to find a solution "
            "out of 3 attempts.",
        )
        self.assertTrue(
            isinstance(student_path[-1], tuple),
            msg="please use tuple to store configurations."
            "The computed path should be a list of tuples",
        )
        self.assertTrue(
            student_path[-1] == iiwa_problem_test.goal,
            msg="the last configuration of the path " "does not equal to goal.",
        )
        self.assertTrue(
            student_path[0] == iiwa_problem_test.start,
            msg="the first configuration of the path "
            "does not match start configuration",
        )
