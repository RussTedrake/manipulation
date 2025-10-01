import unittest

import numpy as np
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight


def _run_rrt_like_test(
    testcase: unittest.TestCase, notebook_locals: dict, planner_key: str
):
    """Shared test body for RRT-style planners (single or connect)."""
    # Ensure IKSolver is defined (side-effect in some notebooks)
    notebook_locals["IKSolver"]
    IiwaProblem = notebook_locals["IiwaProblem"]
    env = notebook_locals["env"]

    np.random.seed(0)
    q_init = env.q0
    q_goal = notebook_locals["q_goal"]

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

    # Load the student planner by name
    student_planner = notebook_locals[planner_key]

    student_path = None
    print("Started to solve the problem using RRT planner ...")
    for i in range(4):  # allow 4 attempts
        student_path, student_iter = student_planner(
            iiwa_problem_test, max_iterations=1000
        )
        if student_path is not None:
            print("run 1 found a solution!")
            break
        print("run {} fails to find a solution".format(i))

    testcase.assertTrue(
        expr=(student_path is not None),
        msg="computed path is None. This may imply that "
        "the RRT planner fails to find a solution "
        "out of 3 attempts.",
    )
    testcase.assertTrue(
        isinstance(student_path[-1], tuple),
        msg="please use tuple to store configurations. "
        "The computed path should be a list of tuples",
    )
    testcase.assertTrue(
        student_path[-1] == iiwa_problem_test.goal,
        msg="the last configuration of the path does " "not equal the goal.",
    )
    testcase.assertTrue(
        student_path[0] == iiwa_problem_test.start,
        msg="the first configuration of the path "
        "does not match the start configuration",
    )

    collisions = False
    for i in range(len(student_path)):
        if iiwa_problem_test.collide(student_path[i]):
            collisions = True
            break
    testcase.assertTrue(
        collisions is False,
        msg="Detected collision on robot path!",
    )


class TestRRT(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(5)
    @timeout_decorator.timeout(80.0)
    def test_rrt(self):
        _run_rrt_like_test(self, self.notebook_locals, "rrt_planning")


class TestRRT_Connect(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(5)
    @timeout_decorator.timeout(80.0)
    def test_rrt(self):
        _run_rrt_like_test(self, self.notebook_locals, "rrt_connect_planning")
