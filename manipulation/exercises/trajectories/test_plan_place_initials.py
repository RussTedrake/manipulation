import unittest

import numpy as np
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight
from pydrake.all import RigidTransform, RotationMatrix


def _test_path(
    testcase: unittest.TestCase, notebook_locals: dict, planner_key: str
) -> list[tuple[float, float, float, float, float, float, float]]:

    student_planner = notebook_locals["rrt_connect_planning"]
    sim = notebook_locals["sim"]
    np.random.seed(0)
    scenario = notebook_locals["scenario_base_file"]

    if planner_key == "path_pick":
        q_i = notebook_locals["q_initial"]
        q_g = notebook_locals["q_approach"]
        max_iter = 1000
        # scenario = notebook_locals["scenario_base_file"]
    elif planner_key == "path_place":
        q_i = notebook_locals["q_approach"]
        q_g = notebook_locals["q_goal"]
        max_iter = 4500
        # scenario = notebook_locals["scenario_grasp_file"]
    elif planner_key == "path_reset":
        q_i = notebook_locals["q_goal"]
        q_g = notebook_locals["q_initial"]
        max_iter = 1000
        # scenario = notebook_locals["scenario_base_file"]
    else:
        testcase.assertTrue(False, msg="Ran test case with invalid planner_key")

    sim.choose_sim(scenario, q_iiwa=q_i)

    for i in range(3):
        student_path, student_iter = student_planner(sim, q_i, q_g, max_iter)
        if student_path is not None:
            print("run {} found a solution!".format(i))
            break
        print("run {} fails to find a solution".format(i))

    testcase.assertTrue(
        expr=(student_path is not None),
        msg="computed path is None. This may imply that "
        "the RRT planner fails to find a solution "
        "out of 3 attempts for path {}.".format(planner_key),
    )
    testcase.assertTrue(
        isinstance(student_path[-1], tuple),
        msg="please use tuple to store configurations. "
        "The computed path should be a list of tuples",
    )
    testcase.assertTrue(
        student_path[-1] == q_g,
        msg="the last configuration of the path does " "not equal the goal.",
    )
    testcase.assertTrue(
        student_path[0] == q_i,
        msg="the first configuration of the path "
        "does not match the start configuration",
    )

    collisions = False
    for i in range(len(student_path)):
        if sim.ExistsCollision(student_path[i], 0.1):
            collisions = True
            break
    testcase.assertTrue(
        collisions == False,
        msg="Detected collision on {}!".format(planner_key),
    )

    return student_path


class TestRRT_Connect_initials(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(3)
    @timeout_decorator.timeout(60.0)
    def test_path_pick(self):
        _test_path(self, self.notebook_locals, "path_pick")

    @weight(6)
    @timeout_decorator.timeout(60.0)
    def test_path_place(self):
        _test_path(self, self.notebook_locals, "path_place")

    @weight(3)
    @timeout_decorator.timeout(60.0)
    def test_path_reset(self):
        _test_path(self, self.notebook_locals, "path_reset")


class TestShortcut(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(10)
    @timeout_decorator.timeout(90.0)
    def test_shortcut(self):
        print("Note: this test does not explicitly verify a shorter path length")
        np.random.seed(0)

        short_cutter = self.notebook_locals["shortcut_path"]
        student_planner = self.notebook_locals["rrt_connect_planning"]
        sim = self.notebook_locals["sim"]
        scenario = self.notebook_locals["scenario_base_file"]
        q_i = self.notebook_locals["q_approach"]
        q_g = self.notebook_locals["q_goal"]

        sim.choose_sim(scenario, q_iiwa=q_i)
        for i in range(3):
            path, student_iter = student_planner(sim, q_i, q_g, 4500)
            if path is not None:
                print("run {} found a solution in RRT-Connect!".format(i))
                break
            print("run {} fails to find a solution with RRT-Connect".format(i))

        self.assertTrue(
            path is not None,
            msg="RRT-Connect failed to find valid path in 3 tries!",
        )

        short_path = short_cutter(sim, path)

        self.assertTrue(
            expr=(short_path is not None),
            msg="Shortcut path is None.",
        )
        self.assertTrue(
            isinstance(short_path[-1], tuple),
            msg="please use tuple to store configurations. "
            "The computed path should be a list of tuples",
        )
        self.assertTrue(
            short_path[-1] == path[-1],
            msg="the last configuration of the shortcut path does "
            "not equal the last configuration of the original path",
        )
        self.assertTrue(
            short_path[0] == path[0],
            msg="the first configuration of the shortcut path "
            "does not match the start configuration of the original path",
        )

        collisions = False
        for i in range(len(short_path)):
            if sim.ExistsCollision(short_path[i], 0.1):
                collisions = True
                break
        self.assertTrue(
            collisions == False,
            msg="Detected collision in Shortcut Path!",
        )
        print("\n All Shortcut tests passed!!")


class TestIK_initials(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(5)
    @timeout_decorator.timeout(60.0)
    def test_ik_initials(self):
        """Test solve_ik_for_pose"""
        sim = self.notebook_locals["sim"]
        solve_ik_for_pose = self.notebook_locals["solve_ik_for_pose"]

        goal_pose_1 = RigidTransform(
            RotationMatrix(
                [
                    [-1.8190895947787466e-16, 0.2392493292139824, -0.970958165149591],
                    [
                        1.0000000000000004,
                        3.1165434525973206e-16,
                        -1.7364165536903736e-16,
                    ],
                    [1.8684985011115333e-16, -0.9709581651495918, -0.23924932921398262],
                ]
            ),
            [0.4656169556555604, 1.7009951107294353e-16, 0.6793215789060889],
        )

        goal_pose_2 = RigidTransform(
            RotationMatrix(
                [
                    [0.0, 1.2246467991473532e-16, -1.0],
                    [0.0, 1.0, 1.2246467991473532e-16],
                    [1.0, 0.0, 0.0],
                ]
            ),
            [0.49574, -0.410, 0.305],
        )

        goal_pose_3 = RigidTransform(
            R=RotationMatrix(
                [
                    [0, 1, 0.0],
                    [0.0, 0.0, 1.0],
                    [1, 0, 0.0],
                ]
            ),
            p=[0.8, -0.2, 0.3],
        )

        for goal_pose in [goal_pose_1, goal_pose_2, goal_pose_3]:
            q = solve_ik_for_pose(sim.plant, goal_pose)
            self.assertTrue(q is not None, "IK failed, no configuration returned!")

            sim.SetStationConfiguration(q, sim.gripper_setpoint)

            gripper_frame = sim.plant.GetFrameByName("body")
            X_WG_actual = sim.plant.CalcRelativeTransform(
                sim.context_plant, sim.plant.world_frame(), gripper_frame
            )

            # Position check
            pos_err = np.linalg.norm(
                goal_pose.translation() - X_WG_actual.translation(), ord=np.inf
            )
            self.assertTrue(
                pos_err <= 0.016,  # just slightly more than default for tolerance
                "End effector position doesn't match goal position.",
            )

            # Orientation check (angle-axis)
            rot_err = (
                goal_pose.rotation()
                .InvertAndCompose(X_WG_actual.rotation())
                .ToAngleAxis()
                .angle()
            )
            self.assertTrue(
                rot_err
                <= 0.011 * np.pi,  # just slightly more than default for tolerance
                "End effector orientation doesn't match goal orientation.",
            )
