import unittest

import numpy as np
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight
from pydrake.all import RigidTransform, RotationMatrix


class TestMobileBaseIk(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(5)
    @timeout_decorator.timeout(60.0)
    def test_mobile_base_ik(self):
        """Test solve_ik"""
        # yapf: disable
        goal_rotation1 = RotationMatrix([  # noqa
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1],
        ])
        # yapf: enable
        goal_position1 = np.array([-0.83, 0.18, 1.4])
        goal_pose1 = RigidTransform(goal_rotation1, goal_position1)

        # yapf: disable
        goal_rotation2 = RotationMatrix([  # noqa
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ])
        # yapf: enable
        goal_position2 = np.array([0.83, 0.18, 1.1])
        goal_pose2 = RigidTransform(goal_rotation2, goal_position2)

        # yapf: disable
        goal_rotation3 = RotationMatrix([  # noqa
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0],
        ])
        # yapf: enable
        goal_position3 = np.array([0, 1.13, 1.6])
        goal_pose3 = RigidTransform(goal_rotation3, goal_position3)

        for goal_pose in [goal_pose1, goal_pose2, goal_pose3]:
            q = self.notebook_locals["solve_ik"](goal_pose)

            self.assertTrue(q is not None, "IK failed, no configuration returned!")

            diagram, plant, scene_graph = self.notebook_locals["build_env"]()

            context = diagram.CreateDefaultContext()
            plant_context = plant.GetMyContextFromRoot(context)
            sg_context = scene_graph.GetMyContextFromRoot(context)

            plant.SetPositions(plant_context, q)

            query_object = plant.get_geometry_query_input_port().Eval(plant_context)
            inspector = query_object.inspector()

            pairs = (
                scene_graph.get_query_output_port()
                .Eval(sg_context)
                .inspector()
                .GetCollisionCandidates()
            )
            min_dist = np.inf
            for pair in pairs:
                dist = query_object.ComputeSignedDistancePairClosestPoints(
                    pair[0], pair[1]
                ).distance
                if dist < min_dist:
                    min_dist = dist
                    min_pair_name = (
                        inspector.GetName(inspector.GetFrameId(pair[0])),
                        inspector.GetName(inspector.GetFrameId(pair[1])),
                    )

            self.assertTrue(
                min_dist >= 0.0,
                "Collision present between %s and %s." % min_pair_name,
            )

            gripper_body = plant.GetBodyByName("l_gripper_palm_link")
            ee_pose = gripper_body.EvalPoseInWorld(plant_context)

            self.assertTrue(
                np.linalg.norm(
                    goal_pose.translation() - ee_pose.translation(), ord=np.inf
                )
                <= 0.002,
                "End effector position doesn't match goal position.",
            )
            self.assertTrue(
                goal_pose.rotation()
                .InvertAndCompose(ee_pose.rotation())
                .ToAngleAxis()
                .angle()
                <= 0.02,
                "End effector orientation doesn't match goal orientation.",
            )

        q = self.notebook_locals["solve_ik"](
            goal_pose3,
            max_tries=5,
            fix_base=True,
            base_pose=np.array([-1.23, 0.05, 0]),
        )
        self.assertTrue(
            q is None, "Function solve_ik does not follow fix_base parameter."
        )
