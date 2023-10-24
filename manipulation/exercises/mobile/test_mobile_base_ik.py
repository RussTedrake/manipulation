import unittest
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight
import numpy as np
from pydrake.all import RotationMatrix, RigidTransform


class TestMobileBaseIk(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(5)
    @timeout_decorator.timeout(10.0)
    def test_mobile_base_ik(self):
        """Test solve_ik"""
        # yapf: disable
        goal_rotation = RotationMatrix([  # noqa
            [0.9969323810421103, -0.0015776037860101618, -0.07825176544975043],
            [-0.010615080452914535, -0.9932842569807606, -0.11521156583067553],
            [-0.07754448849974165, 0.11568878943299686, -0.9902539857545847],
        ])
        # yapf: enable
        goal_position = np.array(
            [-1.0305885019472236, 0.17511127864621603, 1.396881893512182]
        )
        goal_pose = RigidTransform(goal_rotation, goal_position)
        q = self.notebook_locals["solve_ik"](goal_pose)

        self.assertTrue(q is not None, "IK failed, no configuration returned!")

        diagram, plant, scene_graph = self.notebook_locals["build_env"]()

        context = diagram.CreateDefaultContext()
        plant_context = plant.GetMyContextFromRoot(context)
        sg_context = scene_graph.GetMyContextFromRoot(context)
        self.notebook_locals["filterCollsionGeometry"](scene_graph, sg_context)

        plant.SetPositions(plant_context, q)

        query_object = plant.get_geometry_query_input_port().Eval(
            plant_context
        )
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
