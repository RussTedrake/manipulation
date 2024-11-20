"""
This file contains a number of helper systems useful for setting up diagrams.
"""

import numpy as np
from pydrake.all import (
    AbstractValue,
    Body,
    DiagramBuilder,
    DifferentialInverseKinematicsIntegrator,
    DifferentialInverseKinematicsParameters,
    Frame,
    LeafSystem,
    MultibodyPlant,
    RigidTransform,
)


class ExtractPose(LeafSystem):
    """A system that extracts a single pose from a List[RigidTransform], which is the
    output of the MultibodyPlant body_poses output port.

    Args:
        index: The index of the element in the vector whose pose we want to extract
            (e.g. `int(plant.GetBodyByName("body").index())`).
        X_BA: An optional fixed transform from the frame B in the list to an output
            frame A.
    """

    def __init__(
        self,
        index: int,
        X_BA: RigidTransform = RigidTransform(),
    ):
        LeafSystem.__init__(self)
        self.index = index
        self.DeclareAbstractInputPort("poses", AbstractValue.Make([RigidTransform()]))
        self.DeclareAbstractOutputPort(
            "pose",
            lambda: AbstractValue.Make(RigidTransform()),
            self._CalcOutput,
        )
        self.X_BA = X_BA

    def _CalcOutput(self, context, output):
        poses = self.EvalAbstractInput(context, 0).get_value()
        pose = poses[self.index] @ self.X_BA
        output.get_mutable_value().set(pose.rotation(), pose.translation())


class MultibodyPositionToBodyPose(LeafSystem):
    """A system that computes a body pose from a MultibodyPlant position vector. The
    output port calls `plant.SetPositions()` and then `plant.EvalBodyPoseInWorld()`.

    Args:
        plant: The MultibodyPlant.
        body: A body in the plant whose pose we want to compute (e.g. `plant.
            GetBodyByName("body")`).
    """

    def __init__(self, plant: MultibodyPlant, body: Body):
        LeafSystem.__init__(self)
        self.plant = plant
        self.body = body
        self.plant_context = plant.CreateDefaultContext()
        self.DeclareVectorInputPort("position", plant.num_positions())
        self.DeclareAbstractOutputPort(
            "pose",
            lambda: AbstractValue.Make(RigidTransform()),
            self._CalcOutput,
        )

    def _CalcOutput(self, context, output):
        position = self.get_input_port().Eval(context)
        self.plant.SetPositions(self.plant_context, position)
        pose = self.plant.EvalBodyPoseInWorld(self.plant_context, self.body)
        output.get_mutable_value().set(pose.rotation(), pose.translation())


def AddIiwaDifferentialIK(
    builder: DiagramBuilder, plant: MultibodyPlant, frame: Frame | None = None
) -> DifferentialInverseKinematicsIntegrator:
    """Adds a DifferentialInverseKinematicsIntegrator system to the builder with default parameters suitable for use with the standard 7-link iiwa models or the 3-link planar iiwa models.

    Args:
        builder: The DiagramBuilder to which the system should be added.

        plant: The MultibodyPlant passed to the DifferentialInverseKinematicsIntegrator.

        frame: The frame to use for the end effector command. Defaults to the body
            frame of "iiwa_link_7".

    Returns:
        The DifferentialInverseKinematicsIntegrator system.
    """
    params = DifferentialInverseKinematicsParameters(
        plant.num_positions(), plant.num_velocities()
    )
    time_step = plant.time_step()
    q0 = plant.GetPositions(plant.CreateDefaultContext())
    params.set_nominal_joint_position(q0)
    params.set_end_effector_angular_speed_limit(2)
    params.set_end_effector_translational_velocity_limits([-2, -2, -2], [2, 2, 2])
    if frame is None:
        frame = plant.GetFrameByName("iiwa_link_7")
    if plant.num_positions() == 3:  # planar iiwa
        iiwa14_velocity_limits = np.array([1.4, 1.3, 2.3])
        params.set_joint_velocity_limits(
            (-iiwa14_velocity_limits, iiwa14_velocity_limits)
        )
        # These constants are in body frame
        assert (
            frame.name() == "iiwa_link_7"
        ), "Still need to generalize the remaining planar diff IK params for different frames"  # noqa
        params.set_end_effector_velocity_flag([True, False, False, True, False, True])
    else:
        iiwa14_velocity_limits = np.array([1.4, 1.4, 1.7, 1.3, 2.2, 2.3, 2.3])
        params.set_joint_velocity_limits(
            (-iiwa14_velocity_limits, iiwa14_velocity_limits)
        )
        params.set_joint_centering_gain(10 * np.eye(7))
    differential_ik = builder.AddSystem(
        DifferentialInverseKinematicsIntegrator(
            plant,
            frame,
            time_step,
            params,
            log_only_when_result_state_changes=True,
        )
    )
    return differential_ik
