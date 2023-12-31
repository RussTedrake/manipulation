"""
This file contains a number of helper systems useful for setting up diagrams.
"""

import numpy as np
from pydrake.all import (
    AbstractValue,
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


def AddIiwaDifferentialIK(
    builder: DiagramBuilder, plant: MultibodyPlant, frame: Frame = None
) -> DifferentialInverseKinematicsIntegrator:
    """Adds a DifferentialInverseKinematicsIntegrator system to the builder with default parameters suitable for use with the standard 7-link iiwa models or the 3-link planar iiwa models.

    Args:
        builder: The DiagramBuilder to which the system should be added.

        plant: The MultibodyPlant passed to the DifferentialInverseKinematicsIntegrator.

        frame: The frame to use for the end effector command. Defaults to the "body"
            frame, which is commonly used for the wsg gripper.

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
    if frame is None:
        frame = plant.GetFrameByName("body")
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
