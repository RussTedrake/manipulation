import gymnasium as gym
import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    ConnectPlanarSceneGraphVisualizer,
    ConstantVectorSource,
    Demultiplexer,
    DiagramBuilder,
    FixedOffsetFrame,
    InverseDynamicsController,
    LeafSystem,
    MultibodyPlant,
    Multiplexer,
    Parser,
    PlanarJoint,
    PrismaticJoint,
    RevoluteJoint,
    RigidTransform,
    RotationMatrix,
    Simulator,
    SpatialInertia,
    UnitInertia,
)
from pydrake.gym import DrakeGymEnv

from manipulation.utils import ConfigureParser

""" Defines the PlanarPusherPushingABoxEnv

PlanarPusherPushingABoxEnv is an extremely simple environment where a planar gripper needs to push a foam block to maximize its x coordinate.
"""


def AddPlanarBinAndManipuland(plant):
    parser = Parser(plant)
    ConfigureParser(parser)
    bin = parser.AddModelsFromUrl("package://manipulation/planar_bin.sdf")[0]
    plant.WeldFrames(
        plant.world_frame(),
        plant.GetFrameByName("bin_base", bin),
        RigidTransform(RotationMatrix.MakeZRotation(np.pi / 2.0), [0, 0, -0.015]),
    )
    planar_joint_frame = plant.AddFrame(
        FixedOffsetFrame(
            "planar_joint_frame",
            plant.world_frame(),
            RigidTransform(RotationMatrix.MakeXRotation(np.pi / 2)),
        )
    )
    box = parser.AddModelsFromUrl(
        "package://drake_models/manipulation_station/061_foam_brick.sdf"
    )[0]
    box_frame = plant.AddFrame(
        FixedOffsetFrame(
            "box_frame",
            plant.GetFrameByName("base_link", box),
            RigidTransform(RotationMatrix.MakeXRotation(np.pi / 2)),
        )
    )
    box_joint = plant.AddJoint(PlanarJoint("box_joint", planar_joint_frame, box_frame))
    box_joint.set_default_translation([0, 0.033400])


def AddPlanarGripper(plant):
    parser = Parser(plant)
    ConfigureParser(parser)
    gripper = parser.AddModelsFromUrl(
        "package://manipulation/schunk_wsg_50_welded_fingers.sdf"
    )[0]
    plant.GetBodyByName("body", gripper)

    # Add a planar joint the old fashioned way (so that I can have three actuators):
    plant.AddRigidBody(
        "false_body1",
        gripper,
        SpatialInertia(0, [0, 0, 0], UnitInertia(0, 0, 0)),
    )
    plant.AddRigidBody(
        "false_body2",
        gripper,
        SpatialInertia(0, [0, 0, 0], UnitInertia(0, 0, 0)),
    )
    gripper_x = plant.AddJoint(
        PrismaticJoint(
            "gripper_x",
            plant.world_frame(),
            plant.GetFrameByName("false_body1"),
            [1, 0, 0],
            -0.2,
            0.2,
        )
    )
    plant.AddJointActuator("gripper_x", gripper_x)
    gripper_z = plant.AddJoint(
        PrismaticJoint(
            "gripper_z",
            plant.GetFrameByName("false_body1"),
            plant.GetFrameByName("false_body2"),
            [0, 0, 1],
            0.0,
            0.3,
        )
    )
    gripper_z.set_default_translation(0.25)
    plant.AddJointActuator("gripper_z", gripper_z)
    gripper_frame = plant.AddFrame(
        FixedOffsetFrame(
            "gripper_frame",
            plant.GetFrameByName("body", gripper),
            RigidTransform(RotationMatrix.MakeXRotation(np.pi / 2)),
        )
    )
    gripper_theta = plant.AddJoint(
        RevoluteJoint(
            "gripper_theta",
            plant.GetFrameByName("false_body2"),
            gripper_frame,
            [0, -1, 0],
            -np.pi / 2,
            np.pi / 2,
        )
    )
    plant.AddJointActuator("gripper_theta", gripper_theta)

    return gripper


def MakePlanarGripperOnlyPlant():
    plant = MultibodyPlant(time_step=0.005)
    AddPlanarGripper(plant)
    plant.Finalize()
    return plant


class TruncateAction(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)
        self.DeclareVectorInputPort("action", 3)
        self.DeclareVectorInputPort("state", 12)
        self.DeclareVectorOutputPort("action", 3, self.CalcOutput)

    def CalcOutput(self, context, output):
        action = self.get_input_port(0).Eval(context)
        state = self.get_input_port(1).Eval(context)

        x, y = state[3:5]
        if x < -0.2 or x > 0.2 or y > 0.35 or y < 0.1:
            # push gripper back towards the center of the env
            action[0:2] = np.array([0, 0.25]) - state[0:2]

        theta = state[5]
        theta + action[2]
        pos_mag = np.linalg.norm(action[0:2])
        dp_upper_bound = 0.02
        dtheta_upper_bound = 0.5
        if pos_mag > dp_upper_bound:
            action[0:2] = action[0:2] / pos_mag * dp_upper_bound
        if abs(action[2]) > dtheta_upper_bound:
            action[2] = dtheta_upper_bound * np.sign(action[2])
        output.SetFromVector(action)


# TODO(russt): Add optional "time_step" to Drake's Integrator, and use that
# here instead.
class DiscreteIntegrator(LeafSystem):
    def __init__(self, size, time_step):
        LeafSystem.__init__(self)
        self.DeclareVectorInputPort("u0", size)
        state_index = self.DeclareDiscreteState(size)
        self.DeclareStateOutputPort("y0", state_index)
        self.DeclarePeriodicDiscreteUpdateEvent(
            period_sec=time_step, offset_sec=0.0, update=self.Update
        )

    def Update(self, context, discrete_state):
        u = self.get_input_port().Eval(context)
        x = context.get_discrete_state_vector().GetAtIndex(0)
        x_next = x + u
        discrete_state.get_mutable_vector().SetFromVector(x_next)

    def set_integral_value(self, context, value):
        context.get_mutable_discrete_state_vector().SetAtIndex(0, value)


def PlanarGripperPushingABoxEnv(
    meshcat=None, render=True, render_mode="drake", reward_function=None
):
    # TODO(russt): Convert to HardwareStation.
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.01)
    AddPlanarBinAndManipuland(plant)
    gripper = AddPlanarGripper(plant)

    plant.Finalize()

    controller_plant = MakePlanarGripperOnlyPlant()
    N = controller_plant.num_positions()
    kp = [100] * N
    ki = [1] * N
    kd = [2 * np.sqrt(kp[0])] * N
    controller = builder.AddSystem(
        InverseDynamicsController(controller_plant, kp, ki, kd, False)
    )
    builder.Connect(
        plant.get_state_output_port(gripper),
        controller.get_input_port_estimated_state(),
    )
    builder.Connect(
        controller.get_output_port_control(),
        plant.get_actuation_input_port(),
    )

    time_step = 0.2
    step_interface = DiscreteIntegrator(3, time_step)
    builder.AddSystem(step_interface)
    mux = builder.AddSystem(Multiplexer([3, 3]))
    builder.Connect(step_interface.get_output_port(0), mux.get_input_port(0))
    desired_vel = builder.AddSystem(ConstantVectorSource([0, 0, 0]))
    builder.Connect(desired_vel.get_output_port(0), mux.get_input_port(1))
    builder.Connect(
        mux.get_output_port(0),
        controller.get_input_port_desired_state(),
    )
    truncate = builder.AddSystem(TruncateAction())
    builder.Connect(truncate.get_output_port(), step_interface.get_input_port(0))
    builder.Connect(plant.get_state_output_port(), truncate.get_input_port(1))
    builder.ExportInput(truncate.get_input_port(0), "action")

    if render:
        vis = ConnectPlanarSceneGraphVisualizer(
            builder,
            scene_graph,
            xlim=[-0.3, 0.3],
            ylim=[-0.1, 0.5],
            show=False,
        )
        vis.set_name("visualizer")

    state_demux = builder.AddSystem(Demultiplexer([6, 6]))
    builder.Connect(plant.get_state_output_port(), state_demux.get_input_port())
    builder.ExportOutput(state_demux.get_output_port(0), "position")

    diagram = builder.Build()
    simulator = Simulator(diagram)

    # TODO(russt): Make finite action and observation spaces if we ever want to use this with stable baselines.
    action_space = gym.spaces.Box(
        low=np.array([-np.inf] * 3),
        high=np.array([np.inf] * 3),
        dtype=np.float32,
    )
    observation_space = gym.spaces.Box(
        low=np.array([-np.inf] * 6),
        high=np.array([np.inf] * 6),
        dtype=np.float32,
    )

    env = DrakeGymEnv(
        simulator=simulator,
        time_step=time_step,
        action_space=action_space,
        observation_space=observation_space,
        reward=reward_function,
        action_port_id="action",
        observation_port_id="position",
    )
    return env
