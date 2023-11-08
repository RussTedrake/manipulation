import dataclasses as dc
from functools import partial
import os
import sys
import typing

import numpy as np

from pydrake.all import (
    AbstractValue,
    Adder,
    AddMultibodyPlant,
    ApplyLcmBusConfig,
    ApplyMultibodyPlantConfig,
    ApplyVisualizationConfig,
    BaseField,
    CameraConfig,
    CameraInfo,
    Demultiplexer,
    DepthImageToPointCloud,
    Diagram,
    DiagramBuilder,
    DrakeLcmParams,
    GetScopedFrameByName,
    IiwaCommandSender,
    IiwaDriver,
    IiwaStatusReceiver,
    InverseDynamicsController,
    LeafSystem,
    LcmPublisherSystem,
    LcmSubscriberSystem,
    MakeMultibodyStateToWsgStateSystem,
    MakeRenderEngineVtk,
    RenderEngineVtkParams,
    Meshcat,
    MeshcatPointCloudVisualizer,
    ModelDirective,
    ModelDirectives,
    MultibodyPlant,
    MultibodyPlantConfig,
    OutputPort,
    Parser,
    PassThrough,
    ProcessModelDirectives,
    RgbdSensor,
    RigidTransform,
    SceneGraph,
    SchunkWsgDriver,
    SchunkWsgPositionController,
    SchunkWsgCommandSender,
    SchunkWsgStatusReceiver,
    ScopedName,
    SimulatorConfig,
    StateInterpolatorWithDiscreteDerivative,
    VisualizationConfig,
    ZeroForceDriver,
)
from pydrake.common.yaml import yaml_load_typed
from drake import (
    lcmt_iiwa_command,
    lcmt_iiwa_status,
    lcmt_schunk_wsg_command,
    lcmt_schunk_wsg_status,
)

from manipulation.scenarios import (
    AddIiwa,
    AddPlanarIiwa,
    AddWsg,
)
from manipulation.utils import ConfigureParser


@dc.dataclass
class InverseDynamicsDriver:
    """A simulation-only driver that adds the InverseDynamicsController to the station and exports the output ports. Multiple model instances can be provided using `instance_name1+instance_name2` as the key; the output ports will be named similarly."""


@dc.dataclass
class Scenario:
    """Defines the YAML format for a (possibly stochastic) scenario to be
    simulated.
    """

    # Random seed for any random elements in the scenario. The seed is always
    # deterministic in the `Scenario`; a caller who wants randomness must
    # populate this value from their own randomness.
    random_seed: int = 0

    # The maximum simulation time (in seconds).  The simulator will attempt to
    # run until this time and then terminate.
    simulation_duration: float = np.inf

    # Simulator configuration (integrator and publisher parameters).
    simulator_config: SimulatorConfig = SimulatorConfig(
        max_step_size=0.01,
        use_error_control=False,
        accuracy=1.0e-2,
    )

    # Plant configuration (time step and contact parameters).
    plant_config: MultibodyPlantConfig = MultibodyPlantConfig(
        discrete_contact_solver="sap"
    )

    # All of the fully deterministic elements of the simulation.
    directives: typing.List[ModelDirective] = dc.field(default_factory=list)

    # A map of {bus_name: lcm_params} for LCM transceivers to be used by
    # drivers, sensors, etc.
    lcm_buses: typing.Mapping[str, DrakeLcmParams] = dc.field(
        default_factory=lambda: dict(default=DrakeLcmParams())
    )

    # For actuated models, specifies where each model's actuation inputs come
    # from, keyed on the ModelInstance name.
    model_drivers: typing.Mapping[
        str,
        typing.Union[
            IiwaDriver,
            InverseDynamicsDriver,
            SchunkWsgDriver,
            ZeroForceDriver,
        ],
    ] = dc.field(default_factory=dict)

    # Cameras to add to the scene (and broadcast over LCM). The key for each
    # camera is a helpful mnemonic, but does not serve a technical role. The
    # CameraConfig::name field is still the name that will appear in the
    # Diagram artifacts.
    cameras: typing.Mapping[str, CameraConfig] = dc.field(default_factory=dict)

    visualization: VisualizationConfig = VisualizationConfig()


@dc.dataclass
class Directives:
    directives: typing.List[ModelDirective] = dc.field(default_factory=list)


# TODO(russt): load from url (using packagemap).
def load_scenario(
    *, filename=None, data=None, scenario_name=None, defaults=Scenario()
):
    """Implements the command-line handling logic for scenario data.
    Returns a `Scenario` object loaded from the given input arguments.

    Args:
        filename (optional): A yaml filename to load the scenario from.

        data (optional): A yaml string to load the scenario from. If both
            filename and string are specified, then the filename is parsed
            first, and then the string is _also_ parsed, potentially overwriting defaults from the filename..

        scenario_name (optional): The name of the scenario/child to load from
            the yaml file. If None, then the entire file is loaded.
    """
    result = defaults
    if filename:
        result = yaml_load_typed(
            schema=Scenario,
            filename=filename,
            child_name=scenario_name,
            defaults=result,
            retain_map_defaults=True,
        )
    if data:
        result = yaml_load_typed(
            schema=Scenario,
            data=data,
            child_name=scenario_name,
            defaults=result,
            retain_map_defaults=True,
        )
    return result


def add_directives(
    scenario,
    *,
    filename=None,
    data=None,
    scenario_name=None,
):
    d = Directives()
    if filename:
        d = yaml_load_typed(
            schema=Directives,
            filename=filename,
            child_name=scenario_name,
        )
    if data:
        d = yaml_load_typed(
            schema=Directives,
            data=data,
            child_name=scenario_name,
        )
    scenario.directives.extend(d.directives)
    return scenario


class MultiplexState(LeafSystem):
    def __init__(self, plant, model_instance_names):
        LeafSystem.__init__(self)
        total_states = 0
        for name in model_instance_names:
            model_instance = plant.GetModelInstanceByName(name)
            num_states = plant.num_multibody_states(model_instance)
            # The logic below assumes num_positions == num_velocities (though
            # it would be simple to generalize).
            assert plant.num_positions(model_instance) == plant.num_velocities(
                model_instance
            )
            self.DeclareVectorInputPort(name + ".state", num_states)
            total_states += num_states
        self.DeclareVectorOutputPort(
            "combined_state",
            total_states,
            self.CalcOutput,
        )

    def CalcOutput(self, context, output):
        # The order should should be [q, q, ..., v, v, ...].
        positions = np.array([])
        velocities = np.array([])
        for i in range(self.num_input_ports()):
            state = self.get_input_port(i).Eval(context)
            num_q = len(state) // 2
            positions = np.append(positions, state[:num_q])
            velocities = np.append(velocities, state[num_q:])
        output.SetFromVector(np.concatenate((positions, velocities)))


class DemultiplexInput(LeafSystem):
    def __init__(self, plant, model_instance_names):
        LeafSystem.__init__(self)
        total_inputs = 0
        for name in model_instance_names:
            num_actuators = plant.num_actuators(
                plant.GetModelInstanceByName(name)
            )
            self.DeclareVectorOutputPort(
                name + ".input",
                num_actuators,
                partial(
                    self.CalcOutput,
                    size=num_actuators,
                    start_index=total_inputs,
                ),
            )
            total_inputs += num_actuators
        self.DeclareVectorInputPort("combined_input", total_inputs)

    def CalcOutput(self, context, output, size, start_index):
        input = self.get_input_port().Eval(context)
        output.SetFromVector(input[start_index : start_index + size])


# TODO(russt): Use the c++ version pending https://github.com/RobotLocomotion/drake/issues/20055
def _ApplyDriverConfigSim(
    driver_config,
    model_instance_name,
    sim_plant,
    directives,
    models_from_directives_map,
    package_xmls,
    builder,
):
    if isinstance(driver_config, IiwaDriver):
        model_instance = sim_plant.GetModelInstanceByName(model_instance_name)
        num_iiwa_positions = sim_plant.num_positions(model_instance)

        # I need a PassThrough system so that I can export the input port.
        iiwa_position = builder.AddSystem(PassThrough(num_iiwa_positions))
        builder.ExportInput(
            iiwa_position.get_input_port(),
            model_instance_name + ".position",
        )
        builder.ExportOutput(
            iiwa_position.get_output_port(),
            model_instance_name + ".position_commanded",
        )

        # Export the iiwa "state" outputs.
        demux = builder.AddSystem(
            Demultiplexer(2 * num_iiwa_positions, num_iiwa_positions)
        )
        builder.Connect(
            sim_plant.get_state_output_port(model_instance),
            demux.get_input_port(),
        )
        builder.ExportOutput(
            demux.get_output_port(0),
            model_instance_name + ".position_measured",
        )
        builder.ExportOutput(
            demux.get_output_port(1),
            model_instance_name + ".velocity_estimated",
        )
        builder.ExportOutput(
            sim_plant.get_state_output_port(model_instance),
            model_instance_name + ".state_estimated",
        )

        # Make the plant for the iiwa controller to use.
        controller_plant = MultibodyPlant(time_step=sim_plant.time_step())
        # TODO: Add the correct IIWA model (introspected from MBP). See
        # build_iiwa_control in Drake for a slightly closer attempt.
        if num_iiwa_positions == 3:
            controller_iiwa = AddPlanarIiwa(controller_plant)
        else:
            controller_iiwa = AddIiwa(controller_plant)
        AddWsg(controller_plant, controller_iiwa, welded=True)
        controller_plant.Finalize()

        # Add the iiwa controller
        iiwa_controller = builder.AddSystem(
            InverseDynamicsController(
                controller_plant,
                kp=[100] * num_iiwa_positions,
                ki=[1] * num_iiwa_positions,
                kd=[20] * num_iiwa_positions,
                has_reference_acceleration=False,
            )
        )
        iiwa_controller.set_name(model_instance_name + ".controller")
        builder.Connect(
            sim_plant.get_state_output_port(model_instance),
            iiwa_controller.get_input_port_estimated_state(),
        )

        # Add in the feed-forward torque
        adder = builder.AddSystem(Adder(2, num_iiwa_positions))
        builder.Connect(
            iiwa_controller.get_output_port_control(),
            adder.get_input_port(0),
        )
        # Use a PassThrough to make the port optional (it will provide zero
        # values if not connected).
        torque_passthrough = builder.AddSystem(
            PassThrough([0] * num_iiwa_positions)
        )
        builder.Connect(
            torque_passthrough.get_output_port(), adder.get_input_port(1)
        )
        builder.ExportInput(
            torque_passthrough.get_input_port(),
            model_instance_name + ".feedforward_torque",
        )
        builder.Connect(
            adder.get_output_port(),
            sim_plant.get_actuation_input_port(model_instance),
        )

        # Add discrete derivative to command velocities.
        desired_state_from_position = builder.AddSystem(
            StateInterpolatorWithDiscreteDerivative(
                num_iiwa_positions,
                sim_plant.time_step(),
                suppress_initial_transient=True,
            )
        )
        desired_state_from_position.set_name(
            model_instance_name + ".desired_state_from_position"
        )
        builder.Connect(
            desired_state_from_position.get_output_port(),
            iiwa_controller.get_input_port_desired_state(),
        )
        builder.Connect(
            iiwa_position.get_output_port(),
            desired_state_from_position.get_input_port(),
        )

        # Export commanded torques.
        builder.ExportOutput(
            adder.get_output_port(),
            model_instance_name + ".torque_commanded",
        )
        builder.ExportOutput(
            adder.get_output_port(),
            model_instance_name + ".torque_measured",
        )

        builder.ExportOutput(
            sim_plant.get_generalized_contact_forces_output_port(
                model_instance
            ),
            model_instance_name + ".torque_external",
        )

    if isinstance(driver_config, SchunkWsgDriver):
        model_instance = sim_plant.GetModelInstanceByName(model_instance_name)
        # Wsg controller.
        wsg_controller = builder.AddSystem(SchunkWsgPositionController())
        wsg_controller.set_name(model_instance_name + ".controller")
        builder.Connect(
            wsg_controller.get_generalized_force_output_port(),
            sim_plant.get_actuation_input_port(model_instance),
        )
        builder.Connect(
            sim_plant.get_state_output_port(model_instance),
            wsg_controller.get_state_input_port(),
        )
        builder.ExportInput(
            wsg_controller.get_desired_position_input_port(),
            model_instance_name + ".position",
        )
        builder.ExportInput(
            wsg_controller.get_force_limit_input_port(),
            model_instance_name + ".force_limit",
        )
        wsg_mbp_state_to_wsg_state = builder.AddSystem(
            MakeMultibodyStateToWsgStateSystem()
        )
        builder.Connect(
            sim_plant.get_state_output_port(model_instance),
            wsg_mbp_state_to_wsg_state.get_input_port(),
        )
        builder.ExportOutput(
            wsg_mbp_state_to_wsg_state.get_output_port(),
            model_instance_name + ".state_measured",
        )
        builder.ExportOutput(
            wsg_controller.get_grip_force_output_port(),
            model_instance_name + ".force_measured",
        )

    if isinstance(driver_config, InverseDynamicsDriver):
        model_instance_names = model_instance_name.split("+")
        model_instances = [
            sim_plant.GetModelInstanceByName(n) for n in model_instance_names
        ]

        # Make the plant for the iiwa controller to use.
        controller_plant = MultibodyPlant(time_step=sim_plant.time_step())
        controller_directives = []
        for d in directives:
            if d.add_model and (d.add_model.name in model_instance_names):
                controller_directives.append(d)
            if (
                d.add_weld
                and (
                    ScopedName.Parse(d.add_weld.child).get_namespace()
                    in model_instance_names
                )
                and (
                    d.add_weld.parent == "world"
                    or ScopedName.Parse(d.add_weld.parent).get_namespace()
                    in model_instance_names
                )
            ):
                controller_directives.append(d)
        parser = Parser(controller_plant)
        for p in package_xmls:
            parser.package_map().AddPackageXml(p)
        ConfigureParser(parser)
        ProcessModelDirectives(
            directives=ModelDirectives(directives=controller_directives),
            parser=parser,
        )
        controller_plant.Finalize()

        # Add the controller
        # TODO(russt): Take the gains as parameters.
        controller = builder.AddSystem(
            InverseDynamicsController(
                controller_plant,
                kp=[100] * controller_plant.num_positions(),
                ki=[1] * controller_plant.num_positions(),
                kd=[20] * controller_plant.num_positions(),
                has_reference_acceleration=False,
            )
        )
        controller.set_name(model_instance_name + ".controller")
        if len(model_instances) == 1:
            builder.Connect(
                sim_plant.get_state_output_port(model_instances[0]),
                controller.get_input_port_estimated_state(),
            )
            builder.Connect(
                controller.get_output_port(),
                sim_plant.get_actuation_input_port(model_instances[0]),
            )
            builder.ExportOutput(
                sim_plant.get_state_output_port(model_instances[0]),
                model_instance_name + ".state_estimated",
            )
        else:
            combined_state = builder.AddSystem(
                MultiplexState(sim_plant, model_instance_names)
            )
            combined_state.set_name(model_instance_name + ".combined_state")
            combined_input = builder.AddSystem(
                DemultiplexInput(sim_plant, model_instance_names)
            )
            combined_input.set_name(model_instance_name + ".combined_input")
            for index, model_instance in enumerate(model_instances):
                builder.Connect(
                    sim_plant.get_state_output_port(model_instance),
                    combined_state.get_input_port(index),
                )
                builder.Connect(
                    combined_input.get_output_port(index),
                    sim_plant.get_actuation_input_port(model_instance),
                )
            builder.Connect(
                combined_state.get_output_port(),
                controller.get_input_port_estimated_state(),
            )
            builder.Connect(
                controller.get_output_port(), combined_input.get_input_port()
            )
            builder.ExportOutput(
                combined_state.get_output_port(),
                model_instance_name + ".state_estimated",
            )

        builder.ExportInput(
            controller.get_input_port_desired_state(),
            model_instance_name + ".desired_state",
        )


def _ApplyDriverConfigsSim(
    *,
    driver_configs,
    sim_plant,
    directives,
    models_from_directives,
    package_xmls,
    builder,
):
    models_from_directives_map = dict(
        [(info.model_name, info) for info in models_from_directives]
    )
    for model_instance_name, driver_config in driver_configs.items():
        _ApplyDriverConfigSim(
            driver_config,
            model_instance_name,
            sim_plant,
            directives,
            models_from_directives_map,
            package_xmls,
            builder,
        )


# TODO(russt): Remove this in favor of the Drake version once LCM becomes
# optional. https://github.com/RobotLocomotion/drake/issues/20055
def _ApplyCameraConfigSim(*, config, builder):
    if not (config.rgb or config.depth):
        return

    plant = builder.GetMutableSubsystemByName("plant")
    scene_graph = builder.GetMutableSubsystemByName("scene_graph")

    if sys.platform == "linux" and os.getenv("DISPLAY") is None:
        from pyvirtualdisplay import Display

        virtual_display = Display(visible=0, size=(1400, 900))
        virtual_display.start()

    if not scene_graph.HasRenderer(config.renderer_name):
        scene_graph.AddRenderer(
            config.renderer_name, MakeRenderEngineVtk(RenderEngineVtkParams())
        )

    # frame names in local variables:
    # P for parent frame, B for base frame, C for camera frame.

    # Extract the camera extrinsics from the config struct.
    P = (
        GetScopedFrameByName(plant, config.X_PB.base_frame)
        if config.X_PB.base_frame
        else plant.world_frame()
    )
    X_PC = config.X_PB.GetDeterministicValue()

    # convert mbp frame to geometry frame
    body = P.body()
    body_frame_id = plant.GetBodyFrameIdIfExists(body.index())
    # assert body_frame_id.has_value()

    X_BP = P.GetFixedPoseInBodyFrame()
    X_BC = X_BP @ X_PC

    # Extract camera intrinsics from the config struct.
    color_camera, depth_camera = config.MakeCameras()

    camera_sys = builder.AddSystem(
        RgbdSensor(
            parent_id=body_frame_id,
            X_PB=X_BC,
            depth_camera=depth_camera,
            show_window=False,
        )
    )
    camera_sys.set_name(f"rgbd_sensor_{config.name}")
    builder.Connect(
        scene_graph.get_query_output_port(), camera_sys.get_input_port()
    )

    # TODO(russt): export output ports
    builder.ExportOutput(
        camera_sys.color_image_output_port(), f"{config.name}.rgb_image"
    )
    builder.ExportOutput(
        camera_sys.depth_image_32F_output_port(), f"{config.name}.depth_image"
    )
    builder.ExportOutput(
        camera_sys.label_image_output_port(), f"{config.name}.label_image"
    )


def MakeHardwareStation(
    scenario: Scenario,
    meshcat: Meshcat = None,
    *,
    package_xmls: typing.List[str] = [],
    hardware: bool = False,
    parser_preload_callback: typing.Callable[[Parser], None] = None,
    parser_prefinalize_callback: typing.Callable[[Parser], None] = None,
):
    """
    If `hardware=False`, (the default) returns a HardwareStation diagram containing:
      - A MultibodyPlant with populated via the directives in `scenario`.
      - A SceneGraph
      - The default Drake visualizers
      - Any robot / sensors drivers specified in the YAML description.

    If `hardware=True`, returns a HardwareStationInterface diagram containing the network interfaces to communicate directly with the hardware drivers.

    Args:
        scenario: A Scenario structure, populated using the `load_scenario` method.

        meshcat: If not None, then AddDefaultVisualization will be added to the subdiagram using this meshcat instance.

        package_xmls: A list of package.xml file paths that will be passed to the parser, using Parser.AddPackageXml().
    """
    if hardware:
        return MakeHardwareStationInterface(
            scenario=scenario, meshcat=meshcat, package_xmls=package_xmls
        )

    builder = DiagramBuilder()

    # Create the multibody plant and scene graph.
    sim_plant, scene_graph = AddMultibodyPlant(
        config=scenario.plant_config, builder=builder
    )

    parser = Parser(sim_plant)
    for p in package_xmls:
        parser.package_map().AddPackageXml(p)
    ConfigureParser(parser)
    if parser_preload_callback:
        parser_preload_callback(parser)

    # Add model directives.
    added_models = ProcessModelDirectives(
        directives=ModelDirectives(directives=scenario.directives),
        parser=parser,
    )

    if parser_prefinalize_callback:
        parser_prefinalize_callback(parser)

    # Now the plant is complete.
    sim_plant.Finalize()

    # Add drivers.
    _ApplyDriverConfigsSim(
        driver_configs=scenario.model_drivers,
        sim_plant=sim_plant,
        directives=scenario.directives,
        models_from_directives=added_models,
        package_xmls=package_xmls,
        builder=builder,
    )

    # Add scene cameras.
    for _, camera in scenario.cameras.items():
        _ApplyCameraConfigSim(config=camera, builder=builder)

    # Add visualization.
    ApplyVisualizationConfig(scenario.visualization, builder, meshcat=meshcat)

    # Export "cheat" ports.
    builder.ExportOutput(scene_graph.get_query_output_port(), "query_object")
    builder.ExportOutput(
        sim_plant.get_contact_results_output_port(), "contact_results"
    )
    builder.ExportOutput(
        sim_plant.get_state_output_port(), "plant_continuous_state"
    )
    builder.ExportOutput(sim_plant.get_body_poses_output_port(), "body_poses")

    diagram = builder.Build()
    diagram.set_name("station")
    return diagram


# TODO(russt): Use the c++ version pending https://github.com/RobotLocomotion/drake/issues/20055
def ApplyDriverConfigInterface(
    driver_config,
    model_instance_name,
    plant,
    models_from_directives_map,
    lcm_buses,
    builder,
):
    if isinstance(driver_config, IiwaDriver):
        lcm = lcm_buses.Find(
            "Driver for " + model_instance_name, driver_config.lcm_bus
        )

        # Publish IIWA command.
        iiwa_command_sender = builder.AddSystem(IiwaCommandSender())
        # Note on publish period: IIWA driver won't respond faster than 200Hz
        iiwa_command_publisher = builder.AddSystem(
            LcmPublisherSystem.Make(
                channel="IIWA_COMMAND",
                lcm_type=lcmt_iiwa_command,
                lcm=lcm,
                publish_period=0.005,
                use_cpp_serializer=True,
            )
        )
        iiwa_command_publisher.set_name(
            model_instance_name + ".command_publisher"
        )
        builder.ExportInput(
            iiwa_command_sender.get_position_input_port(),
            model_instance_name + ".position",
        )
        builder.ExportInput(
            iiwa_command_sender.get_torque_input_port(),
            model_instance_name + ".feedforward_torque",
        )
        builder.Connect(
            iiwa_command_sender.get_output_port(),
            iiwa_command_publisher.get_input_port(),
        )
        # Receive IIWA status and populate the output ports.
        iiwa_status_receiver = builder.AddSystem(IiwaStatusReceiver())
        iiwa_status_subscriber = builder.AddSystem(
            LcmSubscriberSystem.Make(
                channel="IIWA_STATUS",
                lcm_type=lcmt_iiwa_status,
                lcm=lcm,
                use_cpp_serializer=True,
                wait_for_message_on_initialization_timeout=10,
            )
        )
        iiwa_status_subscriber.set_name(
            model_instance_name + ".status_subscriber"
        )

        # builder.Connect(
        #    iiwa_status_receiver.get_position_measured_output_port(),
        #    to_pose.get_input_port(),
        # )

        builder.ExportOutput(
            iiwa_status_receiver.get_position_commanded_output_port(),
            model_instance_name + ".position_commanded",
        )
        builder.ExportOutput(
            iiwa_status_receiver.get_position_measured_output_port(),
            model_instance_name + ".position_measured",
        )
        builder.ExportOutput(
            iiwa_status_receiver.get_velocity_estimated_output_port(),
            model_instance_name + ".velocity_estimated",
        )
        builder.ExportOutput(
            iiwa_status_receiver.get_torque_commanded_output_port(),
            model_instance_name + ".torque_commanded",
        )
        builder.ExportOutput(
            iiwa_status_receiver.get_torque_measured_output_port(),
            model_instance_name + ".torque_measured",
        )
        builder.ExportOutput(
            iiwa_status_receiver.get_torque_external_output_port(),
            model_instance_name + ".torque_external",
        )
        builder.Connect(
            iiwa_status_subscriber.get_output_port(),
            iiwa_status_receiver.get_input_port(),
        )
    if isinstance(driver_config, SchunkWsgDriver):
        lcm = lcm_buses.Find(
            "Driver for " + model_instance_name, driver_config.lcm_bus
        )

        # Publish WSG command.
        wsg_command_sender = builder.AddSystem(SchunkWsgCommandSender())
        wsg_command_publisher = builder.AddSystem(
            LcmPublisherSystem.Make(
                channel="SCHUNK_WSG_COMMAND",
                lcm_type=lcmt_schunk_wsg_command,
                lcm=lcm,
                publish_period=0.05,  # Schunk driver won't respond faster than 20Hz
                use_cpp_serializer=True,
            )
        )
        wsg_command_publisher.set_name(
            model_instance_name + ".command_publisher"
        )
        builder.ExportInput(
            wsg_command_sender.get_position_input_port(),
            model_instance_name + ".position",
        )
        builder.ExportInput(
            wsg_command_sender.get_force_limit_input_port(),
            model_instance_name + ".force_limit",
        )
        builder.Connect(
            wsg_command_sender.get_output_port(0),
            wsg_command_publisher.get_input_port(),
        )

        # Receive WSG status and populate the output ports.
        wsg_status_receiver = builder.AddSystem(SchunkWsgStatusReceiver())
        wsg_status_subscriber = builder.AddSystem(
            LcmSubscriberSystem.Make(
                channel="SCHUNK_WSG_STATUS",
                lcm_type=lcmt_schunk_wsg_status,
                lcm=lcm,
                use_cpp_serializer=True,
                wait_for_message_on_initialization_timeout=10,
            )
        )
        wsg_status_subscriber.set_name(
            model_instance_name + ".status_subscriber"
        )
        builder.ExportOutput(
            wsg_status_receiver.get_state_output_port(),
            model_instance_name + ".state_measured",
        )
        builder.ExportOutput(
            wsg_status_receiver.get_force_output_port(),
            model_instance_name + ".force_measured",
        )
        builder.Connect(
            wsg_status_subscriber.get_output_port(),
            wsg_status_receiver.get_input_port(0),
        )


def ApplyDriverConfigsInterface(
    *, driver_configs, plant, models_from_directives, lcm_buses, builder
):
    models_from_directives_map = dict(
        [(info.model_name, info) for info in models_from_directives]
    )
    for model_instance_name, driver_config in driver_configs.items():
        ApplyDriverConfigInterface(
            driver_config,
            model_instance_name,
            plant,
            models_from_directives_map,
            lcm_buses,
            builder,
        )


def MakeHardwareStationInterface(
    scenario: Scenario,
    meshcat: Meshcat = None,
    *,
    package_xmls: typing.List[str] = [],
):
    builder = DiagramBuilder()

    # Visualization
    scene_graph = builder.AddNamedSystem("scene_graph", SceneGraph())
    plant = MultibodyPlant(time_step=scenario.plant_config.time_step)
    ApplyMultibodyPlantConfig(scenario.plant_config, plant)
    plant.RegisterAsSourceForSceneGraph(scene_graph)
    parser = Parser(plant)
    for p in package_xmls:
        parser.package_map().AddPackageXml(p)
    ConfigureParser(parser)

    # Add model directives.
    added_models = ProcessModelDirectives(
        directives=ModelDirectives(directives=scenario.directives),
        parser=parser,
    )

    # Now the plant is complete.
    plant.Finalize()

    # to_pose = builder.AddSystem(MultibodyPositionToGeometryPose(plant))
    # builder.Connect(
    #     to_pose.get_output_port(),
    #     scene_graph.get_source_pose_port(plant.get_source_id()),
    # )

    # config = VisualizationConfig()
    # config.publish_contacts = False
    # config.publish_inertia = False
    # ApplyVisualizationConfig(
    #     config, builder=builder, plant=plant, meshcat=meshcat
    # )

    # Add LCM buses. (The simulator will handle polling the network for new
    # messages and dispatching them to the receivers, i.e., "pump" the bus.)
    lcm_buses = ApplyLcmBusConfig(
        lcm_buses=scenario.lcm_buses, builder=builder
    )

    # Add drivers.
    ApplyDriverConfigsInterface(
        driver_configs=scenario.model_drivers,
        plant=plant,
        models_from_directives=added_models,
        lcm_buses=lcm_buses,
        builder=builder,
    )

    # Add cameras.

    diagram = builder.Build()
    diagram.set_name("HardwareStationInterface")
    return diagram


class ExtractPose(LeafSystem):
    def __init__(
        self, body_poses_output_port, body_index, X_BA=RigidTransform()
    ):
        LeafSystem.__init__(self)
        self.body_index = body_index
        self.DeclareAbstractInputPort(
            "poses",
            body_poses_output_port.Allocate(),
        )
        self.DeclareAbstractOutputPort(
            "pose",
            lambda: AbstractValue.Make(RigidTransform()),
            self.CalcOutput,
        )
        self.X_BA = X_BA

    def CalcOutput(self, context, output):
        poses = self.EvalAbstractInput(context, 0).get_value()
        pose = poses[int(self.body_index)] @ self.X_BA
        output.get_mutable_value().set(pose.rotation(), pose.translation())


def AddPointClouds(
    *,
    scenario: Scenario,
    station: Diagram,
    builder: DiagramBuilder,
    poses_output_port: OutputPort = None,
    meshcat: Meshcat = None,
):
    """
    Adds one DepthImageToPointCloud system to the `builder` for each camera in `scenario`, and connects it to the respective camera station output ports.

    Args:
        scenario: A Scenario structure, populated using the `load_scenario` method.

        station: A HardwareStation system (e.g. from MakeHardwareStation) that has already been added to `builder`.

        builder: The DiagramBuilder containing `station` into which the new systems will be added.

        poses_output_port: (optional) HardwareStation will have a body_poses output port iff it was created with `hardware=False`. Alternatively, one could create a MultibodyPositionsToGeometryPoses system to consume the position measurements; this optional input can be used to support that workflow.

        meshcat: If not None, then a MeshcatPointCloudVisualizer will be added to the builder using this meshcat instance.
    """
    to_point_cloud = dict()
    for _, config in scenario.cameras.items():
        if not config.depth:
            return

        plant = station.GetSubsystemByName("plant")
        # frame names in local variables:
        # P for parent frame, B for base frame, C for camera frame.

        # Extract the camera extrinsics from the config struct.
        P = (
            GetScopedFrameByName(plant, config.X_PB.base_frame)
            if config.X_PB.base_frame
            else plant.world_frame()
        )
        X_PC = config.X_PB.GetDeterministicValue()

        # convert mbp frame to geometry frame
        body = P.body()
        plant.GetBodyFrameIdIfExists(body.index())
        # assert body_frame_id.has_value()

        X_BP = P.GetFixedPoseInBodyFrame()
        X_BC = X_BP @ X_PC

        intrinsics = CameraInfo(
            config.width,
            config.height,
            config.focal_x(),
            config.focal_y(),
            config.principal_point()[0],
            config.principal_point()[1],
        )

        to_point_cloud[config.name] = builder.AddSystem(
            DepthImageToPointCloud(
                camera_info=intrinsics,
                fields=BaseField.kXYZs | BaseField.kRGBs,
            )
        )
        to_point_cloud[config.name].set_name(f"{config.name}.point_cloud")

        builder.Connect(
            station.GetOutputPort(f"{config.name}.depth_image"),
            to_point_cloud[config.name].depth_image_input_port(),
        )
        builder.Connect(
            station.GetOutputPort(f"{config.name}.rgb_image"),
            to_point_cloud[config.name].color_image_input_port(),
        )

        if poses_output_port is None:
            # Note: this is a cheat port; it will only work in single process
            # mode.
            poses_output_port = station.GetOutputPort("body_poses")

        camera_pose = builder.AddSystem(
            ExtractPose(poses_output_port, body.index(), X_BC)
        )
        camera_pose.set_name(f"{config.name}.pose")
        builder.Connect(
            poses_output_port,
            camera_pose.get_input_port(),
        )
        builder.Connect(
            camera_pose.get_output_port(),
            to_point_cloud[config.name].GetInputPort("camera_pose"),
        )

        if meshcat:
            # Send the point cloud to meshcat for visualization, too.
            point_cloud_visualizer = builder.AddSystem(
                MeshcatPointCloudVisualizer(meshcat, f"{config.name}.cloud")
            )
            builder.Connect(
                to_point_cloud[config.name].point_cloud_output_port(),
                point_cloud_visualizer.cloud_input_port(),
            )

    return to_point_cloud
