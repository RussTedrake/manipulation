import dataclasses as dc
import typing

import numpy as np

from pydrake.all import (
    Adder,
    AddMultibodyPlant,
    ApplyLcmBusConfig,
    ApplyMultibodyPlantConfig,
    ApplyVisualizationConfig,
    CameraConfig,
    Demultiplexer,
    DiagramBuilder,
    DrakeLcmParams,
    IiwaCommandSender,
    IiwaDriver,
    IiwaStatusReceiver,
    InverseDynamicsController,
    LcmPublisherSystem,
    LcmSubscriberSystem,
    MakeMultibodyStateToWsgStateSystem,
    Meshcat,
    ModelDirective,
    ModelDirectives,
    MultibodyPlant,
    MultibodyPlantConfig,
    Parser,
    PassThrough,
    ProcessModelDirectives,
    SceneGraph,
    SchunkWsgDriver,
    SchunkWsgPositionController,
    SchunkWsgCommandSender,
    SchunkWsgStatusReceiver,
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


def load_scenario(*, filename=None, data=None, scenario_name=None):
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
    result = Scenario()
    if filename:
        result = yaml_load_typed(
            schema=Scenario,
            filename=filename,
            child_name=scenario_name,
            defaults=result,
        )
    if data:
        result = yaml_load_typed(schema=Scenario, data=data, defaults=result)
    return result


# TODO(russt): Use the c++ version pending https://github.com/RobotLocomotion/drake/issues/20055
def ApplyDriverConfigSim(
    driver_config,
    model_instance_name,
    sim_plant,
    models_from_directives_map,
    builder,
):
    model_instance = sim_plant.GetModelInstanceByName(model_instance_name)
    if isinstance(driver_config, IiwaDriver):
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


def ApplyDriverConfigsSim(
    *, driver_configs, sim_plant, models_from_directives, builder
):
    models_from_directives_map = dict(
        [(info.model_name, info) for info in models_from_directives]
    )
    for model_instance_name, driver_config in driver_configs.items():
        ApplyDriverConfigSim(
            driver_config,
            model_instance_name,
            sim_plant,
            models_from_directives_map,
            builder,
        )


def MakeHardwareStation(
    scenario: Scenario,
    meshcat: Meshcat = None,
    *,
    package_xmls: typing.List[str] = [],
    hardware: bool = False,
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

    # Add model directives.
    added_models = ProcessModelDirectives(
        directives=ModelDirectives(directives=scenario.directives),
        parser=parser,
    )

    # Now the plant is complete.
    sim_plant.Finalize()

    # Add drivers.
    ApplyDriverConfigsSim(
        driver_configs=scenario.model_drivers,
        sim_plant=sim_plant,
        models_from_directives=added_models,
        builder=builder,
    )

    # TODO(russt): Add scene cameras. https://github.com/RobotLocomotion/drake/issues/20055

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
    diagram.set_name("HardwareStation")
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
