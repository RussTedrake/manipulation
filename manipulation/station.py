import dataclasses as dc
import os
import sys
import typing
import warnings
from copy import copy
from functools import partial

import numpy as np
from drake import (
    lcmt_iiwa_command,
    lcmt_iiwa_status,
    lcmt_image_array,
    lcmt_schunk_wsg_command,
    lcmt_schunk_wsg_status,
)
from pydrake.all import (
    ApplyCameraConfig,
    ApplyLcmBusConfig,
    ApplyMultibodyPlantConfig,
    ApplyVisualizationConfig,
    BaseField,
    CameraConfig,
    CameraInfo,
    DepthImageToPointCloud,
    Diagram,
    DiagramBuilder,
    DrakeLcmParams,
    FlattenModelDirectives,
    Gain,
    GetScopedFrameByName,
    IiwaCommandSender,
    IiwaControlMode,
    IiwaDriver,
    IiwaStatusReceiver,
    InverseDynamicsController,
    Joint,
    LcmBuses,
    LcmImageArrayToImages,
    LcmPublisherSystem,
    LcmSubscriberSystem,
    LeafSystem,
    MakeMultibodyStateToWsgStateSystem,
    Meshcat,
    MeshcatPointCloudVisualizer,
    ModelDirective,
    ModelDirectives,
    ModelInstanceIndex,
    ModelInstanceInfo,
    MultibodyPlant,
    MultibodyPlantConfig,
    MultibodyPositionToGeometryPose,
    Multiplexer,
    OutputPort,
    ParseIiwaControlMode,
    Parser,
    PdControllerGains,
    ProcessModelDirectives,
    RigidTransform,
    RobotDiagram,
    RobotDiagramBuilder,
    SceneGraph,
    SchunkWsgCommandSender,
    SchunkWsgDriver,
    SchunkWsgPositionController,
    SchunkWsgStatusReceiver,
    SharedPointerSystem,
    SimIiwaDriver,
    SimulatorConfig,
    VisualizationConfig,
    WeldJoint,
    ZeroForceDriver,
    position_enabled,
    torque_enabled,
)
from pydrake.common.yaml import yaml_load_typed

from manipulation.directives_tree import DirectivesTree
from manipulation.systems import ExtractPose
from manipulation.utils import ConfigureParser


@dc.dataclass
class JointPidControllerGains:
    """Defines the Proportional-Integral-Derivative gains for a single joint.

    Args:
        kp: The proportional gain.
        ki: The integral gain.
        kd: The derivative gain.
    """

    kp: float = 100  # Position gain
    ki: float = 1  # Integral gain
    kd: float = 20  # Velocity gain


@dc.dataclass
class InverseDynamicsDriver:
    """A simulation-only driver that adds the InverseDynamicsController to the
    station and exports the output ports. Multiple model instances can be driven with a
    single controller using `instance_name1+instance_name2` as the key; the output ports
    will be named similarly."""

    # Must have one element for every (named) actuator in the model_instance.
    gains: typing.Mapping[str, JointPidControllerGains] = dc.field(default_factory=dict)


@dc.dataclass
class JointPdControllerGains:
    """Defines the Proportional-Derivative gains for a single joint.

    Args:
        kp: The proportional gain.
        kd: The derivative gain.
    """

    kp: float = 0  # Position gain
    kd: float = 0  # Velocity gain


@dc.dataclass
class JointStiffnessDriver:
    """A simulation-only driver that sets up MultibodyPlant to act as if it is
    being controlled with a JointStiffnessController. The MultibodyPlant must
    be using SAP as the (discrete-time) contact solver.

    Args:
        gains: A mapping of {actuator_name: JointPdControllerGains} for each
            actuator that should be controlled.
        hand_model_name: If set, then the gravity compensation will be turned
            off for this model instance (e.g. for a hand).
    """

    # Must have one element for every (named) actuator in the model_instance.
    gains: typing.Mapping[str, JointPdControllerGains] = dc.field(default_factory=dict)

    hand_model_name: str = ""


@dc.dataclass
class Scenario:
    """Defines the YAML format for a (possibly stochastic) scenario to be
    simulated.

    Args:
        random_seed: Random seed for any random elements in the scenario. The
            seed is always deterministic in the `Scenario`; a caller who wants
            randomness must populate this value from their own randomness.
        simulation_duration: The maximum simulation time (in seconds).  The
            simulator will attempt to run until this time and then terminate.
        simulator_config: Simulator configuration (integrator and publisher
            parameters).
        plant_config: Plant configuration (time step and contact parameters).
        directives: All of the fully deterministic elements of the simulation.
        lcm_buses: A map of {bus_name: lcm_params} for LCM transceivers to be
            used by drivers, sensors, etc.
        model_drivers: For actuated models, specifies where each model's
            actuation inputs come from, keyed on the ModelInstance name.
        cameras: Cameras to add to the scene (and broadcast over LCM). The key
            for each camera is a helpful mnemonic, but does not serve a
            technical role. The CameraConfig::name field is still the name that
            will appear in the Diagram artifacts.
        visualization: Visualization configuration.
    """

    random_seed: int = 0

    simulation_duration: float = np.inf

    simulator_config: SimulatorConfig = SimulatorConfig(
        max_step_size=0.01,
        use_error_control=False,
        accuracy=1.0e-2,
    )

    plant_config: MultibodyPlantConfig = MultibodyPlantConfig(
        discrete_contact_approximation="sap"
    )

    directives: typing.List[ModelDirective] = dc.field(default_factory=list)

    # Opt-out of LCM by default.
    lcm_buses: typing.Mapping[str, DrakeLcmParams] = dc.field(
        default_factory=lambda: dict(default=DrakeLcmParams(lcm_url="memq://null"))
    )

    model_drivers: typing.Mapping[
        str,
        typing.Union[
            IiwaDriver,
            InverseDynamicsDriver,
            JointStiffnessDriver,
            SchunkWsgDriver,
            ZeroForceDriver,
        ],
    ] = dc.field(default_factory=dict)

    cameras: typing.Mapping[str, CameraConfig] = dc.field(default_factory=dict)

    camera_ids: typing.Mapping[str, str] = dc.field(default_factory=dict)

    visualization: VisualizationConfig = VisualizationConfig()


@dc.dataclass
class Directives:
    """Defines the YAML format for directives to be applied to a `Scenario`.

    Args:
        directives: A list of `ModelDirective` objects.
    """

    directives: typing.List[ModelDirective] = dc.field(default_factory=list)


def load_scenario(
    *,
    filename: str | None = None,
    data: str | None = None,
    scenario_name: str | None = None,
    defaults: Scenario = Scenario(),
):
    warnings.warn("load_scenario is deprecated. Use LoadScenario instead.")
    return LoadScenario(
        filename=filename, data=data, scenario_name=scenario_name, defaults=defaults
    )


def add_scenario(
    *,
    filename: str | None = None,
    data: str | None = None,
    scenario_name: str | None = None,
    defaults: Scenario = Scenario(),
):
    warnings.warn("add_directives is deprecated. Use AppendDirectives instead.")
    return AppendDirectives(
        filename=filename, data=data, scenario_name=scenario_name, defaults=defaults
    )


# TODO(russt): load from url (using packagemap).
def LoadScenario(
    *,
    filename: str | None = None,
    data: str | None = None,
    scenario_name: str | None = None,
    defaults: Scenario = Scenario(),
) -> Scenario:
    """Implements the command-line handling logic for scenario data.

    Args:
        filename: A yaml filename to load the scenario from.

        data: A yaml _string_ to load the scenario from. If both filename and string are
            specified, then the filename is parsed first, and then the string is _also_
            parsed, potentially overwriting defaults from the filename. Note: this will
            not append additional `directives`, it will replace them; see
            AppendDirectives.

        scenario_name: The name of the scenario/child to load from the yaml file. If
            None, then the entire file is loaded.

        defaults: A `Scenario` object to use as the default values.

    Returns:
        A `Scenario` object loaded from the given input arguments.
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


def AppendDirectives(
    scenario: Scenario,
    *,
    filename: str | None = None,
    data: str | None = None,
    scenario_name: str | None = None,
) -> Scenario:
    """Append additional directives to an existing scenario.

    Args:
        scenario: The scenario to append to.

        filename: A yaml filename to load the directives from.

        data: A yaml string to load the directives from. If both filename and string are
            specified, then the filename is parsed first, and then the string is _also_
            parsed, presumably overwriting any directives from the filename.

        scenario_name: The name of the scenario/child to load from the yaml file. If
            None, then the entire file is loaded.

    Returns:
        The scenario with the additional directives appended.
    """
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


def _FreezeChildren(
    plant: MultibodyPlant,
    children_to_freeze: typing.List[str],
) -> None:
    """
    Freeze the joints of the given children in the plant.

    Freezing really means removing a joint belonging to any of the children,
    and replacing it with a weld joint.
    """
    if len(children_to_freeze) == 0:
        return

    # Enumerate joints that need to be frozen i.e. removed and replaced by weld
    # joints. These are joints of child instances that are not already welds.
    joints_to_freeze: typing.Set[Joint] = set()
    for child_instance_name in children_to_freeze:
        child_instance = plant.GetModelInstanceByName(child_instance_name)
        for joint_index in plant.GetJointIndices(child_instance):
            joint = plant.get_joint(joint_index)
            if joint.type_name() != "weld":
                joints_to_freeze.add(joint)

    # Before removing joints, we need to remove associated actuators.
    for actuator_index in plant.GetJointActuatorIndices():
        actuator = plant.get_joint_actuator(actuator_index)
        if actuator.joint() in joints_to_freeze:
            plant.RemoveJointActuator(actuator)

    # Remove non-weld joints and replace them with weld joints.
    for joint in joints_to_freeze:
        weld = WeldJoint(
            joint.name(),
            joint.frame_on_parent(),
            joint.frame_on_child(),
            RigidTransform(),
        )
        plant.RemoveJoint(joint)
        plant.AddJoint(weld)


def _PopulatePlantOrDiagram(
    plant: MultibodyPlant,
    parser: Parser,
    scenario: Scenario,
    model_instance_names: typing.List[str] | None,
    add_frozen_child_instances: bool = True,
    package_xmls: typing.List[str] = [],
    parser_preload_callback: typing.Callable[[Parser], None] | None = None,
    parser_prefinalize_callback: typing.Callable[[Parser], None] | None = None,
) -> None:
    """See MakeMultibodyPlant and MakeRobotDiagram for details."""
    if model_instance_names is None:
        assert not add_frozen_child_instances, (
            "add_frozen_child_instances cannot be used without specifying "
            "model_instance_names."
        )

    ApplyMultibodyPlantConfig(scenario.plant_config, plant)
    for p in package_xmls:
        parser.package_map().AddPackageXml(p)
    ConfigureParser(parser)
    if parser_preload_callback:
        parser_preload_callback(parser)

    # Make the plant for the iiwa controller to use.
    flattened_directives = FlattenModelDirectives(
        ModelDirectives(directives=scenario.directives), parser.package_map()
    ).directives

    if not model_instance_names is None:
        tree = DirectivesTree(flattened_directives)
        directives = tree.GetDirectivesFromModelsToRoot(model_instance_names)
        children_to_freeze = set()

        if add_frozen_child_instances:
            children_to_freeze, additional_directives = (
                tree.GetWeldedDescendantsAndDirectives(model_instance_names)
            )
            directives.update(additional_directives)

        sorted_directives = tree.TopologicallySortDirectives(directives)
    else:
        # Add all model instances.
        sorted_directives = flattened_directives
        children_to_freeze = set()

    ProcessModelDirectives(
        directives=ModelDirectives(directives=sorted_directives),
        parser=parser,
    )

    _FreezeChildren(plant, children_to_freeze)

    if parser_prefinalize_callback:
        parser_prefinalize_callback(parser)

    plant.Finalize()


def MakeMultibodyPlant(
    scenario: Scenario,
    *,
    model_instance_names: typing.List[str] | None = None,
    add_frozen_child_instances: bool = False,
    package_xmls: typing.List[str] = [],
    parser_preload_callback: typing.Callable[[Parser], None] | None = None,
    parser_prefinalize_callback: typing.Callable[[Parser], None] | None = None,
) -> MultibodyPlant:
    """Use a scenario to create a MultibodyPlant. This is intended, e.g., to facilitate
    easily building subsets of a scenario, for instance, to make a plant for a
    controller.

    Args:
        scenario: A Scenario structure, populated using the `load_scenario`
            method.

        model_instance_names: If specified, then only the named model instances
            will be added to the plant. Otherwise, all model instances will be added.
            `add_weld` directives connecting added model instances to each other or to
            the world are also preserved.

        add_frozen_child_instances: If True and model_instance_names is not None, then
            model_instances that are not listed in model_instance_names, but are welded
            to a model_instance that is listed, will be added to the plant; with all
            joints replaced by welded joints.

        package_xmls: A list of package.xml file paths that will be passed to
            the parser, using Parser.AddPackageXml().

        parser_preload_callback: A callback function that will be called after
            the Parser is created, but before any directives are processed. This can be
            used to add additional packages to the parser, or to add additional model
            directives.

        parser_prefinalize_callback: A callback function that will be called
            after the directives are processed, but before the plant is finalized. This
            can be used to add additional model directives.

    Returns:
        A MultibodyPlant populated from (a subset of) the scenario.
    """
    plant = MultibodyPlant(time_step=scenario.plant_config.time_step)
    parser = Parser(plant)
    _PopulatePlantOrDiagram(
        plant,
        parser,
        scenario,
        model_instance_names,
        add_frozen_child_instances,
        package_xmls,
        parser_preload_callback,
        parser_prefinalize_callback,
    )
    return plant


def MakeRobotDiagram(
    scenario: Scenario,
    *,
    model_instance_names: typing.List[str] | None = None,
    add_frozen_child_instances: bool = True,
    package_xmls: typing.List[str] = [],
    parser_preload_callback: typing.Callable[[Parser], None] | None = None,
    parser_prefinalize_callback: typing.Callable[[Parser], None] | None = None,
) -> RobotDiagram:
    """Use a scenario to create a RobotDiagram (MultibodyPlant + SceneGraph). This is
    intended, e.g., to facilitate easily building subsets of a scenario, for instance,
    to make a plant for a controller which needs to make collision queries.

    Args:
        scenario: A Scenario structure, populated using the `load_scenario`
            method.

        model_instance_names: If specified, then only the named model instances
            will be added to the plant. Otherwise, all model instances will be added.

        add_frozen_child_instances: If True and model_instance_names is not None, then
            model_instances that are not listed in model_instance_names, but are welded
            to a model_instance that is listed, will be added to the plant; with all
            joints replaced by welded joints.

        package_xmls: A list of package.xml file paths that will be passed to
            the parser, using Parser.AddPackageXml().

        parser_preload_callback: A callback function that will be called after
            the Parser is created, but before any directives are processed. This can be
            used to add additional packages to the parser, or to add additional model
            directives.

        parser_prefinalize_callback: A callback function that will be called
            after the directives are processed, but before the plant is finalized. This
            can be used to add additional model directives.

    Returns:
        A RobotDiagram populated from (a subset of) the scenario.
    """
    robot_builder = RobotDiagramBuilder(time_step=scenario.plant_config.time_step)
    plant = robot_builder().builder().plant()
    parser = robot_builder().parser()
    _PopulatePlantOrDiagram(
        plant,
        parser,
        scenario,
        model_instance_names,
        add_frozen_child_instances,
        package_xmls,
        parser_preload_callback,
        parser_prefinalize_callback,
    )
    return robot_builder.Build()


class _MultiplexState(LeafSystem):
    def __init__(self, plant: MultibodyPlant, model_instance_names: typing.List[str]):
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


class _DemultiplexInput(LeafSystem):
    def __init__(self, plant: MultibodyPlant, model_instance_names: typing.List[str]):
        LeafSystem.__init__(self)
        total_inputs = 0
        for name in model_instance_names:
            num_actuators = plant.num_actuators(plant.GetModelInstanceByName(name))
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
    driver_config,  # See Scenario.model_drivers for typing
    model_instance_name: str,
    sim_plant: MultibodyPlant,
    scenario: Scenario,
    package_xmls: typing.List[str],
    builder: DiagramBuilder,
) -> None:
    if isinstance(driver_config, IiwaDriver):
        model_instance = sim_plant.GetModelInstanceByName(model_instance_name)
        num_iiwa_positions = sim_plant.num_positions(model_instance)

        # Make the plant for the iiwa controller to use.
        controller_plant = MakeMultibodyPlant(
            scenario=scenario,
            model_instance_names=[model_instance_name],
            add_frozen_child_instances=True,
            package_xmls=package_xmls,
        )
        # Keep the controller plant alive during the Diagram lifespan.
        builder.AddNamedSystem(
            f"{model_instance_name}_controller_plant_pointer_system",
            SharedPointerSystem(controller_plant),
        )

        control_mode = ParseIiwaControlMode(driver_config.control_mode)
        sim_iiwa_driver = SimIiwaDriver.AddToBuilder(
            plant=sim_plant,
            iiwa_instance=model_instance,
            controller_plant=controller_plant,
            builder=builder,
            ext_joint_filter_tau=0.01,
            desired_iiwa_kp_gains=np.full(num_iiwa_positions, 100),
            control_mode=control_mode,
        )
        for i in range(sim_iiwa_driver.num_input_ports()):
            port = sim_iiwa_driver.get_input_port(i)
            if not builder.IsConnectedOrExported(port):
                builder.ExportInput(port, f"{model_instance_name}.{port.get_name()}")
        for i in range(sim_iiwa_driver.num_output_ports()):
            port = sim_iiwa_driver.get_output_port(i)
            builder.ExportOutput(port, f"{model_instance_name}.{port.get_name()}")

    elif isinstance(driver_config, SchunkWsgDriver):
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

    elif isinstance(driver_config, InverseDynamicsDriver):
        model_instance_names = model_instance_name.split("+")
        model_instances = [
            sim_plant.GetModelInstanceByName(n) for n in model_instance_names
        ]

        # Make the controller plant.
        controller_plant = MakeMultibodyPlant(
            scenario=scenario,
            model_instance_names=model_instance_names,
            add_frozen_child_instances=True,
            package_xmls=package_xmls,
        )

        # Add the controller

        # When using multiple model instances, the model instance name must be prefixed.
        # The strings should take the form {model_instance_name}_{joint_actuator_name}, as
        # prescribed by MultiBodyPlant::GetActuatorNames().
        add_model_instance_prefix = len(model_instance_names) > 1
        actuator_names = controller_plant.GetActuatorNames(add_model_instance_prefix)

        # Check that all actuator names are valid.
        for actuator_name in driver_config.gains.keys():
            if actuator_name not in actuator_names:
                raise ValueError(
                    f"Actuator '{actuator_name}' not found. Valid names are: {actuator_names}"
                )

        # Get gains for each joint from the config. Use default gains if it doesn't exist in the config.
        default_gains = JointPidControllerGains()
        gains: typing.List[JointPidControllerGains] = []
        for actuator_name in actuator_names:
            joint_gains = driver_config.gains.get(actuator_name, default_gains)
            gains.append(joint_gains)

        controller = builder.AddSystem(
            InverseDynamicsController(
                controller_plant,
                kp=[joint_gains.kp for joint_gains in gains],
                ki=[joint_gains.ki for joint_gains in gains],
                kd=[joint_gains.kd for joint_gains in gains],
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
                _MultiplexState(sim_plant, model_instance_names)
            )
            combined_state.set_name(model_instance_name + ".combined_state")
            combined_input = builder.AddSystem(
                _DemultiplexInput(sim_plant, model_instance_names)
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

    elif isinstance(driver_config, JointStiffnessDriver):
        model_instance = sim_plant.GetModelInstanceByName(model_instance_name)

        # PD gains and gravity comp are set in ApplyPrefinalizeDriverConfigsSim

        builder.ExportInput(
            sim_plant.get_desired_state_input_port(model_instance),
            model_instance_name + ".desired_state",
        )
        builder.ExportInput(
            sim_plant.get_actuation_input_port(model_instance),
            model_instance_name + ".tau_feedforward",
        )
        builder.ExportOutput(
            sim_plant.get_state_output_port(model_instance),
            model_instance_name + ".state_estimated",
        )


def _ApplyDriverConfigsSim(
    *,
    driver_configs,  # See Scenario.model_drivers for typing
    sim_plant: MultibodyPlant,
    scenario: Scenario,
    package_xmls: typing.List[str],
    builder: DiagramBuilder,
) -> None:
    for model_instance_name, driver_config in driver_configs.items():
        _ApplyDriverConfigSim(
            driver_config=driver_config,
            model_instance_name=model_instance_name,
            sim_plant=sim_plant,
            scenario=scenario,
            package_xmls=package_xmls,
            builder=builder,
        )


def _ApplyPrefinalizeDriverConfigSim(
    driver_config,  # See Scenario.model_drivers for typing
    model_instance_name: str,
    sim_plant: MultibodyPlant,
    directives: typing.List[ModelDirective],
    models_from_directives_map: typing.Mapping[str, typing.List[ModelInstanceInfo]],
    package_xmls: typing.List[str],
    builder: DiagramBuilder,
) -> None:
    if isinstance(driver_config, JointStiffnessDriver):
        model_instance = sim_plant.GetModelInstanceByName(model_instance_name)

        # Set PD gains.
        for name, gains in driver_config.gains.items():
            actuator = sim_plant.GetJointActuatorByName(name, model_instance)
            actuator.set_controller_gains(PdControllerGains(p=gains.kp, d=gains.kd))

        # Turn off gravity to model (perfect) gravity compensation.
        sim_plant.set_gravity_enabled(model_instance, False)
        if driver_config.hand_model_name:
            sim_plant.set_gravity_enabled(
                sim_plant.GetModelInstanceByName(driver_config.hand_model_name),
                False,
            )


def _ApplyPrefinalizeDriverConfigsSim(
    *,
    driver_configs,  # See Scenario.model_drivers for typing
    sim_plant: MultibodyPlant,
    directives: typing.List[ModelDirective],
    models_from_directives: typing.Mapping[str, typing.List[ModelInstanceInfo]],
    package_xmls: typing.List[str],
    builder: DiagramBuilder,
) -> None:
    models_from_directives_map = dict(
        [(info.model_name, info) for info in models_from_directives]
    )
    for model_instance_name, driver_config in driver_configs.items():
        _ApplyPrefinalizeDriverConfigSim(
            driver_config,
            model_instance_name,
            sim_plant,
            directives,
            models_from_directives_map,
            package_xmls,
            builder,
        )


def _ApplyCameraConfigSim(
    *, config: CameraConfig, builder: DiagramBuilder, lcm_buses: LcmBuses
) -> None:
    # Always opt-out of lcm in this workflow.
    this_config = copy(config)
    this_config.lcm_bus = "opt_out"
    ApplyCameraConfig(config=this_config, builder=builder, lcm_buses=lcm_buses)

    camera_sys = builder.GetSubsystemByName(f"rgbd_sensor_{config.name}")
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
    meshcat: Meshcat | None = None,
    *,
    package_xmls: typing.List[str] = [],
    hardware: bool = False,
    parser_preload_callback: typing.Callable[[Parser], None] | None = None,
    parser_prefinalize_callback: typing.Callable[[Parser], None] | None = None,
    prebuild_callback: typing.Callable[[DiagramBuilder], None] | None = None,
) -> Diagram:
    """Make a diagram encapsulating a simulation of (or the communications
    interface to/from) a physical robot, including sensors and controllers.

    Args:
        scenario: A Scenario structure, populated using the LoadScenario method.

        meshcat: If not None, then ApplyVisualizationConfig will called to add
            visualizers to the subdiagram using this meshcat instance.

        package_xmls: A list of package.xml file paths that will be passed to
            the parser, using Parser.AddPackageXml().

        hardware: If True, then the HardwareStationInterface diagram will be
            returned. Otherwise, the HardwareStation diagram will be returned.

        parser_preload_callback: A callback function that will be called after
            the Parser is created, but before any directives are processed. This
            can be used to add additional packages to the parser, or to add
            additional model directives.

        parser_prefinalize_callback: A callback function that will be called
            after the directives are processed, but before the plant is
            finalized. This can be used to add additional model directives.

        prebuild_callback: A callback function that will be called after the
            diagram builder is created, but before the diagram is built. This
            can be used to add additional systems to the diagram.

    Returns:
        If `hardware=False`, (the default) returns a HardwareStation diagram
        containing:
        - A MultibodyPlant with populated via the directives in `scenario`.
        - A SceneGraph
        - The default Drake visualizers
        - Any robot / sensors drivers specified in the YAML description.

        If `hardware=True`, returns a HardwareStationInterface diagram containing
        the network interfaces to communicate directly with the hardware drivers.
    """
    if hardware:
        return _MakeHardwareStationInterface(
            scenario=scenario, meshcat=meshcat, package_xmls=package_xmls
        )

    robot_builder = RobotDiagramBuilder(time_step=scenario.plant_config.time_step)
    builder = robot_builder.builder()
    sim_plant = robot_builder.plant()
    scene_graph = robot_builder.scene_graph()
    ApplyMultibodyPlantConfig(scenario.plant_config, sim_plant)

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

    # To support JointStiffnessControl, we need a pre-finalize version, too.
    _ApplyPrefinalizeDriverConfigsSim(
        driver_configs=scenario.model_drivers,
        sim_plant=sim_plant,
        directives=scenario.directives,
        models_from_directives=added_models,
        package_xmls=package_xmls,
        builder=builder,
    )

    sim_plant.Finalize()

    # For some Apply* functions in this workflow, we _never_ want LCM.
    scenario.lcm_buses["opt_out"] = DrakeLcmParams(lcm_url="memq://null")

    # Add LCM buses. (The simulator will handle polling the network for new
    # messages and dispatching them to the receivers, i.e., "pump" the bus.)
    lcm_buses = ApplyLcmBusConfig(lcm_buses=scenario.lcm_buses, builder=builder)

    # Add drivers.
    _ApplyDriverConfigsSim(
        driver_configs=scenario.model_drivers,
        sim_plant=sim_plant,
        scenario=scenario,
        package_xmls=package_xmls,
        builder=builder,
    )

    # Setup a virtual display if needed (for simulating cameras)
    if scenario.cameras and sys.platform == "linux" and os.getenv("DISPLAY") is None:
        from pyvirtualdisplay import Display

        virtual_display = Display(visible=0, size=(1400, 900))
        virtual_display.start()

    # Add scene cameras.
    for _, camera in scenario.cameras.items():
        _ApplyCameraConfigSim(config=camera, builder=builder, lcm_buses=lcm_buses)

    # Add visualization.
    if meshcat:
        ApplyVisualizationConfig(
            scenario.visualization, builder, meshcat=meshcat, lcm_buses=lcm_buses
        )

    # Export "cheat" ports.
    builder.ExportInput(
        sim_plant.get_applied_generalized_force_input_port(),
        "applied_generalized_force",
    )
    builder.ExportInput(
        sim_plant.get_applied_spatial_force_input_port(),
        "applied_spatial_force",
    )
    # Export any actuation (non-empty) input ports that are not already
    # connected (e.g. by a driver).
    for i in range(sim_plant.num_model_instances()):
        port = sim_plant.get_actuation_input_port(ModelInstanceIndex(i))
        if port.size() > 0 and not builder.IsConnectedOrExported(port):
            builder.ExportInput(port, port.get_name())
    # Export all MultibodyPlant output ports.
    for i in range(sim_plant.num_output_ports()):
        builder.ExportOutput(
            sim_plant.get_output_port(i),
            sim_plant.get_output_port(i).get_name(),
        )
    # Export the only SceneGraph output port.
    builder.ExportOutput(scene_graph.get_query_output_port(), "query_object")

    if prebuild_callback:
        prebuild_callback(builder)

    diagram = robot_builder.Build()
    diagram.set_name("station")
    return diagram


def NegatedPort(
    builder: DiagramBuilder, output_port: OutputPort, prefix: str = ""
) -> OutputPort:
    """Negates the output of the given port.

    Args:
        builder: The diagram builder to add the negation system to.
        output_port: The output port to negate.
        prefix: The prefix to use for the negation system.

    Returns:
        The output port of the negation system.
    """
    negater = builder.AddNamedSystem(
        f"sign_flip_{prefix}{output_port.get_name()}", Gain(-1, size=output_port.size())
    )
    builder.Connect(output_port, negater.get_input_port())
    return negater.get_output_port()


# TODO(russt): Use the c++ version pending https://github.com/RobotLocomotion/drake/issues/20055
def _ApplyDriverConfigInterface(
    driver_config,  # See Scenario.model_drivers for typing
    model_instance_name: str,
    lcm_buses: LcmBuses,
    builder: DiagramBuilder,
) -> None:
    if isinstance(driver_config, IiwaDriver):
        lcm = lcm_buses.Find("Driver for " + model_instance_name, driver_config.lcm_bus)

        # Publish IIWA command.
        # Note on publish period: IIWA driver won't respond faster than 1000Hz in
        # torque_only mode and 200Hz in other modes
        control_mode = ParseIiwaControlMode(driver_config.control_mode)
        publish_period = 0.005
        if control_mode == IiwaControlMode.kTorqueOnly:
            publish_period = 0.001
        iiwa_command_sender = builder.AddSystem(
            IiwaCommandSender(control_mode=control_mode)
        )
        iiwa_command_publisher = builder.AddSystem(
            LcmPublisherSystem.Make(
                channel="IIWA_COMMAND",
                lcm_type=lcmt_iiwa_command,
                lcm=lcm,
                publish_period=publish_period,
                use_cpp_serializer=True,
            )
        )
        iiwa_command_publisher.set_name(model_instance_name + ".command_publisher")
        if position_enabled(control_mode):
            builder.ExportInput(
                iiwa_command_sender.get_position_input_port(),
                model_instance_name + ".position",
            )
        if torque_enabled(control_mode):
            builder.ExportInput(
                iiwa_command_sender.get_torque_input_port(),
                model_instance_name + ".torque",
            )
        builder.Connect(
            iiwa_command_sender.get_output_port(),
            iiwa_command_publisher.get_input_port(),
        )
        # Receive IIWA status and populate the output ports.
        iiwa_status_receiver = builder.AddSystem(IiwaStatusReceiver())
        iiwa_status_receiver.set_name(model_instance_name + ".status_receiver")
        iiwa_status_subscriber = builder.AddSystem(
            LcmSubscriberSystem.Make(
                channel="IIWA_STATUS",
                lcm_type=lcmt_iiwa_status,
                lcm=lcm,
                use_cpp_serializer=True,
                wait_for_message_on_initialization_timeout=10,
            )
        )
        iiwa_status_subscriber.set_name(model_instance_name + ".status_subscriber")

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
        # These are *negative* w.r.t. the conventions outlined in
        # drake/manipulation/README.
        builder.ExportOutput(
            NegatedPort(
                builder=builder,
                output_port=iiwa_status_receiver.get_torque_commanded_output_port(),
                prefix=model_instance_name,
            ),
            model_instance_name + ".torque_commanded",
        )
        builder.ExportOutput(
            NegatedPort(
                builder=builder,
                output_port=iiwa_status_receiver.get_torque_measured_output_port(),
                prefix=model_instance_name,
            ),
            model_instance_name + ".torque_measured",
        )
        builder.ExportOutput(
            iiwa_status_receiver.get_torque_external_output_port(),
            model_instance_name + ".torque_external",
        )

        iiwa_state_mux: Multiplexer = builder.AddSystem(Multiplexer([7, 7]))
        builder.Connect(
            iiwa_status_receiver.get_position_measured_output_port(),
            iiwa_state_mux.get_input_port(0),
        )
        builder.Connect(
            iiwa_status_receiver.get_velocity_estimated_output_port(),
            iiwa_state_mux.get_input_port(1),
        )
        builder.ExportOutput(
            iiwa_state_mux.get_output_port(), model_instance_name + ".state_estimated"
        )

        builder.Connect(
            iiwa_status_subscriber.get_output_port(),
            iiwa_status_receiver.get_input_port(),
        )
    if isinstance(driver_config, SchunkWsgDriver):
        lcm = lcm_buses.Find("Driver for " + model_instance_name, driver_config.lcm_bus)

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
        wsg_command_publisher.set_name(model_instance_name + ".command_publisher")
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
        wsg_status_receiver.set_name(model_instance_name + ".status_receiver")
        wsg_status_subscriber = builder.AddSystem(
            LcmSubscriberSystem.Make(
                channel="SCHUNK_WSG_STATUS",
                lcm_type=lcmt_schunk_wsg_status,
                lcm=lcm,
                use_cpp_serializer=True,
                wait_for_message_on_initialization_timeout=10,
            )
        )
        wsg_status_subscriber.set_name(model_instance_name + ".status_subscriber")
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


def _ApplyDriverConfigsInterface(
    *,
    driver_configs,  # See Scenario.model_drivers for typing
    lcm_buses: LcmBuses,
    builder: DiagramBuilder,
) -> None:
    for model_instance_name, driver_config in driver_configs.items():
        _ApplyDriverConfigInterface(
            driver_config,
            model_instance_name,
            lcm_buses,
            builder,
        )


class ConcatenateVectors(LeafSystem):
    """
    Concatenates multiple input vectors into a single output vector.
    """

    def __init__(self, input_vector_sizes: typing.List[int]):
        """
        Args:
            input_vector_sizes (List[int]): A list of input vector sizes. This
                system creates an input port corresponding to each size.
        """
        super().__init__()

        self.output_vector_size = sum(input_vector_sizes)

        # One input port per input vector size.
        for i, input_size in enumerate(input_vector_sizes):
            self.DeclareVectorInputPort(
                f"input_vector_{i + 1}",
                input_size,
            )

        # Single output port.
        self.DeclareVectorOutputPort(
            "concatenated_vector",
            self.output_vector_size,
            self.Calc,
        )

    def Calc(self, context, output):
        output_vector = np.zeros(self.output_vector_size)

        start_index = 0
        for i in range(self.num_input_ports()):
            input_vector = self.get_input_port(i).Eval(context)
            end_index = start_index + len(input_vector)
            output_vector[start_index:end_index] = input_vector
            start_index = end_index
        output.SetFromVector(output_vector)


def _WireDriverStatusReceiversToToPose(
    model_instance_names: typing.List[str],
    builder: DiagramBuilder,
    plant: MultibodyPlant,
    to_pose: MultibodyPositionToGeometryPose,
):

    # Create a ConcatenateVector system to concatenate positions for each model
    # instance.
    input_vector_sizes: typing.List[int] = []
    for model_instance_name in model_instance_names:
        model_instance_index = plant.GetModelInstanceByName(model_instance_name)
        num_model_positions = plant.num_positions(model_instance_index)
        input_vector_sizes.append(num_model_positions)
    concatenator = builder.AddSystem(ConcatenateVectors(input_vector_sizes))

    # Wire the status receiver for each model driver to the concatenator.
    for i, model_instance_name in enumerate(model_instance_names):
        status_receiver = builder.GetSubsystemByName(
            f"{model_instance_name}.status_receiver"
        )
        if isinstance(status_receiver, IiwaStatusReceiver):
            builder.Connect(
                status_receiver.get_position_measured_output_port(),
                concatenator.get_input_port(i),
            )
        elif isinstance(status_receiver, SchunkWsgStatusReceiver):
            builder.Connect(
                status_receiver.get_state_output_port(),
                concatenator.get_input_port(i),
            )
        else:
            raise ValueError(f"Unknown status subscriber: {status_receiver}")

    # Wire the concatenator to the to_pose system.
    builder.Connect(
        concatenator.get_output_port(),
        to_pose.get_input_port(),
    )


def _ApplyCameraLcmIdInterface(
    camera_config,  # See Scenario.cameras for typing
    camera_id: str,
    lcm_buses: LcmBuses,
    builder: DiagramBuilder,
) -> None:
    lcm = lcm_buses.Find("Driver for " + camera_config.name, camera_config.lcm_bus)

    camera_data_receiver = builder.AddSystem(LcmImageArrayToImages())
    camera_data_receiver.set_name(camera_config.name + ".data_receiver")
    camera_data_subscriber = builder.AddSystem(
        LcmSubscriberSystem.Make(
            channel=camera_id,
            lcm_type=lcmt_image_array,
            lcm=lcm,
            use_cpp_serializer=True,
            wait_for_message_on_initialization_timeout=10,
        )
    )
    camera_data_subscriber.set_name(camera_config.name + ".data_subscriber")

    builder.ExportOutput(
        camera_data_receiver.color_image_output_port(),
        f"{camera_config.name}.rgb_image",
    )
    builder.ExportOutput(
        camera_data_receiver.depth_image_output_port(),
        f"{camera_config.name}.depth_image",
    )
    builder.ExportOutput(
        camera_data_receiver.label_image_output_port(),
        f"{camera_config.name}.label_image",
    )

    builder.Connect(
        camera_data_subscriber.get_output_port(),
        camera_data_receiver.image_array_t_input_port(),
    )


def _MakeHardwareStationInterface(
    scenario: Scenario,
    meshcat: Meshcat | None = None,
    *,
    package_xmls: typing.List[str] = [],
) -> Diagram:
    builder = DiagramBuilder()

    # Visualize plant if Meshcat is provided.
    if meshcat is not None:
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

        to_pose = builder.AddSystem(MultibodyPositionToGeometryPose(plant))
        builder.Connect(
            to_pose.get_output_port(),
            scene_graph.get_source_pose_port(plant.get_source_id()),
        )

        config = VisualizationConfig()
        config.publish_contacts = False
        config.publish_inertia = False
        ApplyVisualizationConfig(
            config,
            builder=builder,
            plant=plant,
            scene_graph=scene_graph,
            meshcat=meshcat,
        )

    # Add LCM buses. (The simulator will handle polling the network for new
    # messages and dispatching them to the receivers, i.e., "pump" the bus.)
    lcm_buses = ApplyLcmBusConfig(lcm_buses=scenario.lcm_buses, builder=builder)

    # Add drivers.
    _ApplyDriverConfigsInterface(
        driver_configs=scenario.model_drivers,
        lcm_buses=lcm_buses,
        builder=builder,
    )

    if meshcat is not None:
        _WireDriverStatusReceiversToToPose(
            model_instance_names=scenario.model_drivers.keys(),
            builder=builder,
            plant=plant,
            to_pose=to_pose,
        )

    # Add camera ids
    for camera_name, camera_id in scenario.camera_ids.items():
        _ApplyCameraLcmIdInterface(
            scenario.cameras[camera_name],
            camera_id,
            lcm_buses,
            builder,
        )

    diagram = builder.Build()
    diagram.set_name("HardwareStationInterface")
    return diagram


def AddPointClouds(
    *,
    scenario: Scenario,
    station: Diagram,
    builder: DiagramBuilder,
    poses_output_port: OutputPort | None = None,
    meshcat: Meshcat | None = None,
) -> typing.Mapping[str, DepthImageToPointCloud]:
    """
    Adds one DepthImageToPointCloud system to the `builder` for each camera in `scenario`, and connects it to the respective camera station output ports.

    Args:
        scenario: A Scenario structure, populated using the `LoadScenario` method.

        station: A HardwareStation system (e.g. from MakeHardwareStation) that has already been added to `builder`.

        builder: The DiagramBuilder containing `station` into which the new systems will be added.

        poses_output_port: (optional) HardwareStation will have a body_poses output port iff it was created with `hardware=False`. Alternatively, one could create a MultibodyPositionsToGeometryPoses system to consume the position measurements; this optional input can be used to support that workflow.

        meshcat: If not None, then a MeshcatPointCloudVisualizer will be added to the builder using this meshcat instance.

    Returns:
        A mapping from camera name to the DepthImageToPointCloud system.
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

        camera_pose = builder.AddSystem(ExtractPose(int(body.index()), X_BC))
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
