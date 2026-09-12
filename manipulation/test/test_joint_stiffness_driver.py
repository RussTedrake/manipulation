"""Behavioral regressions for JointStiffnessDriver gravity compensation (#309).

These tests deliberately require physical gravity and actuator-limited compensation;
they must not pass merely because gravity has been disabled on the driven model.
All models are local, analytical fixtures with no contact or remote downloads.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from pydrake.all import Simulator

from manipulation.station import LoadScenario, MakeHardwareStation

_GRAVITY = 9.81
_DT = 0.001


def _link(name, mass=1, com="0 0 0"):
    return f"""<link name="{name}"><inertial>
      <origin xyz="{com}"/><mass value="{mass}"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial></link>"""


def _joint(name, kind="prismatic", axis="0 0 1", effort=100):
    return f"""<joint name="{name}" type="{kind}">
      <parent link="base"/><child link="{name}"/><axis xyz="{axis}"/>
      <limit lower="-10" upper="10" effort="{effort}" velocity="100"/>
    </joint>"""


def _transmission(name):
    return f"""<transmission name="{name}_transmission">
      <type>transmission_interface/SimpleTransmission</type>
      <joint name="{name}"><hardwareInterface>EffortJointInterface</hardwareInterface></joint>
      <actuator name="{name}_motor"><mechanicalReduction>1</mechanicalReduction></actuator>
    </transmission>"""


def _station(
    tmp_path,
    *,
    effort=100,
    pendulum=False,
    hand=False,
    name_hand=True,
    unrelated=False,
    multiple=False,
):
    model = '<robot name="robot"><link name="base"/>'
    model += _link("joint", com="1 0 0" if pendulum else "0 0 0")
    model += _joint(
        "joint",
        kind="revolute" if pendulum else "prismatic",
        axis="0 1 0" if pendulum else "0 0 1",
        effort=effort,
    )
    if multiple:
        # Different loads, reverse actuator order, and an unactuated coordinate
        # catch accidental use of generalized forces as actuator-ordered input.
        model += _link("second", mass=2) + _joint("second")
        model += _link("passive") + _joint("passive", axis="1 0 0")
        model += _transmission("second")
    model += _transmission("joint") + "</robot>"
    path = tmp_path / "robot.urdf"
    path.write_text(model)
    directives = f"""
- add_model:
    name: robot
    file: {path.as_uri()}
- add_weld:
    parent: world
    child: robot::base
"""
    if hand:
        path = tmp_path / "hand.urdf"
        path.write_text('<robot name="hand">' + _link("body", mass=2) + "</robot>")
        directives += f"""
- add_model:
    name: hand
    file: {path.as_uri()}
- add_weld:
    parent: robot::joint
    child: hand::body
"""
    if unrelated:
        path = tmp_path / "other.urdf"
        path.write_text(
            '<robot name="other"><link name="base"/>'
            + _link("joint", mass=2)
            + _joint("joint")
            + _transmission("joint")
            + "</robot>"
        )
        directives = (
            f"""
- add_model:
    name: other
    file: {path.as_uri()}
- add_weld:
    parent: world
    child: other::base
"""
            + directives
        )
    scenario = LoadScenario(
        data=f"""
plant_config:
    time_step: {_DT}
    discrete_contact_approximation: sap
directives:
{directives}
model_drivers:
    robot: !JointStiffnessDriver
        hand_model_name: {"hand" if hand and name_hand else '""'}
        gains:
            joint_motor: {{kp: 100, kd: 20}}
{"            second_motor: {kp: 100, kd: 20}" if multiple else ""}
{"    other: !ZeroForceDriver {}" if unrelated else ""}
"""
    )
    station = MakeHardwareStation(scenario)
    simulator = Simulator(station)
    context = simulator.get_mutable_context()
    plant = station.GetSubsystemByName("plant")
    robot = plant.GetModelInstanceByName("robot")
    plant_context = plant.GetMyMutableContextFromRoot(context)
    station.GetInputPort("robot.desired_state").FixValue(
        context, np.zeros(2 * plant.num_actuators(robot))
    )
    return station, simulator, plant, robot, plant_context


@pytest.mark.parametrize("hand", [False, True])
def test_physical_gravity_remains_enabled(tmp_path, hand):
    _, _, plant, robot, _ = _station(tmp_path, hand=hand)
    assert plant.is_gravity_enabled(robot)
    if hand:
        assert plant.is_gravity_enabled(plant.GetModelInstanceByName("hand"))


@pytest.mark.parametrize(
    "hand, name_hand", [(False, False), (True, True), (True, False)]
)
def test_holds_load_without_feedforward(tmp_path, hand, name_hand):
    _, simulator, plant, robot, pc = _station(tmp_path, hand=hand, name_hand=name_hand)
    # Leaving tau_feedforward unconnected is an existing supported use case.
    simulator.AdvanceTo(0.1)
    assert_allclose(plant.GetPositionsAndVelocities(pc, robot), [0, 0], atol=1e-10)
    assert_allclose(
        plant.get_net_actuation_output_port(robot).Eval(pc),
        [(3 if hand else 1) * _GRAVITY],
        atol=1e-10,
    )


@pytest.mark.parametrize("q, velocity", [(0, 0), (0.6, 0), (0.6, 0.4)])
def test_compensation_tracks_configuration(tmp_path, q, velocity):
    station, simulator, plant, robot, pc = _station(tmp_path, pendulum=True)
    station.GetInputPort("robot.tau_feedforward").FixValue(
        simulator.get_mutable_context(), [0]
    )
    simulator.Initialize()
    # Change the state after initialization to catch stale compensation.
    plant.get_actuation_input_port(robot).Eval(pc)
    plant.SetPositions(pc, robot, [q])
    plant.SetVelocities(pc, robot, [velocity])
    # A unit mass at x=1 on a y-axis hinge needs -mg*cos(q).
    # Gravity compensation must not introduce velocity-dependent forces.
    assert_allclose(
        plant.get_actuation_input_port(robot).Eval(pc),
        [-_GRAVITY * np.cos(q)],
        atol=1e-10,
    )


@pytest.mark.parametrize("feedforward", [-2.0, 0.0, 3.0])
def test_feedforward_adds_to_compensation(tmp_path, feedforward):
    station, simulator, plant, robot, pc = _station(tmp_path)
    station.GetInputPort("robot.tau_feedforward").FixValue(
        simulator.get_mutable_context(), [feedforward]
    )
    assert_allclose(
        plant.get_actuation_input_port(robot).Eval(pc),
        [_GRAVITY + feedforward],
        atol=1e-10,
    )


@pytest.mark.parametrize(
    "effort, feedforward, desired_position, expected_effort",
    [
        (5, 0, 0, 5),  # Gravity alone exceeds available effort.
        (12, 4, 0, 12),  # Gravity + feedforward exceed the limit.
        (12, 0, 0.05, 12),  # Gravity + PD exceed the limit.
        (5, -100, 0, -5),  # Lower saturation limit also includes compensation.
    ],
)
def test_total_effort_is_limited(
    tmp_path, effort, feedforward, desired_position, expected_effort
):
    station, simulator, plant, robot, pc = _station(tmp_path, effort=effort)
    context = simulator.get_mutable_context()
    station.GetInputPort("robot.tau_feedforward").FixValue(context, [feedforward])
    station.GetInputPort("robot.desired_state").FixValue(context, [desired_position, 0])
    simulator.AdvanceTo(_DT)
    assert_allclose(
        plant.get_net_actuation_output_port(robot).Eval(pc),
        [expected_effort],
        atol=1e-10,
    )
    # Unit mass: the actual motion must include gravity even at saturation.
    assert_allclose(
        plant.GetVelocities(pc, robot), [_DT * (expected_effort - _GRAVITY)], atol=1e-10
    )


def test_unrelated_model_still_falls(tmp_path):
    _, simulator, plant, robot, pc = _station(tmp_path, unrelated=True)
    simulator.AdvanceTo(_DT)
    other = plant.GetModelInstanceByName("other")
    assert plant.is_gravity_enabled(other)
    assert_allclose(plant.GetVelocities(pc, other), [-_GRAVITY * _DT], atol=1e-10)
    assert_allclose(plant.GetVelocities(pc, robot), [0], atol=1e-10)
    assert_allclose(
        plant.get_net_actuation_output_port(robot).Eval(pc), [_GRAVITY], atol=1e-10
    )


def test_actuator_mapping_with_passive_coordinate(tmp_path):
    station, simulator, plant, robot, pc = _station(tmp_path, multiple=True)
    station.GetInputPort("robot.tau_feedforward").FixValue(
        simulator.get_mutable_context(), [0, 0]
    )
    assert plant.num_velocities(robot) == 3
    assert plant.num_actuators(robot) == 2
    actuators = [
        plant.get_joint_actuator(i) for i in plant.GetJointActuatorIndices(robot)
    ]
    assert [a.joint().name() for a in actuators] == ["second", "joint"]
    assert plant.GetVelocityNames(robot) == ["joint_v", "second_v", "passive_v"]
    loads = {"joint": _GRAVITY, "second": 2 * _GRAVITY}
    expected = [loads[a.joint().name()] for a in actuators]
    simulator.AdvanceTo(0.1)
    assert_allclose(
        plant.get_actuation_input_port(robot).Eval(pc), expected, atol=1e-10
    )
    assert_allclose(
        plant.get_net_actuation_output_port(robot).Eval(pc), expected, atol=1e-10
    )
    assert_allclose(plant.GetPositionsAndVelocities(pc, robot), np.zeros(6), atol=1e-10)
