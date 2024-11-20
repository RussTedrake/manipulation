import time
import typing
from collections import namedtuple
from functools import partial

import numpy as np
from pydrake.common.value import AbstractValue
from pydrake.geometry import Cylinder, Meshcat, MeshcatVisualizer, Rgba, Sphere
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.multibody.plant import MultibodyPlant
from pydrake.solvers import (
    BoundingBoxConstraint,
    MathematicalProgram,
    MathematicalProgramResult,
)
from pydrake.systems.framework import Context, EventStatus, LeafSystem
from pydrake.trajectories import Trajectory

from manipulation import running_as_notebook


class MeshcatSliders(LeafSystem):
    """
    A system that outputs the values from meshcat sliders.

    An output port is created for each element in the list `slider_names`.
    Each element of `slider_names` must itself be an iterable collection
    (list, tuple, set, ...) of strings, with the names of sliders that have
    *already* been added to Meshcat via Meshcat.AddSlider().

    The same slider may be used in multiple ports.
    """

    def __init__(self, meshcat: Meshcat, slider_names: typing.List[str]):
        LeafSystem.__init__(self)

        self._meshcat = meshcat
        self._sliders = slider_names
        for i, slider_iterable in enumerate(self._sliders):
            port = self.DeclareVectorOutputPort(
                f"slider_group_{i}",
                len(slider_iterable),
                partial(self._DoCalcOutput, port_index=i),
            )
            port.disable_caching_by_default()

    def _DoCalcOutput(self, context, output, port_index):
        for i, slider in enumerate(self._sliders[port_index]):
            output[i] = self._meshcat.GetSliderValue(slider)


# This class is scheduled for removal.  Use the pydrake version of
# MeshcatPoseSliders instead.
class _MeshcatPoseSliders(LeafSystem):
    """
    Provides a set of ipywidget sliders (to be used in a Jupyter notebook) with
    one slider for each of roll, pitch, yaw, x, y, and z.  This can be used,
    for instance, as an interface to teleoperate the end-effector of a robot.

    The optional `pose` input port is used ONLY at initialization; it can be
    used to set the initial pose e.g. from the current pose of a MultibodyPlant
    frame.
    """

    # TODO(russt): Use namedtuple defaults parameter once we are Python >= 3.7.
    Visible = namedtuple("Visible", ("roll", "pitch", "yaw", "x", "y", "z"))
    Visible.__new__.__defaults__ = (True, True, True, True, True, True)
    MinRange = namedtuple("MinRange", ("roll", "pitch", "yaw", "x", "y", "z"))
    MinRange.__new__.__defaults__ = (-np.pi, -np.pi, -np.pi, -1.0, -1.0, -1.0)
    MaxRange = namedtuple("MaxRange", ("roll", "pitch", "yaw", "x", "y", "z"))
    MaxRange.__new__.__defaults__ = (np.pi, np.pi, np.pi, 1.0, 1.0, 1.0)
    Value = namedtuple("Value", ("roll", "pitch", "yaw", "x", "y", "z"))
    Value.__new__.__defaults__ = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    DecrementKey = namedtuple("DecrementKey", ("roll", "pitch", "yaw", "x", "y", "z"))
    DecrementKey.__new__.__defaults__ = (
        "KeyQ",
        "KeyW",
        "KeyA",
        "KeyJ",
        "KeyI",
        "KeyO",
    )
    IncrementKey = namedtuple("IncrementKey", ("roll", "pitch", "yaw", "x", "y", "z"))
    IncrementKey.__new__.__defaults__ = (
        "KeyE",
        "KeyS",
        "KeyD",
        "KeyL",
        "KeyK",
        "KeyU",
    )

    def __init__(
        self,
        meshcat,
        visible=Visible(),
        min_range=MinRange(),
        max_range=MaxRange(),
        value=Value(),
        decrement_keycode=DecrementKey(),
        increment_keycode=IncrementKey(),
        body_index=None,
    ):
        """
        Args:
            meshcat: A Meshcat instance.
            visible: An object with boolean elements for 'roll', 'pitch',
                'yaw', 'x', 'y', 'z'; the intention is for this to be the
                PoseSliders.Visible() namedtuple.  Defaults to all true.
            min_range, max_range, value: Objects with float values for 'roll',
                'pitch', 'yaw', 'x', 'y', 'z'; the intention is for the
                caller to use the PoseSliders.MinRange, MaxRange, and
                Value namedtuples.  See those tuples for default values.
            body_index: if the body_poses input port is connected, then this
                index determine which pose is used to set the initial
                slider positions during the Initialization event.
        """
        LeafSystem.__init__(self)
        port = self.DeclareAbstractOutputPort(
            "pose",
            lambda: AbstractValue.Make(RigidTransform()),
            self._DoCalcOutput,
        )

        self.DeclareAbstractInputPort(
            "body_poses", AbstractValue.Make([RigidTransform()])
        )
        self.DeclareInitializationDiscreteUpdateEvent(self._Initialize)

        # The widgets themselves have undeclared state.  For now, we accept it,
        # and simply disable caching on the output port.
        # TODO(russt): consider implementing the more elaborate methods seen
        # in, e.g., LcmMessageSubscriber.
        port.disable_caching_by_default()

        self._meshcat = meshcat
        self._visible = visible
        self._value = list(value)
        self._body_index = body_index

        print("Keyboard Controls:")
        for i in range(6):
            if visible[i]:
                meshcat.AddSlider(
                    min=min_range[i],
                    max=max_range[i],
                    value=value[i],
                    step=0.02,
                    name=value._fields[i],
                    decrement_keycode=decrement_keycode[i],
                    increment_keycode=increment_keycode[i],
                )
                print(
                    f"{value._fields[i]} : {decrement_keycode[i]} / {increment_keycode[i]}"  # noqa
                )

    def __del__(self):
        for s in ["roll", "pitch", "yaw", "x", "y", "z"]:
            if visible[s]:
                self._meshcat.DeleteSlider(s)

    def SetPose(self, pose):
        """
        Sets the current value of the sliders.

        Args:
            pose: Any viable argument for the RigidTransform
                constructor.
        """
        tf = RigidTransform(pose)
        self.SetRpy(RollPitchYaw(tf.rotation()))
        self.SetXyz(tf.translation())

    def SetRpy(self, rpy):
        """
        Sets the current value of the sliders for roll, pitch, and yaw.

        Args:
            rpy: An instance of drake.math.RollPitchYaw
        """
        self._value[0] = rpy.roll_angle()
        self._value[1] = rpy.pitch_angle()
        self._value[2] = rpy.yaw_angle()
        for i in range(3):
            if self._visible[i]:
                self._meshcat.SetSliderValue(self._visible._fields[i], self._value[i])

    def SetXyz(self, xyz):
        """
        Sets the current value of the sliders for x, y, and z.

        Args:
            xyz: A 3 element iterable object with x, y, z.
        """
        self._value[3:] = xyz
        for i in range(3, 6):
            if self._visible[i]:
                self._meshcat.SetSliderValue(self._visible._fields[i], self._value[i])

    def _update_values(self):
        changed = False
        for i in range(6):
            if self._visible[i]:
                old_value = self._value[i]
                self._value[i] = self._meshcat.GetSliderValue(self._visible._fields[i])
                changed = changed or self._value[i] != old_value
        return changed

    def _get_transform(self):
        return RigidTransform(
            RollPitchYaw(self._value[0], self._value[1], self._value[2]),
            self._value[3:],
        )

    def _DoCalcOutput(self, context, output):
        """Constructs the output values from the sliders."""
        self._update_values()
        output.set_value(self._get_transform())

    def _Initialize(self, context, discrete_state):
        if self.get_input_port().HasValue(context):
            if self._body_index is None:
                raise RuntimeError(
                    "If the `body_poses` input port is connected, then you "
                    "must also pass a `body_index` to the constructor."
                )
            self.SetPose(self.get_input_port().Eval(context)[self._body_index])
            return EventStatus.Succeeded()
        return EventStatus.DidNothing()

    def Run(self, publishing_system, root_context, callback):
        # Calls callback(root_context, pose), then
        # publishing_system.ForcedPublish() each time the sliders change value.
        if not running_as_notebook:
            return

        publishing_context = publishing_system.GetMyContextFromRoot(root_context)

        print("Press the 'Stop PoseSliders' button in Meshcat to continue.")
        self._meshcat.AddButton("Stop PoseSliders", "Escape")
        while self._meshcat.GetButtonClicks("Stop PoseSliders") < 1:
            if self._update_values():
                callback(root_context, self._get_transform())
                publishing_system.ForcedPublish(publishing_context)
            time.sleep(0.1)

        self._meshcat.DeleteButton("Stop PoseSliders")


class WsgButton(LeafSystem):
    """Adds a button named `Open/Close Gripper` to the meshcat GUI, and registers the Space key to press it. Pressing this button will toggle the value of the output port from a wsg position command corresponding to an open position or a closed position.

    Args:
        meshcat: The meshcat instance in which to register the button.
    """

    def __init__(self, meshcat: Meshcat):
        LeafSystem.__init__(self)
        port = self.DeclareVectorOutputPort("wsg_position", 1, self._DoCalcOutput)
        port.disable_caching_by_default()
        self._meshcat = meshcat
        self._button = "Open/Close Gripper"
        meshcat.AddButton(self._button, "Space")
        print("Press Space to open/close the gripper")

    def __del__(self):
        self._meshcat.DeleteButton(self._button)

    def _DoCalcOutput(self, context, output):
        position = 0.107  # open
        if (self._meshcat.GetButtonClicks(self._button) % 2) == 1:
            position = 0.002  # close
        output.SetAtIndex(0, position)


class StopButton(LeafSystem):
    """Adds a button named `Stop Simulation` to the meshcat GUI, and registers
    the `Escape` key to press it. Pressing this button will terminate the
    simulation.

    Args:
        meshcat: The meshcat instance in which to register the button.
        check_interval: The period at which to check for button presses.
    """

    def __init__(self, meshcat: Meshcat, check_interval: float = 0.1):
        LeafSystem.__init__(self)
        self._meshcat = meshcat
        self._button = "Stop Simulation"

        self.DeclareDiscreteState([0])  # button click count
        self.DeclareInitializationDiscreteUpdateEvent(self._Initialize)
        self.DeclarePeriodicDiscreteUpdateEvent(check_interval, 0, self._CheckButton)

        # Create the button now (rather than at initialization) so that the
        # CheckButton method will work even if Initialize has never been
        # called.
        meshcat.AddButton(self._button, "Escape")

    def __del__(self):
        # TODO(russt): Provide a nicer way to check if the button is currently
        # registered.
        try:
            self._meshcat.DeleteButton(self._button)
        except:
            pass

    def _Initialize(self, context, discrete_state):
        print("Press Escape to stop the simulation")
        discrete_state.set_value([self._meshcat.GetButtonClicks(self._button)])

    def _CheckButton(self, context, discrete_state):
        clicks_at_initialization = context.get_discrete_state().value()[0]
        if self._meshcat.GetButtonClicks(self._button) > clicks_at_initialization:
            self._meshcat.DeleteButton(self._button)
            return EventStatus.ReachedTermination(self, "Termination requested by user")
        return EventStatus.DidNothing()


def AddMeshcatTriad(
    meshcat: Meshcat,
    path: str,
    length: float = 0.25,
    radius: float = 0.01,
    opacity: float = 1.0,
    X_PT: RigidTransform = RigidTransform(),
):
    """Adds an X-Y-Z triad to the meshcat scene.

    Args:
        meshcat: A Meshcat instance.
        path: The Meshcat path on which to attach the triad. Using relative paths will attach the triad to the path's coordinate system.
        length: The length of the axes in meters.
        radius: The radius of the axes in meters.
        opacity: The opacity of the axes in [0, 1].
        X_PT: The pose of the triad relative to the path.
    """
    meshcat.SetTransform(path, X_PT)
    # x-axis
    X_TG = RigidTransform(RotationMatrix.MakeYRotation(np.pi / 2), [length / 2.0, 0, 0])
    meshcat.SetTransform(path + "/x-axis", X_TG)
    meshcat.SetObject(
        path + "/x-axis", Cylinder(radius, length), Rgba(1, 0, 0, opacity)
    )

    # y-axis
    X_TG = RigidTransform(RotationMatrix.MakeXRotation(np.pi / 2), [0, length / 2.0, 0])
    meshcat.SetTransform(path + "/y-axis", X_TG)
    meshcat.SetObject(
        path + "/y-axis", Cylinder(radius, length), Rgba(0, 1, 0, opacity)
    )

    # z-axis
    X_TG = RigidTransform([0, 0, length / 2.0])
    meshcat.SetTransform(path + "/z-axis", X_TG)
    meshcat.SetObject(
        path + "/z-axis", Cylinder(radius, length), Rgba(0, 0, 1, opacity)
    )


def plot_mathematical_program(
    meshcat: Meshcat,
    path: str,
    prog: MathematicalProgram,
    X: np.ndarray,
    Y: np.ndarray,
    result: MathematicalProgramResult | None = None,
    point_size: float = 0.05,
):
    """Visualize a MathematicalProgram in Meshcat.

    Args:
        meshcat: A Meshcat instance.
        path: The Meshcat path to the visualization.
        prog: A MathematicalProgram instance.
        X: A 2D array of x values.
        Y: A 2D array of y values.
        result: A MathematicalProgramResult instance; if provided then the
            solution is visualized as a point.
        point_size: The size of the solution point.
    """
    assert prog.num_vars() == 2
    assert X.size == Y.size

    X.size
    values = np.vstack((X.reshape(-1), Y.reshape(-1)))
    costs = prog.GetAllCosts()

    # Vectorized multiply for the quadratic form.
    # Z = (D*np.matmul(Q,D)).sum(0).reshape(nx, ny)

    if costs:
        Z = prog.EvalBindingVectorized(costs[0], values)
        for b in costs[1:]:
            Z = Z + prog.EvalBindingVectorized(b, values)

    cv = f"{path}/constraints"
    for binding in prog.GetAllConstraints():
        if isinstance(binding.evaluator(), BoundingBoxConstraint):
            c = binding.evaluator()
            var_indices = [
                int(prog.decision_variable_index()[v.get_id()])
                for v in binding.variables()
            ]
            satisfied = np.array(
                c.CheckSatisfiedVectorized(values[var_indices, :], 0.001)
            ).reshape(1, -1)
            if costs:
                Z[~satisfied] = np.nan

            v = f"{cv}/{type(c).__name__}"
            Zc = np.zeros(Z.shape)
            Zc[satisfied] = np.nan
            meshcat.PlotSurface(
                v,
                X,
                Y,
                Zc.reshape(X.shape),
                rgba=Rgba(1.0, 0.2, 0.2, 1.0),
                wireframe=True,
            )
        else:
            Zc = prog.EvalBindingVectorized(binding, values)
            evaluator = binding.evaluator()
            low = evaluator.lower_bound()
            up = evaluator.upper_bound()
            cvb = f"{cv}/{type(evaluator).__name__}"
            for index in range(Zc.shape[0]):
                # TODO(russt): Plot infeasible points in a different color.
                infeasible = np.logical_or(
                    Zc[index, :] < low[index], Zc[index, :] > up[index]
                )
                meshcat.PlotSurface(
                    f"{cvb}/{index}",
                    X,
                    Y,
                    Zc[index, :].reshape(X.shape),
                    rgba=Rgba(1.0, 0.3, 1.0, 1.0),
                    wireframe=True,
                )

    if costs:
        meshcat.PlotSurface(
            f"{path}/objective",
            X,
            Y,
            Z.reshape(X.shape),
            rgba=Rgba(0.3, 1.0, 0.3, 1.0),
            wireframe=True,
        )

    if result:
        v = f"{path}/solution"
        meshcat.SetObject(v, Sphere(point_size), Rgba(0.3, 1.0, 0.3, 1.0))
        x_solution = result.get_x_val()
        meshcat.SetTransform(
            v,
            RigidTransform([x_solution[0], x_solution[1], result.get_optimal_cost()]),
        )


def PublishPositionTrajectory(
    trajectory: Trajectory,
    root_context: Context,
    plant: MultibodyPlant,
    visualizer: MeshcatVisualizer,
    time_step: float = 1.0 / 33.0,
):
    """
    Publishes an animation to Meshcat of a MultibodyPlant using a trajectory of the plant positions.

    Args:
        trajectory: A Trajectory instance.
        root_context: The root context of the diagram containing plant.
        plant: A MultibodyPlant instance.
        visualizer: A MeshcatVisualizer instance.
        time_step: The time step between published frames.
    """
    plant_context = plant.GetMyContextFromRoot(root_context)
    visualizer_context = visualizer.GetMyContextFromRoot(root_context)

    visualizer.StartRecording(False)

    for t in np.append(
        np.arange(trajectory.start_time(), trajectory.end_time(), time_step),
        trajectory.end_time(),
    ):
        root_context.SetTime(t)
        plant.SetPositions(plant_context, trajectory.value(t))
        visualizer.ForcedPublish(visualizer_context)

    visualizer.StopRecording()
    visualizer.PublishRecording()
