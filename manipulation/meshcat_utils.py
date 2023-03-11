import time
from collections import namedtuple
from functools import partial

import numpy as np
from pydrake.common.value import AbstractValue
from pydrake.geometry import (
    Cylinder,
    Rgba,
    Sphere,
)
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.solvers import BoundingBoxConstraint
from pydrake.systems.framework import (
    EventStatus,
    LeafSystem,
)

from manipulation import running_as_notebook

# Some GUI code that will be moved into Drake.


class MeshcatSliders(LeafSystem):
    """
    A system that outputs the ``value``s from meshcat sliders.

    .. pydrake_system::

      name: MeshcatSliderSystem
      output_ports:
      - slider_group_0
      - ...
      - slider_group_{N-1}
    """

    def __init__(self, meshcat, slider_names):
        """
        An output port is created for each element in the list `slider_names`.
        Each element of `slider_names` must itself be an iterable collection
        (list, tuple, set, ...) of strings, with the names of sliders that have
        *already* been added to Meshcat via Meshcat.AddSlider().

        The same slider may be used in multiple ports.
        """
        LeafSystem.__init__(self)

        self._meshcat = meshcat
        self._sliders = slider_names
        for i, slider_iterable in enumerate(self._sliders):
            port = self.DeclareVectorOutputPort(
                f"slider_group_{i}",
                len(slider_iterable),
                partial(self.DoCalcOutput, port_index=i),
            )
            port.disable_caching_by_default()

    def DoCalcOutput(self, context, output, port_index):
        for i, slider in enumerate(self._sliders[port_index]):
            output[i] = self._meshcat.GetSliderValue(slider)


class MeshcatPoseSliders(LeafSystem):
    """
    Provides a set of ipywidget sliders (to be used in a Jupyter notebook) with
    one slider for each of roll, pitch, yaw, x, y, and z.  This can be used,
    for instance, as an interface to teleoperate the end-effector of a robot.

    .. pydrake_system::

        name: PoseSliders
        input_ports:
        - pose (optional)
        output_ports:
        - pose

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
    DecrementKey = namedtuple(
        "DecrementKey", ("roll", "pitch", "yaw", "x", "y", "z")
    )
    DecrementKey.__new__.__defaults__ = (
        "KeyQ",
        "KeyW",
        "KeyA",
        "KeyJ",
        "KeyI",
        "KeyO",
    )
    IncrementKey = namedtuple(
        "IncrementKey", ("roll", "pitch", "yaw", "x", "y", "z")
    )
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
            self.DoCalcOutput,
        )

        self.DeclareAbstractInputPort(
            "body_poses", AbstractValue.Make([RigidTransform()])
        )
        self.DeclareInitializationDiscreteUpdateEvent(self.Initialize)

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
                self._meshcat.SetSliderValue(
                    self._visible._fields[i], self._value[i]
                )

    def SetXyz(self, xyz):
        """
        Sets the current value of the sliders for x, y, and z.

        Args:
            xyz: A 3 element iterable object with x, y, z.
        """
        self._value[3:] = xyz
        for i in range(3, 6):
            if self._visible[i]:
                self._meshcat.SetSliderValue(
                    self._visible._fields[i], self._value[i]
                )

    def _update_values(self):
        changed = False
        for i in range(6):
            if self._visible[i]:
                old_value = self._value[i]
                self._value[i] = self._meshcat.GetSliderValue(
                    self._visible._fields[i]
                )
                changed = changed or self._value[i] != old_value
        return changed

    def _get_transform(self):
        return RigidTransform(
            RollPitchYaw(self._value[0], self._value[1], self._value[2]),
            self._value[3:],
        )

    def DoCalcOutput(self, context, output):
        """Constructs the output values from the sliders."""
        self._update_values()
        output.set_value(self._get_transform())

    def Initialize(self, context, discrete_state):
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

        publishing_context = publishing_system.GetMyContextFromRoot(
            root_context
        )

        print("Press the 'Stop PoseSliders' button in Meshcat to continue.")
        self._meshcat.AddButton("Stop PoseSliders", "Escape")
        while self._meshcat.GetButtonClicks("Stop PoseSliders") < 1:
            if self._update_values():
                callback(root_context, self._get_transform())
                publishing_system.ForcedPublish(publishing_context)
            time.sleep(0.1)

        self._meshcat.DeleteButton("Stop PoseSliders")


class WsgButton(LeafSystem):
    def __init__(self, meshcat):
        LeafSystem.__init__(self)
        port = self.DeclareVectorOutputPort(
            "wsg_position", 1, self.DoCalcOutput
        )
        port.disable_caching_by_default()
        self._meshcat = meshcat
        self._button = "Open/Close Gripper"
        meshcat.AddButton(self._button, "Space")
        print("Press Space to open/close the gripper")

    def __del__(self):
        self._meshcat.DeleteButton(self._button)

    def DoCalcOutput(self, context, output):
        position = 0.107  # open
        if (self._meshcat.GetButtonClicks(self._button) % 2) == 1:
            position = 0.002  # close
        output.SetAtIndex(0, position)


def AddMeshcatTriad(
    meshcat, path, length=0.25, radius=0.01, opacity=1.0, X_PT=RigidTransform()
):
    meshcat.SetTransform(path, X_PT)
    # x-axis
    X_TG = RigidTransform(
        RotationMatrix.MakeYRotation(np.pi / 2), [length / 2.0, 0, 0]
    )
    meshcat.SetTransform(path + "/x-axis", X_TG)
    meshcat.SetObject(
        path + "/x-axis", Cylinder(radius, length), Rgba(1, 0, 0, opacity)
    )

    # y-axis
    X_TG = RigidTransform(
        RotationMatrix.MakeXRotation(np.pi / 2), [0, length / 2.0, 0]
    )
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


def plot_surface(
    meshcat,
    path,
    X,
    Y,
    Z,
    rgba=Rgba(0.87, 0.6, 0.6, 1.0),
    wireframe=False,
    wireframe_line_width=1.0,
):
    (rows, cols) = Z.shape
    assert np.array_equal(X.shape, Y.shape)
    assert np.array_equal(X.shape, Z.shape)

    vertices = np.empty((rows * cols, 3), dtype=np.float32)
    vertices[:, 0] = X.reshape((-1))
    vertices[:, 1] = Y.reshape((-1))
    vertices[:, 2] = Z.reshape((-1))

    # Vectorized faces code from https://stackoverflow.com/questions/44934631/making-grid-triangular-mesh-quickly-with-numpy  # noqa
    faces = np.empty((rows - 1, cols - 1, 2, 3), dtype=np.uint32)
    r = np.arange(rows * cols).reshape(rows, cols)
    faces[:, :, 0, 0] = r[:-1, :-1]
    faces[:, :, 1, 0] = r[:-1, 1:]
    faces[:, :, 0, 1] = r[:-1, 1:]
    faces[:, :, 1, 1] = r[1:, 1:]
    faces[:, :, :, 2] = r[1:, :-1, None]
    faces.shape = (-1, 3)

    # TODO(Russ): support per vertex / Colormap colors.
    meshcat.SetTriangleMesh(
        path, vertices.T, faces.T, rgba, wireframe, wireframe_line_width
    )


def plot_mathematical_program(
    meshcat, path, prog, X, Y, result=None, point_size=0.05
):
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
            plot_surface(
                meshcat,
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
                plot_surface(
                    meshcat,
                    f"{cvb}/{index}",
                    X,
                    Y,
                    Zc[index, :].reshape(X.shape),
                    rgba=Rgba(1.0, 0.3, 1.0, 1.0),
                    wireframe=True,
                )

    if costs:
        plot_surface(
            meshcat,
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
            RigidTransform(
                [x_solution[0], x_solution[1], result.get_optimal_cost()]
            ),
        )


def PublishPositionTrajectory(
    trajectory, root_context, plant, visualizer, time_step=1.0 / 33.0
):
    """
    Args:
        trajectory: A Trajectory instance.
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
