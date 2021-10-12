import os
import sys
import time

from IPython.display import display, HTML, Javascript
import numpy as np

from pydrake.common import set_log_level
from pydrake.geometry import Meshcat, Cylinder, Rgba
from pydrake.perception import PointCloud, Fields, BaseField

# imports for the pose sliders
from collections import namedtuple
from pydrake.common.value import AbstractValue
from pydrake.math import RollPitchYaw, RigidTransform, RotationMatrix
from pydrake.systems.framework import LeafSystem, PublishEvent

# imports for the joint sliders
from pydrake.multibody.tree import JointIndex
from pydrake.systems.framework import PublishEvent, VectorSystem

from manipulation import running_as_notebook


def StartMeshcat(open_window=False):
    """
    A wrapper around the Meshcat constructor that supports Deepnote and Google
    Colab via ngrok when necessary.
    """
    prev_log_level = set_log_level("warn")
    use_ngrok = False
    if ("DEEPNOTE_PROJECT_ID" in os.environ):
        # Deepnote exposes port 8080 (only).  If we need multiple meshcats,
        # then we fall back to ngrok.
        try:
            meshcat = Meshcat(8080)
        except RuntimeError:
            use_ngrok = True
        else:
            set_log_level(prev_log_level)
            web_url = f"https://{os.environ['DEEPNOTE_PROJECT_ID']}.deepnoteproject.com"  # noqa
            print(f"Meshcat is now available at {web_url}")
            if open_window:
                display(Javascript(f'window.open("{web_url}");'))
            return meshcat

    if 'google.colab' in sys.modules:
        use_ngrok = True

    meshcat = Meshcat()
    web_url = meshcat.web_url()
    if use_ngrok:
        from pyngrok import ngrok
        http_tunnel = ngrok.connect(meshcat.port(), bind_tls=False)
        web_url = http_tunnel.public_url

    set_log_level(prev_log_level)
    display(
        HTML('Meshcat is now available at '
             f'<a href="{web_url}">{web_url}</a>'))

    if open_window:
        if 'google.colab' in sys.modules:
            from google.colab import output
            output.eval_js(f'window.open("{web_url}");', ignore_result=True)
        else:
            display(Javascript(f'window.open("{web_url}");'))

    return meshcat


# Some GUI code that will be moved into Drake.


class MeshcatPoseSliders(LeafSystem):
    """
    Provides a set of ipywidget sliders (to be used in a Jupyter notebook) with
    one slider for each of roll, pitch, yaw, x, y, and z.  This can be used,
    for instance, as an interface to teleoperate the end-effector of a robot.

    .. pydrake_system::

        name: PoseSliders
        output_ports:
        - pose
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

    def __init__(self,
                 meshcat,
                 visible=Visible(),
                 min_range=MinRange(),
                 max_range=MaxRange(),
                 value=Value()):
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
        """
        LeafSystem.__init__(self)
        port = self.DeclareAbstractOutputPort(
            "pose", lambda: AbstractValue.Make(RigidTransform()),
            self.DoCalcOutput)

        # The widgets themselves have undeclared state.  For now, we accept it,
        # and simply disable caching on the output port.
        # TODO(russt): consider implementing the more elaborate methods seen
        # in, e.g., LcmMessageSubscriber.
        port.disable_caching_by_default()

        self._meshcat = meshcat
        self._visible = visible
        self._value = list(value)

        for i in range(6):
            if visible[i]:
                meshcat.AddSlider(min=min_range[i],
                                  max=max_range[i],
                                  value=value[i],
                                  step=0.01,
                                  name=value._fields[i])

    def __del__(self):
        for s in ['roll', 'pitch', 'yaw', 'x', 'y', 'z']:
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
                self._meshcat.SetSliderValue(self._visible._fields[i],
                                             self._value[i])

    def SetXyz(self, xyz):
        """
        Sets the current value of the sliders for x, y, and z.

        Args:
            xyz: A 3 element iterable object with x, y, z.
        """
        self._value[3:] = xyz
        for i in range(3, 6):
            if self._visible[i]:
                self._meshcat.SetSliderValue(self._visible._fields[i],
                                             self._value[i])

    def DoCalcOutput(self, context, output):
        """Constructs the output values from the widget elements."""
        for i in range(6):
            if self._visible[i]:
                self._value[i] = self._meshcat.GetSliderValue(
                    self._visible._fields[i])
        output.set_value(
            RigidTransform(
                RollPitchYaw(self._value[0], self._value[1], self._value[2]),
                self._value[3:]))


class WsgButton(LeafSystem):

    def __init__(self, meshcat):
        LeafSystem.__init__(self)
        port = self.DeclareVectorOutputPort("wsg_position", 1,
                                            self.DoCalcOutput)
        port.disable_caching_by_default()
        self._meshcat = meshcat
        self._button = "Open/Close Gripper"
        meshcat.AddButton(self._button)

    def __del__(self):
        self._meshcat.DeleteButton(self._button)

    def DoCalcOutput(self, context, output):
        position = 0.107  # open
        if (self._meshcat.GetButtonClicks(self._button) % 2) == 1:
            position = 0.002  # close
        output.SetAtIndex(0, position)


class MeshcatJointSlidersThatPublish():

    def __init__(self,
                 meshcat,
                 plant,
                 publishing_system,
                 root_context,
                 lower_limit=-10.,
                 upper_limit=10.,
                 resolution=0.01):
        """
        Creates an meshcat slider for each joint in the plant.  Unlike the
        JointSliders System, we do not expect this to be used in a Simulator.
        It simply updates the context and calls Publish directly from the
        slider callback.

        Args:
            meshcat:      A Meshcat instance.

            plant:        A MultibodyPlant. publishing_system: The System whose
                          Publish method will be called.  Can be the entire
                          Diagram, but can also be a subsystem.

            publishing_system:  The system to call publish on.  Probably a
                          MeshcatVisualizerCpp.

            root_context: A mutable root Context of the Diagram containing both
                          the ``plant`` and the ``publishing_system``; we will
                          extract the subcontext's using `GetMyContextFromRoot`.

            lower_limit:  A scalar or vector of length robot.num_positions().
                          The lower limit of the slider will be the maximum
                          value of this number and any limit specified in the
                          Joint.

            upper_limit:  A scalar or vector of length robot.num_positions().
                          The upper limit of the slider will be the minimum
                          value of this number and any limit specified in the
                          Joint.

            resolution:   A scalar or vector of length robot.num_positions()
                          that specifies the step argument of the FloatSlider.
        """

        def _broadcast(x, num):
            x = np.array(x)
            assert len(x.shape) <= 1
            return np.array(x) * np.ones(num)

        lower_limit = _broadcast(lower_limit, plant.num_positions())
        upper_limit = _broadcast(upper_limit, plant.num_positions())
        resolution = _broadcast(resolution, plant.num_positions())

        self._meshcat = meshcat
        self._plant = plant
        self._plant_context = plant.GetMyContextFromRoot(root_context)
        self._publishing_system = publishing_system
        self._publishing_context = publishing_system.GetMyContextFromRoot(
            root_context)

        self._sliders = []
        positions = plant.GetPositions(self._plant_context)
        slider_num = 0
        for i in range(plant.num_joints()):
            joint = plant.get_joint(JointIndex(i))
            low = joint.position_lower_limits()
            upp = joint.position_upper_limits()
            for j in range(joint.num_positions()):
                index = joint.position_start() + j
                description = joint.name()
                if joint.num_positions() > 1:
                    description += f"[{j}]"
                meshcat.AddSlider(value=positions[index],
                                  min=max(low[j], lower_limit[slider_num]),
                                  max=min(upp[j], upper_limit[slider_num]),
                                  step=resolution[slider_num],
                                  name=description)
                self._sliders.append(description)
                slider_num += 1

    def Publish(self):
        old_positions = self._plant.GetPositions(self._plant_context)
        positions = np.zeros((len(self._sliders), 1))
        for i, s in enumerate(self._sliders):
            positions[i] = self._meshcat.GetSliderValue(s)
        if not np.array_equal(positions, old_positions):
            self._plant.SetPositions(self._plant_context, positions)
            self._publishing_system.Publish(self._publishing_context)
            return True
        return False

    def Run(self, callback=None):
        if not running_as_notebook:
            return
        print("Press the 'Stop JointSliders' button in Meshcat to continue.")
        self._meshcat.AddButton("Stop JointSliders")
        while self._meshcat.GetButtonClicks("Stop JointSliders") < 1:
            if self.Publish() and callback:
                callback(self._plant_context)
            time.sleep(.1)

        self._meshcat.DeleteButton("Stop JointSliders")


def AddMeshcatTriad(meshcat,
                    path,
                    length=.25,
                    radius=0.01,
                    opacity=1.,
                    X_PT=RigidTransform()):
    meshcat.SetTransform(path, X_PT)
    # x-axis
    X_TG = RigidTransform(RotationMatrix.MakeYRotation(np.pi / 2),
                          [length / 2., 0, 0])
    meshcat.SetTransform(path + "/x-axis", X_TG)
    meshcat.SetObject(path + "/x-axis", Cylinder(radius, length),
                      Rgba(1, 0, 0, opacity))

    # y-axis
    X_TG = RigidTransform(RotationMatrix.MakeXRotation(np.pi / 2),
                          [0, length / 2., 0])
    meshcat.SetTransform(path + "/y-axis", X_TG)
    meshcat.SetObject(path + "/y-axis", Cylinder(radius, length),
                      Rgba(0, 1, 0, opacity))

    # z-axis
    X_TG = RigidTransform([0, 0, length / 2.])
    meshcat.SetTransform(path + "/z-axis", X_TG)
    meshcat.SetObject(path + "/z-axis", Cylinder(radius, length),
                      Rgba(0, 0, 1, opacity))


def draw_open3d_point_cloud(meshcat,
                            path,
                            pcd,
                            normals_scale=0.0,
                            point_size=0.001):
    pts = np.asarray(pcd.points)
    assert (pcd.has_colors())  # TODO(russt): handle this case better
    cloud = PointCloud(pts.shape[0], Fields(BaseField.kXYZs | BaseField.kRGBs))
    cloud.mutable_xyzs()[:] = pts.T
    cloud.mutable_rgbs()[:] = 255 * np.asarray(pcd.colors).T
    meshcat.SetObject(path, cloud, point_size=point_size)
    if pcd.has_normals() and normals_scale > 0.0:
        assert ('need to implement LineSegments in meshcat c++')
        normals = np.asarray(pcd.normals)
        vertices = np.hstack(
            (pts, pts + normals_scale * normals)).reshape(-1, 3).T
        meshcat["normals"].set_object(
            g.LineSegments(g.PointsGeometry(vertices),
                           g.MeshBasicMaterial(color=0x000000)))
