"""
This file contains a number of helper utilities to set up our various
experiments with less code.
"""
import numpy as np
import os
import sys
import warnings

from pydrake.all import (
    AbstractValue, Adder, AddMultibodyPlantSceneGraph, BallRpyJoint, BaseField,
    Box, CameraInfo, ClippingRange, CoulombFriction, Cylinder, Demultiplexer,
    DiagramBuilder, DepthRange, DepthImageToPointCloud, DepthRenderCamera,
    FindResourceOrThrow, GeometryInstance, InverseDynamicsController,
    LeafSystem, MakeMultibodyStateToWsgStateSystem,
    MakePhongIllustrationProperties, MakeRenderEngineVtk, ModelInstanceIndex,
    MultibodyPlant, Parser, PassThrough, PrismaticJoint, RenderCameraCore,
    RenderEngineVtkParams, RevoluteJoint, Rgba, RigidTransform, RollPitchYaw,
    RotationMatrix, RgbdSensor, SchunkWsgPositionController, SpatialInertia,
    Sphere, StateInterpolatorWithDiscreteDerivative, UnitInertia)
from manipulation.utils import FindResource

ycb = [
    "003_cracker_box.sdf", "004_sugar_box.sdf", "005_tomato_soup_can.sdf",
    "006_mustard_bottle.sdf", "009_gelatin_box.sdf", "010_potted_meat_can.sdf"
]


def AddIiwa(plant, collision_model="no_collision"):
    sdf_path = FindResourceOrThrow(
        "drake/manipulation/models/iiwa_description/iiwa7/"
        f"iiwa7_{collision_model}.sdf")

    parser = Parser(plant)
    iiwa = parser.AddModelFromFile(sdf_path)
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0"))

    # Set default positions:
    q0 = [0.0, 0.1, 0, -1.2, 0, 1.6, 0]
    index = 0
    for joint_index in plant.GetJointIndices(iiwa):
        joint = plant.get_mutable_joint(joint_index)
        if isinstance(joint, RevoluteJoint):
            joint.set_default_angle(q0[index])
            index += 1

    return iiwa


def AddPlanarIiwa(plant):
    urdf = FindResourceOrThrow(
        "drake/manipulation/models/iiwa_description/urdf/"
        "planar_iiwa14_spheres_dense_elbow_collision.urdf")

    parser = Parser(plant)
    iiwa = parser.AddModelFromFile(urdf)
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0"))

    # Set default positions:
    q0 = [0.1, -1.2, 1.6]
    index = 0
    for joint_index in plant.GetJointIndices(iiwa):
        joint = plant.get_mutable_joint(joint_index)
        if isinstance(joint, RevoluteJoint):
            joint.set_default_angle(q0[index])
            index += 1

    return iiwa


def AddTwoLinkIiwa(plant, q0=[0.1, -1.2]):
    urdf = FindResource("models/two_link_iiwa14.urdf")

    parser = Parser(plant)
    parser.package_map().Add(
        "iiwa_description",
        os.path.dirname(
            FindResourceOrThrow(
                "drake/manipulation/models/iiwa_description/package.xml")))
    iiwa = parser.AddModelFromFile(urdf)
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0"))

    # Set default positions:
    index = 0
    for joint_index in plant.GetJointIndices(iiwa):
        joint = plant.get_mutable_joint(joint_index)
        if isinstance(joint, RevoluteJoint):
            joint.set_default_angle(q0[index])
            index += 1

    return iiwa


# TODO: take argument for whether we want the welded fingers version or not
def AddWsg(plant, iiwa_model_instance, roll=np.pi / 2.0, welded=False):
    parser = Parser(plant)
    if welded:
        parser.package_map().Add(
            "wsg_50_description",
            os.path.dirname(
                FindResourceOrThrow(
                    "drake/manipulation/models/wsg_50_description/package.xml"))
        )
        gripper = parser.AddModelFromFile(
            FindResource("models/schunk_wsg_50_welded_fingers.sdf"), "gripper")
    else:
        gripper = parser.AddModelFromFile(
            FindResourceOrThrow(
                "drake/manipulation/models/"
                "wsg_50_description/sdf/schunk_wsg_50_with_tip.sdf"))

    X_7G = RigidTransform(RollPitchYaw(np.pi / 2.0, 0, roll), [0, 0, 0.114])
    plant.WeldFrames(plant.GetFrameByName("iiwa_link_7", iiwa_model_instance),
                     plant.GetFrameByName("body", gripper), X_7G)
    return gripper


def AddFloatingRpyJoint(plant, frame, instance):
    inertia = UnitInertia.SolidSphere(1.0)
    x_body = plant.AddRigidBody(
        "x", instance,
        SpatialInertia(mass=0, p_PScm_E=[0., 0., 0.], G_SP_E=inertia))
    plant.AddJoint(
        PrismaticJoint("x", plant.world_frame(), x_body.body_frame(),
                       [1, 0, 0]))
    y_body = plant.AddRigidBody(
        "y", instance,
        SpatialInertia(mass=0, p_PScm_E=[0., 0., 0.], G_SP_E=inertia))
    plant.AddJoint(
        PrismaticJoint("y", x_body.body_frame(), y_body.body_frame(),
                       [0, 1, 0]))
    z_body = plant.AddRigidBody(
        "z", instance,
        SpatialInertia(mass=0, p_PScm_E=[0., 0., 0.], G_SP_E=inertia))
    plant.AddJoint(
        PrismaticJoint("z", y_body.body_frame(), z_body.body_frame(),
                       [0, 0, 1]))
    plant.AddJoint(BallRpyJoint("ball", z_body.body_frame(), frame))


def AddShape(plant, shape, name, mass=1, mu=1, color=[.5, .5, .9, 1.0]):
    instance = plant.AddModelInstance(name)
    # TODO: Add a method to UnitInertia that accepts a geometry shape (unless
    # that dependency is somehow gross) and does this.
    if isinstance(shape, Box):
        inertia = UnitInertia.SolidBox(shape.width(), shape.depth(),
                                       shape.height())
    elif isinstance(shape, Cylinder):
        inertia = UnitInertia.SolidCylinder(shape.radius(), shape.length())
    elif isinstance(shape, Sphere):
        inertia = UnitInertia.SolidSphere(shape.radius())
    else:
        raise RunTimeError(
            f"need to write the unit inertia for shapes of type {shape}")
    body = plant.AddRigidBody(
        name, instance,
        SpatialInertia(mass=mass,
                       p_PScm_E=np.array([0., 0., 0.]),
                       G_SP_E=inertia))
    if plant.geometry_source_is_registered():
        if isinstance(shape, Box):
            plant.RegisterCollisionGeometry(
                body, RigidTransform(),
                Box(shape.width() - 0.001,
                    shape.depth() - 0.001,
                    shape.height() - 0.001), name, CoulombFriction(mu, mu))
            i = 0
            for x in [-shape.width() / 2.0, shape.width() / 2.0]:
                for y in [-shape.depth() / 2.0, shape.depth() / 2.0]:
                    for z in [-shape.height() / 2.0, shape.height() / 2.0]:
                        plant.RegisterCollisionGeometry(
                            body, RigidTransform([x, y, z]),
                            Sphere(radius=1e-7), f"contact_sphere{i}",
                            CoulombFriction(mu, mu))
                        i += 1
        else:
            plant.RegisterCollisionGeometry(body, RigidTransform(), shape, name,
                                            CoulombFriction(mu, mu))

        plant.RegisterVisualGeometry(body, RigidTransform(), shape, name, color)

    return instance


# Add the camera_box.sdf.
def AddCameraBox(plant, X_WC, name="camera0", parent_frame=None):
    # TODO(russt): could be smarter and increment the default camera name (by
    # checking with the plant).
    if not parent_frame:
        parent_frame = plant.world_frame()
    camera = Parser(plant).AddModelFromFile(
        FindResource("models/camera_box.sdf"), name)
    plant.WeldFrames(parent_frame, plant.GetFrameByName("base", camera), X_WC)


def AddCamera(builder, scene_graph, X_WC, depth_camera=None, renderer=None):
    warnings.warn("Please use AddRgbdSensor instead.",
                  warnings.DeprecationWarning)
    return AddRgbdSensor(builder, scene_graph, X_WC, depth_camera, renderer)


def AddRgbdSensor(builder,
                  scene_graph,
                  X_PC,
                  depth_camera=None,
                  renderer=None,
                  parent_frame_id=None):
    """
    Adds a RgbdSensor to to the scene_graph at (fixed) pose X_PC relative to
    the parent_frame.  If depth_camera is None, then a default camera info will
    be used.  If renderer is None, then we will assume the name 'my_renderer',
    and create a VTK renderer if a renderer of that name doesn't exist.  If
    parent_frame is None, then the world frame is used.
    """
    if sys.platform == "linux" and os.getenv("DISPLAY") is None:
        from pyvirtualdisplay import Display
        virtual_display = Display(visible=0, size=(1400, 900))
        virtual_display.start()

    if not renderer:
        renderer = "my_renderer"

    if not parent_frame_id:
        parent_frame_id = scene_graph.world_frame_id()

    if not scene_graph.HasRenderer(renderer):
        scene_graph.AddRenderer(renderer,
                                MakeRenderEngineVtk(RenderEngineVtkParams()))

    if not depth_camera:
        depth_camera = DepthRenderCamera(
            RenderCameraCore(
                renderer, CameraInfo(width=640, height=480, fov_y=np.pi / 4.0),
                ClippingRange(near=0.1, far=10.0), RigidTransform()),
            DepthRange(0.1, 10.0))

    rgbd = builder.AddSystem(
        RgbdSensor(parent_id=parent_frame_id,
                   X_PB=X_PC,
                   depth_camera=depth_camera,
                   show_window=False))

    builder.Connect(scene_graph.get_query_output_port(),
                    rgbd.query_object_input_port())

    return rgbd


def AddRgbdSensors(builder,
                   plant,
                   scene_graph,
                   also_add_point_clouds=True,
                   model_instance_prefix="camera",
                   depth_camera=None,
                   renderer=None):
    """
    Adds a RgbdSensor to every body in the plant with a name starting with
    body_prefix.  If depth_camera is None, then a default camera info will be
    used.  If renderer is None, then we will assume the name 'my_renderer', and
    create a VTK renderer if a renderer of that name doesn't exist.
    """
    if sys.platform == "linux" and os.getenv("DISPLAY") is None:
        from pyvirtualdisplay import Display
        virtual_display = Display(visible=0, size=(1400, 900))
        virtual_display.start()

    if not renderer:
        renderer = "my_renderer"

    if not scene_graph.HasRenderer(renderer):
        scene_graph.AddRenderer(renderer,
                                MakeRenderEngineVtk(RenderEngineVtkParams()))

    if not depth_camera:
        depth_camera = DepthRenderCamera(
            RenderCameraCore(
                renderer, CameraInfo(width=640, height=480, fov_y=np.pi / 4.0),
                ClippingRange(near=0.1, far=10.0), RigidTransform()),
            DepthRange(0.1, 10.0))

    for index in range(plant.num_model_instances()):
        model_instance_index = ModelInstanceIndex(index)
        model_name = plant.GetModelInstanceName(model_instance_index)

        if model_name.startswith(model_instance_prefix):
            body_index = plant.GetBodyIndices(model_instance_index)[0]
            rgbd = builder.AddSystem(
                RgbdSensor(parent_id=plant.GetBodyFrameIdOrThrow(body_index),
                           X_PB=RigidTransform(),
                           depth_camera=depth_camera,
                           show_window=False))
            rgbd.set_name(model_name)

            builder.Connect(scene_graph.get_query_output_port(),
                            rgbd.query_object_input_port())

            # Export the camera outputs
            builder.ExportOutput(rgbd.color_image_output_port(),
                                 f"{model_name}_rgb_image")
            builder.ExportOutput(rgbd.depth_image_32F_output_port(),
                                 f"{model_name}_depth_image")
            builder.ExportOutput(rgbd.label_image_output_port(),
                                 f"{model_name}_label_image")

            if also_add_point_clouds:
                # Add a system to convert the camera output into a point cloud
                to_point_cloud = builder.AddSystem(
                    DepthImageToPointCloud(camera_info=rgbd.depth_camera_info(),
                                           fields=BaseField.kXYZs
                                           | BaseField.kRGBs))
                builder.Connect(rgbd.depth_image_32F_output_port(),
                                to_point_cloud.depth_image_input_port())
                builder.Connect(rgbd.color_image_output_port(),
                                to_point_cloud.color_image_input_port())

                class ExtractBodyPose(LeafSystem):

                    def __init__(self, body_index):
                        LeafSystem.__init__(self)
                        self.body_index = body_index
                        self.DeclareAbstractInputPort(
                            "poses",
                            plant.get_body_poses_output_port().Allocate())
                        self.DeclareAbstractOutputPort(
                            "pose",
                            lambda: AbstractValue.Make(RigidTransform()),
                            self.CalcOutput)

                    def CalcOutput(self, context, output):
                        poses = self.EvalAbstractInput(context, 0).get_value()
                        pose = poses[int(self.body_index)]
                        output.get_mutable_value().set(pose.rotation(),
                                                       pose.translation())

                camera_pose = builder.AddSystem(ExtractBodyPose(body_index))
                builder.Connect(plant.get_body_poses_output_port(),
                                camera_pose.get_input_port())
                builder.Connect(camera_pose.get_output_port(),
                                to_point_cloud.GetInputPort("camera_pose"))

                # Export the point cloud output.
                builder.ExportOutput(to_point_cloud.point_cloud_output_port(),
                                     f"{model_name}_point_cloud")


def SetTransparency(scene_graph, alpha, source_id, geometry_ids=None):
    inspector = scene_graph.model_inspector()
    if not geometry_ids:
        geometry_ids = inspector.GetAllGeometryIds()

    for gid in geometry_ids:
        if not inspector.BelongsToSource(gid, source_id):
            continue
        props = inspector.GetIllustrationProperties(gid)
        if props is None or not props.HasProperty("phong", "diffuse"):
            continue
        c = props.GetProperty("phong", "diffuse")
        new_color = Rgba(c.r(), c.g(), c.b(), alpha)
        props.UpdateProperty("phong", "diffuse", new_color)


# TODO(russt): Use Rgba instead of vector color.
def SetColor(scene_graph, color, source_id, geometry_ids=None):
    inspector = scene_graph.model_inspector()
    if not geometry_ids:
        geometry_ids = inspector.GetAllGeometryIds()

    for gid in geometry_ids:
        if not inspector.BelongsToSource(gid, source_id):
            continue
        props = inspector.GetIllustrationProperties(gid)
        if props is None or not props.HasProperty("phong", "diffuse"):
            continue
        new_color = Rgba(color[0], color[1], color[2], color[3])
        props.UpdateProperty("phong", "diffuse", new_color)


def AddTriad(source_id,
             frame_id,
             scene_graph,
             length=.25,
             radius=0.01,
             opacity=1.,
             X_FT=RigidTransform(),
             name="frame"):
    """
    Adds illustration geometry representing the coordinate frame, with the
    x-axis drawn in red, the y-axis in green and the z-axis in blue. The axes
    point in +x, +y and +z directions, respectively.

    Args:
      source_id: The source registered with SceneGraph.
      frame_id: A geometry::frame_id registered with scene_graph.
      scene_graph: The SceneGraph with which we will register the geometry.
      length: the length of each axis in meters.
      radius: the radius of each axis in meters.
      opacity: the opacity of the coordinate axes, between 0 and 1.
      X_FT: a RigidTransform from the triad frame T to the frame_id frame F
      name: the added geometry will have names name + " x-axis", etc.
    """
    # x-axis
    X_TG = RigidTransform(RotationMatrix.MakeYRotation(np.pi / 2),
                          [length / 2., 0, 0])
    geom = GeometryInstance(X_FT.multiply(X_TG), Cylinder(radius, length),
                            name + " x-axis")
    geom.set_illustration_properties(
        MakePhongIllustrationProperties([1, 0, 0, opacity]))
    scene_graph.RegisterGeometry(source_id, frame_id, geom)

    # y-axis
    X_TG = RigidTransform(RotationMatrix.MakeXRotation(np.pi / 2),
                          [0, length / 2., 0])
    geom = GeometryInstance(X_FT.multiply(X_TG), Cylinder(radius, length),
                            name + " y-axis")
    geom.set_illustration_properties(
        MakePhongIllustrationProperties([0, 1, 0, opacity]))
    scene_graph.RegisterGeometry(source_id, frame_id, geom)

    # z-axis
    X_TG = RigidTransform([0, 0, length / 2.])
    geom = GeometryInstance(X_FT.multiply(X_TG), Cylinder(radius, length),
                            name + " z-axis")
    geom.set_illustration_properties(
        MakePhongIllustrationProperties([0, 0, 1, opacity]))
    scene_graph.RegisterGeometry(source_id, frame_id, geom)


def AddMultibodyTriad(frame, scene_graph, length=.25, radius=0.01, opacity=1.):
    plant = frame.GetParentPlant()
    AddTriad(plant.get_source_id(),
             plant.GetBodyFrameIdOrThrow(frame.body().index()), scene_graph,
             length, radius, opacity, frame.GetFixedPoseInBodyFrame())


def MakeManipulationStation(time_step=0.002,
                            plant_setup_callback=None,
                            camera_prefix="camera"):
    """
    Sets up a manipulation station with the iiwa + wsg + controllers [+
    cameras].  Cameras will be added to each body with a name starting with
    "camera".

    Args:
        time_step: the standard MultibodyPlant time step.

        plant_setup_callback: should be a python callable that takes one
        argument: `plant_setup_callback(plant)`.  It will be called after the
        iiwa and WSG are added to the plant, but before the plant is
        finalized.  Use this to add additional geometry.

        camera_prefix: Any bodies in the plant (created during the
        plant_setup_callback) starting with this prefix will get a camera
        attached.
    """
    builder = DiagramBuilder()

    # Add (only) the iiwa, WSG, and cameras to the scene.
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder,
                                                     time_step=time_step)
    iiwa = AddIiwa(plant)
    wsg = AddWsg(plant, iiwa)
    if plant_setup_callback:
        plant_setup_callback(plant)
    plant.Finalize()

    num_iiwa_positions = plant.num_positions(iiwa)

    # I need a PassThrough system so that I can export the input port.
    iiwa_position = builder.AddSystem(PassThrough(num_iiwa_positions))
    builder.ExportInput(iiwa_position.get_input_port(), "iiwa_position")
    builder.ExportOutput(iiwa_position.get_output_port(),
                         "iiwa_position_command")

    # Export the iiwa "state" outputs.
    demux = builder.AddSystem(
        Demultiplexer(2 * num_iiwa_positions, num_iiwa_positions))
    builder.Connect(plant.get_state_output_port(iiwa), demux.get_input_port())
    builder.ExportOutput(demux.get_output_port(0), "iiwa_position_measured")
    builder.ExportOutput(demux.get_output_port(1), "iiwa_velocity_estimated")
    builder.ExportOutput(plant.get_state_output_port(iiwa),
                         "iiwa_state_estimated")

    # Make the plant for the iiwa controller to use.
    controller_plant = MultibodyPlant(time_step=time_step)
    controller_iiwa = AddIiwa(controller_plant)
    AddWsg(controller_plant, controller_iiwa, welded=True)
    controller_plant.Finalize()

    # Add the iiwa controller
    iiwa_controller = builder.AddSystem(
        InverseDynamicsController(controller_plant,
                                  kp=[100] * num_iiwa_positions,
                                  ki=[1] * num_iiwa_positions,
                                  kd=[20] * num_iiwa_positions,
                                  has_reference_acceleration=False))
    iiwa_controller.set_name("iiwa_controller")
    builder.Connect(plant.get_state_output_port(iiwa),
                    iiwa_controller.get_input_port_estimated_state())

    # Add in the feed-forward torque
    adder = builder.AddSystem(Adder(2, num_iiwa_positions))
    builder.Connect(iiwa_controller.get_output_port_control(),
                    adder.get_input_port(0))
    # Use a PassThrough to make the port optional (it will provide zero values
    # if not connected).
    torque_passthrough = builder.AddSystem(PassThrough([0]
                                                       * num_iiwa_positions))
    builder.Connect(torque_passthrough.get_output_port(),
                    adder.get_input_port(1))
    builder.ExportInput(torque_passthrough.get_input_port(),
                        "iiwa_feedforward_torque")
    builder.Connect(adder.get_output_port(),
                    plant.get_actuation_input_port(iiwa))

    # Add discrete derivative to command velocities.
    desired_state_from_position = builder.AddSystem(
        StateInterpolatorWithDiscreteDerivative(
            num_iiwa_positions, time_step, suppress_initial_transient=True))
    desired_state_from_position.set_name("desired_state_from_position")
    builder.Connect(desired_state_from_position.get_output_port(),
                    iiwa_controller.get_input_port_desired_state())
    builder.Connect(iiwa_position.get_output_port(),
                    desired_state_from_position.get_input_port())

    # Export commanded torques.
    builder.ExportOutput(adder.get_output_port(), "iiwa_torque_commanded")
    builder.ExportOutput(adder.get_output_port(), "iiwa_torque_measured")

    builder.ExportOutput(plant.get_generalized_contact_forces_output_port(iiwa),
                         "iiwa_torque_external")

    # Wsg controller.
    wsg_controller = builder.AddSystem(SchunkWsgPositionController())
    wsg_controller.set_name("wsg_controller")
    builder.Connect(wsg_controller.get_generalized_force_output_port(),
                    plant.get_actuation_input_port(wsg))
    builder.Connect(plant.get_state_output_port(wsg),
                    wsg_controller.get_state_input_port())
    builder.ExportInput(wsg_controller.get_desired_position_input_port(),
                        "wsg_position")
    builder.ExportInput(wsg_controller.get_force_limit_input_port(),
                        "wsg_force_limit")
    wsg_mbp_state_to_wsg_state = builder.AddSystem(
        MakeMultibodyStateToWsgStateSystem())
    builder.Connect(plant.get_state_output_port(wsg),
                    wsg_mbp_state_to_wsg_state.get_input_port())
    builder.ExportOutput(wsg_mbp_state_to_wsg_state.get_output_port(),
                         "wsg_state_measured")
    builder.ExportOutput(wsg_controller.get_grip_force_output_port(),
                         "wsg_force_measured")

    # Cameras.
    AddRgbdSensors(builder,
                   plant,
                   scene_graph,
                   model_instance_prefix=camera_prefix)

    # Export "cheat" ports.
    builder.ExportOutput(scene_graph.get_query_output_port(), "geometry_query")
    builder.ExportOutput(plant.get_contact_results_output_port(),
                         "contact_results")
    builder.ExportOutput(plant.get_state_output_port(),
                         "plant_continuous_state")
    builder.ExportOutput(plant.get_body_poses_output_port(), "body_poses")

    diagram = builder.Build()
    diagram.set_name("ManipulationStation")
    return diagram
