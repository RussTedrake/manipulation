"""
This file contains a number of helper utilities to set up our various
experiments with less code.
"""

import os
import sys
import warnings
from enum import Enum

import numpy as np
from pydrake.all import (
    AddCompliantHydroelasticProperties,
    AddContactMaterial,
    Adder,
    AddMultibodyPlantSceneGraph,
    BallRpyJoint,
    BaseField,
    Box,
    CameraInfo,
    Capsule,
    ClippingRange,
    CoulombFriction,
    Cylinder,
    Demultiplexer,
    DepthImageToPointCloud,
    DepthRange,
    DepthRenderCamera,
    DiagramBuilder,
    DifferentialInverseKinematicsIntegrator,
    Frame,
    GeometryInstance,
    InverseDynamicsController,
    MakeMultibodyStateToWsgStateSystem,
    MakePhongIllustrationProperties,
    MakeRenderEngineVtk,
    ModelInstanceIndex,
    MultibodyPlant,
    Parser,
    PassThrough,
    PrismaticJoint,
    ProximityProperties,
    RenderCameraCore,
    RenderEngineVtkParams,
    RevoluteJoint,
    Rgba,
    RgbdSensor,
    RigidTransform,
    RollPitchYaw,
    RotationMatrix,
    SchunkWsgPositionController,
    SpatialInertia,
    Sphere,
    StateInterpolatorWithDiscreteDerivative,
    UnitInertia,
)

from manipulation.systems import AddIiwaDifferentialIK as systems_AddIiwaDifferentialIK
from manipulation.systems import ExtractPose
from manipulation.utils import ConfigureParser

ycb = [
    "003_cracker_box.sdf",
    "004_sugar_box.sdf",
    "005_tomato_soup_can.sdf",
    "006_mustard_bottle.sdf",
    "009_gelatin_box.sdf",
    "010_potted_meat_can.sdf",
]


def ExtractBodyPose(output_port, body_index, X_BA=RigidTransform()) -> ExtractPose:
    warnings.warn(
        "ExtractBodyPose is deprecated. Use manipulation.systems.ExtractPose instead."
    )
    return ExtractPose(body_index, X_BA)


def AddIiwa(plant, collision_model="no_collision"):
    parser = Parser(plant)
    iiwa = parser.AddModelsFromUrl(
        f"package://drake_models/iiwa_description/sdf/iiwa7_{collision_model}.sdf"
    )[0]
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
    urdf = "package://drake_models/iiwa_description/urdf/planar_iiwa14_spheres_dense_elbow_collision.urdf"

    parser = Parser(plant)
    iiwa = parser.AddModelsFromUrl(urdf)[0]
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
    parser = Parser(plant)
    ConfigureParser(parser)
    iiwa = parser.AddModelsFromUrl("package://manipulation/two_link_iiwa14.urdf")[0]
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0"))

    # Set default positions:
    index = 0
    for joint_index in plant.GetJointIndices(iiwa):
        joint = plant.get_mutable_joint(joint_index)
        if isinstance(joint, RevoluteJoint):
            joint.set_default_angle(q0[index])
            index += 1

    return iiwa


class WsgPositions(Enum):
    OPEN = 0.107
    CLOSED = 0.002


def AddWsg(plant, iiwa_model_instance, roll=np.pi / 2.0, welded=False, sphere=False):
    parser = Parser(plant)
    ConfigureParser(parser)
    if welded:
        if sphere:
            file = "package://manipulation/schunk_wsg_50_welded_fingers_sphere.sdf"
        else:
            file = "package://manipulation/schunk_wsg_50_welded_fingers.sdf"
    else:
        file = (
            "package://drake_models/wsg_50_description/sdf/schunk_wsg_50_with_tip.sdf"
        )

    directives = f"""
directives:
- add_model:
    name: gripper
    file: {file}
"""
    gripper = parser.AddModelsFromString(directives, ".dmd.yaml")[0]

    X_7G = RigidTransform(RollPitchYaw(np.pi / 2.0, 0, roll), [0, 0, 0.09])
    plant.WeldFrames(
        plant.GetFrameByName("iiwa_link_7", iiwa_model_instance),
        plant.GetFrameByName("body", gripper),
        X_7G,
    )
    return gripper


def AddFloatingXyzJoint(plant, frame, instance, actuators=False):
    inertia = UnitInertia.SolidSphere(1.0)
    x_body = plant.AddRigidBody(
        "x",
        instance,
        SpatialInertia(mass=0, p_PScm_E=[0.0, 0.0, 0.0], G_SP_E=inertia),
    )
    x_joint = plant.AddJoint(
        PrismaticJoint("x", plant.world_frame(), x_body.body_frame(), [1, 0, 0])
    )
    if actuators:
        plant.AddJointActuator("x", x_joint)
    y_body = plant.AddRigidBody(
        "y",
        instance,
        SpatialInertia(mass=0, p_PScm_E=[0.0, 0.0, 0.0], G_SP_E=inertia),
    )
    y_joint = plant.AddJoint(
        PrismaticJoint("y", x_body.body_frame(), y_body.body_frame(), [0, 1, 0])
    )
    if actuators:
        plant.AddJointActuator("y", y_joint)
    z_joint = plant.AddJoint(PrismaticJoint("z", y_body.body_frame(), frame, [0, 0, 1]))
    if actuators:
        plant.AddJointActuator("z", z_joint)


def AddFloatingRpyJoint(plant, frame, instance):
    inertia = UnitInertia.SolidSphere(1.0)
    z_body = plant.AddRigidBody(
        "z",
        instance,
        SpatialInertia(mass=0, p_PScm_E=[0.0, 0.0, 0.0], G_SP_E=inertia),
    )
    AddFloatingXyzJoint(plant, z_body.body_frame(), instance)
    plant.AddJoint(BallRpyJoint("ball", z_body.body_frame(), frame))


def AddShape(plant, shape, name, mass=1, mu=1, color=[0.5, 0.5, 0.9, 1.0]):
    instance = plant.AddModelInstance(name)
    # TODO: Add a method to UnitInertia that accepts a geometry shape (unless
    # that dependency is somehow gross) and does this.
    if isinstance(shape, Box):
        inertia = UnitInertia.SolidBox(shape.width(), shape.depth(), shape.height())
    elif isinstance(shape, Cylinder):
        inertia = UnitInertia.SolidCylinder(shape.radius(), shape.length(), [0, 0, 1])
    elif isinstance(shape, Sphere):
        inertia = UnitInertia.SolidSphere(shape.radius())
    elif isinstance(shape, Capsule):
        inertia = UnitInertia.SolidCapsule(shape.radius(), shape.length(), [0, 0, 1])
    else:
        raise RuntimeError(f"need to write the unit inertia for shapes of type {shape}")
    body = plant.AddRigidBody(
        name,
        instance,
        SpatialInertia(mass=mass, p_PScm_E=np.array([0.0, 0.0, 0.0]), G_SP_E=inertia),
    )
    if plant.geometry_source_is_registered():
        proximity_properties = ProximityProperties()
        AddContactMaterial(1e4, 1e7, CoulombFriction(mu, mu), proximity_properties)
        AddCompliantHydroelasticProperties(0.01, 1e8, proximity_properties)
        plant.RegisterCollisionGeometry(
            body, RigidTransform(), shape, name, proximity_properties
        )

        plant.RegisterVisualGeometry(body, RigidTransform(), shape, name, color)

    return instance


def AddCamera(builder, scene_graph, X_WC, depth_camera=None, renderer=None):
    warnings.warn("Please use AddRgbdSensor instead.", warnings.DeprecationWarning)
    return AddRgbdSensor(builder, scene_graph, X_WC, depth_camera, renderer)


def AddRgbdSensor(
    builder,
    scene_graph,
    X_PC,
    depth_camera=None,
    renderer=None,
    parent_frame_id=None,
):
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
        scene_graph.AddRenderer(renderer, MakeRenderEngineVtk(RenderEngineVtkParams()))

    if not depth_camera:
        depth_camera = DepthRenderCamera(
            RenderCameraCore(
                renderer,
                CameraInfo(width=640, height=480, fov_y=np.pi / 4.0),
                ClippingRange(near=0.1, far=10.0),
                RigidTransform(),
            ),
            DepthRange(0.1, 10.0),
        )

    rgbd = builder.AddSystem(
        RgbdSensor(
            parent_id=parent_frame_id,
            X_PB=X_PC,
            depth_camera=depth_camera,
            show_window=False,
        )
    )

    builder.Connect(scene_graph.get_query_output_port(), rgbd.query_object_input_port())

    return rgbd


def AddRgbdSensors(
    builder,
    plant,
    scene_graph,
    also_add_point_clouds=True,
    model_instance_prefix="camera",
    depth_camera=None,
    renderer=None,
):
    """
    Adds a RgbdSensor to the first body in the plant for every model instance
    with a name starting with model_instance_prefix.  If depth_camera is None,
    then a default camera info will be used.  If renderer is None, then we will
    assume the name 'my_renderer', and create a VTK renderer if a renderer of
    that name doesn't exist.
    """
    if sys.platform == "linux" and os.getenv("DISPLAY") is None:
        from pyvirtualdisplay import Display

        virtual_display = Display(visible=0, size=(1400, 900))
        virtual_display.start()

    if not renderer:
        renderer = "my_renderer"

    if not scene_graph.HasRenderer(renderer):
        scene_graph.AddRenderer(renderer, MakeRenderEngineVtk(RenderEngineVtkParams()))

    if not depth_camera:
        depth_camera = DepthRenderCamera(
            RenderCameraCore(
                renderer,
                CameraInfo(width=640, height=480, fov_y=np.pi / 4.0),
                ClippingRange(near=0.1, far=10.0),
                RigidTransform(),
            ),
            DepthRange(0.1, 10.0),
        )

    for index in range(plant.num_model_instances()):
        model_instance_index = ModelInstanceIndex(index)
        model_name = plant.GetModelInstanceName(model_instance_index)

        if model_name.startswith(model_instance_prefix):
            body_index = plant.GetBodyIndices(model_instance_index)[0]
            rgbd = builder.AddSystem(
                RgbdSensor(
                    parent_id=plant.GetBodyFrameIdOrThrow(body_index),
                    X_PB=RigidTransform(),
                    depth_camera=depth_camera,
                    show_window=False,
                )
            )
            rgbd.set_name(model_name)

            builder.Connect(
                scene_graph.get_query_output_port(),
                rgbd.query_object_input_port(),
            )

            # Export the camera outputs
            builder.ExportOutput(
                rgbd.color_image_output_port(), f"{model_name}_rgb_image"
            )
            builder.ExportOutput(
                rgbd.depth_image_32F_output_port(), f"{model_name}_depth_image"
            )
            builder.ExportOutput(
                rgbd.label_image_output_port(), f"{model_name}_label_image"
            )

            if also_add_point_clouds:
                # Add a system to convert the camera output into a point cloud
                to_point_cloud = builder.AddSystem(
                    DepthImageToPointCloud(
                        camera_info=rgbd.default_depth_render_camera()
                        .core()
                        .intrinsics(),
                        fields=BaseField.kXYZs | BaseField.kRGBs,
                    )
                )
                builder.Connect(
                    rgbd.depth_image_32F_output_port(),
                    to_point_cloud.depth_image_input_port(),
                )
                builder.Connect(
                    rgbd.color_image_output_port(),
                    to_point_cloud.color_image_input_port(),
                )

                camera_pose = builder.AddSystem(ExtractPose(body_index))
                builder.Connect(
                    plant.get_body_poses_output_port(),
                    camera_pose.get_input_port(),
                )
                builder.Connect(
                    camera_pose.get_output_port(),
                    to_point_cloud.GetInputPort("camera_pose"),
                )

                # Export the point cloud output.
                builder.ExportOutput(
                    to_point_cloud.point_cloud_output_port(),
                    f"{model_name}_point_cloud",
                )


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


def AddTriad(
    source_id,
    frame_id,
    scene_graph,
    length=0.25,
    radius=0.01,
    opacity=1.0,
    X_FT=RigidTransform(),
    name="frame",
):
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
    X_TG = RigidTransform(RotationMatrix.MakeYRotation(np.pi / 2), [length / 2.0, 0, 0])
    geom = GeometryInstance(
        X_FT.multiply(X_TG), Cylinder(radius, length), name + " x-axis"
    )
    geom.set_illustration_properties(
        MakePhongIllustrationProperties([1, 0, 0, opacity])
    )
    scene_graph.RegisterGeometry(source_id, frame_id, geom)

    # y-axis
    X_TG = RigidTransform(RotationMatrix.MakeXRotation(np.pi / 2), [0, length / 2.0, 0])
    geom = GeometryInstance(
        X_FT.multiply(X_TG), Cylinder(radius, length), name + " y-axis"
    )
    geom.set_illustration_properties(
        MakePhongIllustrationProperties([0, 1, 0, opacity])
    )
    scene_graph.RegisterGeometry(source_id, frame_id, geom)

    # z-axis
    X_TG = RigidTransform([0, 0, length / 2.0])
    geom = GeometryInstance(
        X_FT.multiply(X_TG), Cylinder(radius, length), name + " z-axis"
    )
    geom.set_illustration_properties(
        MakePhongIllustrationProperties([0, 0, 1, opacity])
    )
    scene_graph.RegisterGeometry(source_id, frame_id, geom)


def AddMultibodyTriad(frame, scene_graph, length=0.25, radius=0.01, opacity=1.0):
    plant = frame.GetParentPlant()
    AddTriad(
        plant.get_source_id(),
        plant.GetBodyFrameIdOrThrow(frame.body().index()),
        scene_graph,
        length,
        radius,
        opacity,
        frame.GetFixedPoseInBodyFrame(),
        name=frame.name(),
    )


def AddIiwaDifferentialIK(
    builder: DiagramBuilder, plant: MultibodyPlant, frame: Frame | None = None
) -> DifferentialInverseKinematicsIntegrator:
    warnings.warn(
        "Please use manipulation.systems.AddIiwaDifferentialIK instead.",
        DeprecationWarning,
    )
    if frame is None:
        frame = plant.GetFrameByName("body")
    return systems_AddIiwaDifferentialIK(builder, plant, frame)


# This method is effectively deprecated, and should not be used. See MakeHardwareStation
# instead.
def MakeManipulationStation(
    model_directives=None,
    filename=None,
    time_step=0.002,
    iiwa_prefix="iiwa",
    wsg_prefix="wsg",
    camera_prefix="camera",
    prefinalize_callback=None,
    package_xmls=[],
):
    """
    Creates a manipulation station system, which is a sub-diagram containing:
      - A MultibodyPlant with populated via the Parser from the
        `model_directives` argument AND the `filename` argument.
      - A SceneGraph
      - For each model instance starting with `iiwa_prefix`, we add an
        additional iiwa controller system
      - For each model instance starting with `wsg_prefix`, we add an
        additional schunk controller system
      - For each body starting with `camera_prefix`, we add a RgbdSensor

    Args:
        builder: a DiagramBuilder

        model_directives: a string containing any model directives to be parsed

        filename: a string containing the name of an sdf, urdf, mujoco xml, or
        model directives yaml file.

        time_step: the standard MultibodyPlant time step.

        iiwa_prefix: Any model instances starting with `iiwa_prefix` will get
        an inverse dynamics controller, etc attached

        wsg_prefix: Any model instance starting with `wsg_prefix` will get a
        schunk controller

        camera_prefix: Any bodies in the plant (created during the
        plant_setup_callback) starting with this prefix will get a camera
        attached.

        prefinalize_callback: A function, setup(plant), that will be called
        with the multibody plant before calling finalize.  This can be useful
        for e.g. adding additional bodies/models to the simulation.

        package_xmls: A list of filenames to be passed to
        PackageMap.AddPackageXml().  This is useful if you need to add more
        models to your path (e.g. from your current working directory).
    """
    warnings.warn(
        "MakeManipulationStation is deprecated. Use MakeHardwareStation instead.",
        DeprecationWarning,
    )

    builder = DiagramBuilder()

    # Add (only) the iiwa, WSG, and cameras to the scene.
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=time_step)
    parser = Parser(plant)
    for p in package_xmls:
        parser.package_map().AddPackageXml(p)
    ConfigureParser(parser)
    if model_directives:
        parser.AddModelsFromString(model_directives, ".dmd.yaml")
    if filename:
        parser.AddModelsFromUrl(filename)
    if prefinalize_callback:
        prefinalize_callback(plant)
    plant.Finalize()

    for i in range(plant.num_model_instances()):
        model_instance = ModelInstanceIndex(i)
        model_instance_name = plant.GetModelInstanceName(model_instance)

        if model_instance_name.startswith(iiwa_prefix):
            num_iiwa_positions = plant.num_positions(model_instance)

            # I need a PassThrough system so that I can export the input port.
            iiwa_position = builder.AddSystem(PassThrough(num_iiwa_positions))
            builder.ExportInput(
                iiwa_position.get_input_port(),
                model_instance_name + "_position",
            )
            builder.ExportOutput(
                iiwa_position.get_output_port(),
                model_instance_name + "_position_commanded",
            )

            # Export the iiwa "state" outputs.
            demux = builder.AddSystem(
                Demultiplexer(2 * num_iiwa_positions, num_iiwa_positions)
            )
            builder.Connect(
                plant.get_state_output_port(model_instance),
                demux.get_input_port(),
            )
            builder.ExportOutput(
                demux.get_output_port(0),
                model_instance_name + "_position_measured",
            )
            builder.ExportOutput(
                demux.get_output_port(1),
                model_instance_name + "_velocity_estimated",
            )
            builder.ExportOutput(
                plant.get_state_output_port(model_instance),
                model_instance_name + "_state_estimated",
            )

            # Make the plant for the iiwa controller to use.
            controller_plant = MultibodyPlant(time_step=time_step)
            # TODO: Add the correct IIWA model (introspected from MBP)
            if plant.num_positions(model_instance) == 3:
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
            iiwa_controller.set_name(model_instance_name + "_controller")
            builder.Connect(
                plant.get_state_output_port(model_instance),
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
                model_instance_name + "_torque",
            )
            builder.Connect(
                adder.get_output_port(),
                plant.get_actuation_input_port(model_instance),
            )

            # Add discrete derivative to command velocities.
            desired_state_from_position = builder.AddSystem(
                StateInterpolatorWithDiscreteDerivative(
                    num_iiwa_positions,
                    time_step,
                    suppress_initial_transient=True,
                )
            )
            desired_state_from_position.set_name(
                model_instance_name + "_desired_state_from_position"
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
                model_instance_name + "_torque_commanded",
            )
            builder.ExportOutput(
                adder.get_output_port(),
                model_instance_name + "_torque_measured",
            )

            builder.ExportOutput(
                plant.get_generalized_contact_forces_output_port(model_instance),
                model_instance_name + "_torque_external",
            )

        elif model_instance_name.startswith(wsg_prefix):
            # Wsg controller.
            wsg_controller = builder.AddSystem(SchunkWsgPositionController())
            wsg_controller.set_name(model_instance_name + "_controller")
            builder.Connect(
                wsg_controller.get_generalized_force_output_port(),
                plant.get_actuation_input_port(model_instance),
            )
            builder.Connect(
                plant.get_state_output_port(model_instance),
                wsg_controller.get_state_input_port(),
            )
            builder.ExportInput(
                wsg_controller.get_desired_position_input_port(),
                model_instance_name + "_position",
            )
            builder.ExportInput(
                wsg_controller.get_force_limit_input_port(),
                model_instance_name + "_force_limit",
            )
            wsg_mbp_state_to_wsg_state = builder.AddSystem(
                MakeMultibodyStateToWsgStateSystem()
            )
            builder.Connect(
                plant.get_state_output_port(model_instance),
                wsg_mbp_state_to_wsg_state.get_input_port(),
            )
            builder.ExportOutput(
                wsg_mbp_state_to_wsg_state.get_output_port(),
                model_instance_name + "_state_measured",
            )
            builder.ExportOutput(
                wsg_controller.get_grip_force_output_port(),
                model_instance_name + "_force_measured",
            )

    # Cameras.
    AddRgbdSensors(builder, plant, scene_graph, model_instance_prefix=camera_prefix)

    # Export "cheat" ports.
    builder.ExportOutput(scene_graph.get_query_output_port(), "query_object")
    builder.ExportOutput(plant.get_contact_results_output_port(), "contact_results")
    builder.ExportOutput(plant.get_state_output_port(), "plant_continuous_state")
    builder.ExportOutput(plant.get_body_poses_output_port(), "body_poses")

    diagram = builder.Build()
    diagram.set_name("ManipulationStation")
    return diagram
