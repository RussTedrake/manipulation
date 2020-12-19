"""
This file contains a number of helper utilities to set up our various
experiments with less code.
"""
import numpy as np
import os

import pydrake.all
from pydrake.all import (AbstractValue, BaseField, ModelInstanceIndex,
                         DepthCameraProperties, DepthImageToPointCloud,
                         LeafSystem, MakeRenderEngineVtk, RenderEngineVtkParams,
                         RgbdSensor)
from pydrake.all import RigidTransform, RollPitchYaw
from manipulation.utils import FindResource

ycb = [
    "003_cracker_box.sdf", "004_sugar_box.sdf", "005_tomato_soup_can.sdf",
    "006_mustard_bottle.sdf", "009_gelatin_box.sdf", "010_potted_meat_can.sdf"
]


def AddIiwa(plant, collision_model="no_collision"):
    sdf_path = pydrake.common.FindResourceOrThrow(
        "drake/manipulation/models/iiwa_description/iiwa7/"
        f"iiwa7_{collision_model}.sdf")

    parser = pydrake.multibody.parsing.Parser(plant)
    iiwa = parser.AddModelFromFile(sdf_path)
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0"))

    # Set default positions:
    q0 = [0.0, 0.1, 0, -1.2, 0, 1.6, 0]
    index = 0
    for joint_index in plant.GetJointIndices(iiwa):
        joint = plant.get_mutable_joint(joint_index)
        if isinstance(joint, pydrake.multibody.tree.RevoluteJoint):
            joint.set_default_angle(q0[index])
            index += 1

    return iiwa


def AddPlanarIiwa(plant):
    urdf = pydrake.common.FindResourceOrThrow(
        "drake/manipulation/models/iiwa_description/urdf/"
        "planar_iiwa14_spheres_dense_elbow_collision.urdf")

    parser = pydrake.multibody.parsing.Parser(plant)
    iiwa = parser.AddModelFromFile(urdf)
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0"))

    # Set default positions:
    q0 = [0.1, -1.2, 1.6]
    index = 0
    for joint_index in plant.GetJointIndices(iiwa):
        joint = plant.get_mutable_joint(joint_index)
        if isinstance(joint, pydrake.multibody.tree.RevoluteJoint):
            joint.set_default_angle(q0[index])
            index += 1

    return iiwa


def AddTwoLinkIiwa(plant, q0=[0.1, -1.2]):
    urdf = FindResource("models/two_link_iiwa14.urdf")

    parser = pydrake.multibody.parsing.Parser(plant)
    parser.package_map().Add(
        "iiwa_description",
        os.path.dirname(
            pydrake.common.FindResourceOrThrow(
                "drake/manipulation/models/iiwa_description/package.xml")))
    iiwa = parser.AddModelFromFile(urdf)
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("iiwa_link_0"))

    # Set default positions:
    index = 0
    for joint_index in plant.GetJointIndices(iiwa):
        joint = plant.get_mutable_joint(joint_index)
        if isinstance(joint, pydrake.multibody.tree.RevoluteJoint):
            joint.set_default_angle(q0[index])
            index += 1

    return iiwa


# TODO: take argument for whether we want the welded fingers version or not
def AddWsg(plant, iiwa_model_instance, roll=np.pi / 2.0, welded=False):
    parser = pydrake.multibody.parsing.Parser(plant)
    if welded:
        parser.package_map().Add(
            "wsg_50_description",
            os.path.dirname(
                pydrake.common.FindResourceOrThrow(
                    "drake/manipulation/models/wsg_50_description/package.xml"))
        )
        gripper = parser.AddModelFromFile(
            FindResource("models/schunk_wsg_50_welded_fingers.sdf"), "gripper")
    else:
        gripper = parser.AddModelFromFile(
            pydrake.common.FindResourceOrThrow(
                "drake/manipulation/models/"
                "wsg_50_description/sdf/schunk_wsg_50_no_tip.sdf"))

    X_7G = RigidTransform(RollPitchYaw(np.pi / 2.0, 0, roll), [0, 0, 0.114])
    plant.WeldFrames(plant.GetFrameByName("iiwa_link_7", iiwa_model_instance),
                     plant.GetFrameByName("body", gripper), X_7G)
    return gripper


def AddShape(plant, shape, name, mass=1, mu=1, color=[.5, .5, .9, 1.0]):
    instance = plant.AddModelInstance(name)
    # TODO: Add a method to UnitInertia that accepts a geometry shape (unless
    # that dependency is somehow gross) and does this.
    if isinstance(shape, pydrake.geometry.Box):
        inertia = pydrake.multibody.tree.UnitInertia.SolidBox(
            shape.width(), shape.depth(), shape.height())
    elif isinstance(shape, pydrake.geometry.Cylinder):
        inertia = pydrake.multibody.tree.UnitInertia.SolidCylinder(
            shape.radius(), shape.length())
    elif isinstance(shape, pydrake.geometry.Sphere):
        inertia = pydrake.multibody.tree.UnitInertia.SolidSphere(shape.radius())
    else:
        raise RunTimeError(
            f"need to write the unit inertia for shapes of type {shape}")
    body = plant.AddRigidBody(
        name, instance,
        pydrake.multibody.tree.SpatialInertia(mass=mass,
                                              p_PScm_E=np.array([0., 0., 0.]),
                                              G_SP_E=inertia))
    if plant.geometry_source_is_registered():
        if isinstance(shape, pydrake.geometry.Box):
            plant.RegisterCollisionGeometry(
                body, RigidTransform(),
                pydrake.geometry.Box(shape.width() - 0.001,
                                     shape.depth() - 0.001,
                                     shape.height() - 0.001), name,
                pydrake.multibody.plant.CoulombFriction(mu, mu))
            i = 0
            for x in [-shape.width() / 2.0, shape.width() / 2.0]:
                for y in [-shape.depth() / 2.0, shape.depth() / 2.0]:
                    for z in [-shape.height() / 2.0, shape.height() / 2.0]:
                        plant.RegisterCollisionGeometry(
                            body, RigidTransform([x, y, z]),
                            pydrake.geometry.Sphere(radius=1e-7),
                            f"contact_sphere{i}",
                            pydrake.multibody.plant.CoulombFriction(mu, mu))
                        i += 1
        else:
            plant.RegisterCollisionGeometry(
                body, RigidTransform(), shape, name,
                pydrake.multibody.plant.CoulombFriction(mu, mu))

        plant.RegisterVisualGeometry(body, RigidTransform(), shape, name, color)

    return instance


def AddRgbdSensors(builder,
                   plant,
                   scene_graph,
                   also_add_point_clouds=True,
                   model_instance_prefix="camera",
                   properties=None,
                   renderer=None):
    """
    Adds a RgbdSensor to every body in the plant with a name starting with
    body_prefix.  If camera_info is None, then a default camera info will be
    used.  If renderer is None, then we will assume the name 'my_renderer', and
    create a VTK renderer if a renderer of that name doesn't exist.
    """
    if not renderer:
        renderer = "my_renderer"

    if not scene_graph.HasRenderer(renderer):
        scene_graph.AddRenderer(renderer,
                                MakeRenderEngineVtk(RenderEngineVtkParams()))

    if not properties:
        properties = DepthCameraProperties(width=640,
                                           height=480,
                                           fov_y=np.pi / 4.0,
                                           renderer_name=renderer,
                                           z_near=0.1,
                                           z_far=10.0)

    for index in range(plant.num_model_instances()):
        model_instance_index = ModelInstanceIndex(index)
        model_name = plant.GetModelInstanceName(model_instance_index)

        if model_name.startswith(model_instance_prefix):
            body_index = plant.GetBodyIndices(model_instance_index)[0]
            rgbd = builder.AddSystem(
                RgbdSensor(parent_id=plant.GetBodyFrameIdOrThrow(body_index),
                           X_PB=RigidTransform(),
                           properties=properties,
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
        # just waiting on drake #14259
        #        if not inspector.BelongsToSource(gid, source_id):
        #            continue
        props = inspector.GetIllustrationProperties(gid)
        if props is None or not props.HasProperty("phong", "diffuse"):
            continue
        c = props.GetProperty("phong", "diffuse")
        new_color = pydrake.geometry.Rgba(c.r(), c.g(), c.b(), alpha)
        props.UpdateProperty("phong", "diffuse", new_color)


def SetColor(scene_graph, color, source_id, geometry_ids=None):
    inspector = scene_graph.model_inspector()
    if not geometry_ids:
        geometry_ids = inspector.GetAllGeometryIds()

    for gid in geometry_ids:
        # just waiting on drake #14259
        #        if not inspector.BelongsToSource(gid, source_id):
        #            continue
        props = inspector.GetIllustrationProperties(gid)
        if props is None or not props.HasProperty("phong", "diffuse"):
            continue
        new_color = pydrake.geometry.Rgba(color[0], color[1], color[2],
                                          color[3])
        props.UpdateProperty("phong", "diffuse", new_color)
