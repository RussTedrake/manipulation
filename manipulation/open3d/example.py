import numpy as np

from pydrake.common import FindResourceOrThrow, set_log_level
from pydrake.common.value import AbstractValue
from pydrake.geometry import Box
from pydrake.geometry.render import (DepthCameraProperties, MakeRenderEngineVtk,
                                     RenderEngineVtkParams)
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.multibody.tree import SpatialInertia
from pydrake.perception import BaseField, DepthImageToPointCloud
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.sensors import RgbdSensor
from pydrake.systems.primitives import ConstantValueSource


def DepthCameraExampleSystem():
    builder = DiagramBuilder()

    # If you have trouble finding resources, you can enable trace logging
    # to see how `FindResource*` is searching.
    set_log_level("trace")

    # Create the physics engine + scene graph.
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    # Add a single object into it.
    X_Mustard = RigidTransform(RollPitchYaw(-np.pi / 2., 0, -np.pi / 2.),
                               [0, 0, 0.09515])
    mustard = Parser(plant).AddModelFromFile(
        FindResourceOrThrow(
            "drake/manipulation/models/ycb/sdf/006_mustard_bottle.sdf"))
    plant.WeldFrames(plant.world_frame(),
                     plant.GetFrameByName("base_link_mustard", mustard),
                     X_Mustard)
    # Add a rendering engine
    renderer = "my_renderer"
    scene_graph.AddRenderer(renderer,
                            MakeRenderEngineVtk(RenderEngineVtkParams()))

    # TODO(russt): Use model directives for this
    X_Camera = RigidTransform(
        RotationMatrix.MakeYRotation(-0.2).multiply(
            RollPitchYaw(-np.pi / 2.0, 0.0, np.pi / 2.0).ToRotationMatrix()),
        [.5, 0, .2])
    camera_instance = plant.AddModelInstance("cameras")
    camera_num = 0
    X_C = []
    camera = []
    for theta in [0.2, 2.2, 4.2]:
        X_C.append(
            RigidTransform(
                RotationMatrix.MakeZRotation(theta)).multiply(X_Camera))
        # Add a box for the camera in the environment.
        camera.append(
            plant.AddRigidBody(f"camera{camera_num}", camera_instance,
                               SpatialInertia()))
        plant.RegisterVisualGeometry(camera[camera_num],
                                     RigidTransform([0, 0, -0.01]),
                                     Box(width=.1, depth=.02, height=.02),
                                     "D415", [.4, .4, .4, 1.])
        plant.WeldFrames(plant.world_frame(), camera[camera_num].body_frame(),
                         X_C[camera_num])
        camera_num += 1

    plant.Finalize()

    # Add a visualizer just to help us see the object.
    use_meshcat = False
    if use_meshcat:
        meshcat = builder.AddSystem(MeshcatVisualizer(scene_graph))
        builder.Connect(scene_graph.get_pose_bundle_output_port(),
                        meshcat.get_input_port(0))

    # Add some cameras to the environment.
    properties = DepthCameraProperties(width=640,
                                       height=480,
                                       fov_y=np.pi / 4.0,
                                       renderer_name=renderer,
                                       z_near=0.1,
                                       z_far=10.0)
    rgbd = []
    to_point_cloud = []
    for i in range(camera_num):
        rgbd.append(builder.AddSystem(RgbdSensor(
            parent_id=plant.GetBodyFrameIdOrThrow(camera[i].index()),
            X_PB=RigidTransform(),
            properties=properties,
            show_window=False)))
        rgbd[i].set_name(f"rgbd_sensor{i}")
        builder.Connect(scene_graph.get_query_output_port(),
                        rgbd[i].query_object_input_port())

        # Export the camera outputs
        builder.ExportOutput(rgbd[i].color_image_output_port(),
                             f"color_image{i}")
        builder.ExportOutput(rgbd[i].depth_image_32F_output_port(),
                             f"depth_image{i}")

        # Add a system to convert the camera output into a point cloud
        to_point_cloud.append(
            builder.AddSystem(
                DepthImageToPointCloud(camera_info=rgbd[i].depth_camera_info(),
                                       fields=BaseField.kXYZs
                                       | BaseField.kRGBs)))
        builder.Connect(rgbd[i].depth_image_32F_output_port(),
                        to_point_cloud[i].depth_image_input_port())
        builder.Connect(rgbd[i].color_image_output_port(),
                        to_point_cloud[i].color_image_input_port())
        camera_pose = builder.AddSystem(
            ConstantValueSource(AbstractValue.Make(X_C[i])))
        builder.Connect(camera_pose.get_output_port(),
                        to_point_cloud[i].GetInputPort("camera_pose"))

        # Export the point cloud output.
        builder.ExportOutput(to_point_cloud[i].point_cloud_output_port(),
                             f"point_cloud{i}")

    diagram = builder.Build()
    diagram.set_name("depth_camera_demo_system")
    return diagram
