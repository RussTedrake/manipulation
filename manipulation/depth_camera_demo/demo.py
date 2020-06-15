import numpy as np
import open3d as o3d

from pydrake.common import FindResourceOrThrow, set_log_level
from pydrake.geometry.render import (DepthCameraProperties, MakeRenderEngineVtk,
                                     RenderEngineVtkParams)
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.perception import BaseField, DepthImageToPointCloud
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.sensors import RgbdSensor


def DepthCameraDemoSystem():
    builder = DiagramBuilder()

    # If you have trouble finding resources, you can enable trace logging
    # to see how `FindResource*` is searching.
    set_log_level("trace")

    # Create the physics engine + scene graph.
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    # Add a single object into it.
    Parser(plant, scene_graph).AddModelFromFile(
        FindResourceOrThrow(
            "drake/manipulation/models/ycb/sdf/006_mustard_bottle.sdf"))
    # Add a rendering engine
    renderer = "my_renderer"
    scene_graph.AddRenderer(renderer,
                            MakeRenderEngineVtk(RenderEngineVtkParams()))
    plant.Finalize()

    # Add a visualizer just to help us see the object.
    use_meshcat = False
    if use_meshcat:
        meshcat = builder.AddSystem(MeshcatVisualizer(scene_graph))
        builder.Connect(scene_graph.get_pose_bundle_output_port(),
                        meshcat.get_input_port(0))

    # Add a camera to the environment.
    pose = RigidTransform(RollPitchYaw(-0.2, 0.2, 0), [-.1, -0.1, -.5])
    properties = DepthCameraProperties(width=640,
                                       height=480,
                                       fov_y=np.pi / 4.0,
                                       renderer_name=renderer,
                                       z_near=0.1,
                                       z_far=10.0)
    camera = builder.AddSystem(
        RgbdSensor(parent_id=scene_graph.world_frame_id(),
                   X_PB=pose,
                   properties=properties,
                   show_window=False))
    camera.set_name("rgbd_sensor")
    builder.Connect(scene_graph.get_query_output_port(),
                    camera.query_object_input_port())

    # Export the camera outputs
    builder.ExportOutput(camera.color_image_output_port(), "color_image")
    builder.ExportOutput(camera.depth_image_32F_output_port(), "depth_image")

    # Add a system to convert the camera output into a point cloud
    to_point_cloud = builder.AddSystem(
        DepthImageToPointCloud(camera_info=camera.depth_camera_info(),
                               fields=BaseField.kXYZs | BaseField.kRGBs))
    builder.Connect(camera.depth_image_32F_output_port(),
                    to_point_cloud.depth_image_input_port())
    builder.Connect(camera.color_image_output_port(),
                    to_point_cloud.color_image_input_port())

    # Export the point cloud output.
    builder.ExportOutput(to_point_cloud.point_cloud_output_port(),
                         "point_cloud")

    diagram = builder.Build()
    diagram.set_name("depth_camera_demo_system")
    return diagram


def create_open3d_rgbd_image(color_image, depth_image):
    color_image = o3d.geometry.Image(np.copy(
        color_image.data[:, :, :3]))  # No alpha
    depth_image = o3d.geometry.Image(np.squeeze(np.copy(depth_image.data)))
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=color_image,
        depth=depth_image,
        depth_scale=1.0,
        depth_trunc=3.0,
        convert_rgb_to_intensity=False)
    return rgbd_image
