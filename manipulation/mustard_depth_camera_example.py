import numpy as np
import open3d as o3d

from pydrake.multibody.parsing import (Parser, ProcessModelDirectives,
                                       LoadModelDirectives)
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.framework import DiagramBuilder

from manipulation.open3d_utils import (create_open3d_point_cloud,
                                       open3d_cloud_to_drake)
from manipulation.scenarios import AddRgbdSensors
from manipulation.utils import FindResource, AddPackagePaths


def MustardExampleSystem():
    builder = DiagramBuilder()

    # Create the physics engine + scene graph.
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)
    AddPackagePaths(parser)
    ProcessModelDirectives(
        LoadModelDirectives(FindResource("models/mustard_w_cameras.yaml")),
        plant, parser)

    plant.Finalize()

    # Add a visualizer just to help us see the object.
    use_meshcat = False
    if use_meshcat:
        meshcat = builder.AddSystem(MeshcatVisualizer(scene_graph))
        builder.Connect(scene_graph.get_query_output_port(),
                        meshcat.get_geometry_query_input_port())

    AddRgbdSensors(builder, plant, scene_graph)

    diagram = builder.Build()
    diagram.set_name("depth_camera_demo_system")
    return diagram


def MustardPointCloud():
    system = MustardExampleSystem()
    context = system.CreateDefaultContext()
    pcd = []
    for i in range(3):
        point_cloud = system.GetOutputPort(f"camera{i}_point_cloud").Eval(
            context)

        cloud = create_open3d_point_cloud(point_cloud)
        # Crop to region of interest.
        pcd.append(
            cloud.crop(
                o3d.geometry.AxisAlignedBoundingBox(min_bound=[-.3, -.3, -.3],
                                                    max_bound=[.3, .3, .3])))
    # Merge point clouds.
    merged_pcd = pcd[0] + pcd[1] + pcd[2]
    # Voxelize down-sample.
    down_sampled_pcd = merged_pcd.voxel_down_sample(voxel_size=0.005)

    return open3d_cloud_to_drake(down_sampled_pcd)
