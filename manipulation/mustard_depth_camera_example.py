import numpy as np

from pydrake.multibody.parsing import (Parser, ProcessModelDirectives,
                                       LoadModelDirectives)
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.perception import Concatenate
from pydrake.systems.framework import DiagramBuilder

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
        cloud = system.GetOutputPort(f"camera{i}_point_cloud").Eval(context)

        # Crop to region of interest.
        pcd.append(cloud.Crop(lower_xyz=[-.3, -.3, -.3], upper_xyz=[.3, .3,
                                                                    .3]))
    # Merge point clouds.
    merged_pcd = Concatenate(pcd)
    # Voxelize down-sample.
    down_sampled_pcd = merged_pcd.VoxelizedDownSample(voxel_size=0.005)

    return down_sampled_pcd
