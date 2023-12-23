from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.perception import Concatenate
from pydrake.systems.framework import DiagramBuilder

from manipulation.scenarios import AddRgbdSensors
from manipulation.utils import ConfigureParser


def MustardExampleSystem():
    builder = DiagramBuilder()

    # Create the physics engine + scene graph.
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)
    ConfigureParser(parser)
    parser.AddModelsFromUrl("package://manipulation/mustard_w_cameras.dmd.yaml")
    plant.Finalize()

    # Add a visualizer just to help us see the object.
    use_meshcat = False
    if use_meshcat:
        meshcat = builder.AddSystem(MeshcatVisualizer(scene_graph))
        builder.Connect(
            scene_graph.get_query_output_port(),
            meshcat.get_geometry_query_input_port(),
        )

    AddRgbdSensors(builder, plant, scene_graph)

    diagram = builder.Build()
    diagram.set_name("depth_camera_demo_system")
    return diagram


def MustardPointCloud(normals=False, down_sample=True):
    system = MustardExampleSystem()
    context = system.CreateDefaultContext()
    plant = system.GetSubsystemByName("plant")
    plant_context = plant.GetMyMutableContextFromRoot(context)

    pcd = []
    for i in range(3):
        cloud = system.GetOutputPort(f"camera{i}_point_cloud").Eval(context)

        # Crop to region of interest.
        pcd.append(cloud.Crop(lower_xyz=[-0.3, -0.3, -0.3], upper_xyz=[0.3, 0.3, 0.3]))

        if normals:
            pcd[i].EstimateNormals(radius=0.1, num_closest=30)

            camera = plant.GetModelInstanceByName(f"camera{i}")
            body = plant.GetBodyByName("base", camera)
            X_C = plant.EvalBodyPoseInWorld(plant_context, body)
            pcd[i].FlipNormalsTowardPoint(X_C.translation())

    # Merge point clouds.
    merged_pcd = Concatenate(pcd)
    if not down_sample:
        return merged_pcd
    # Down sample.
    down_sampled_pcd = merged_pcd.VoxelizedDownSample(voxel_size=0.005)
    return down_sampled_pcd
