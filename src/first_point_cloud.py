from pydrake.examples.manipulation_station import ManipulationStation
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.meshcat_visualizer import (
  MeshcatVisualizer, MeshcatPointCloudVisualizer )

builder = DiagramBuilder()

# Set up the manipulation station.
station = builder.AddSystem(ManipulationStation())
station.SetupManipulationClassStation()
station.Finalize()

# Visualize the station geometry in meshcat.
meshcat = builder.AddSystem(MeshcatVisualizer(station.get_scene_graph()))
builder.Connect(station.GetOutputPort("pose_bundle"),
                        meshcat.get_input_port(0))

# Convert the depth camera output into a point cloud
builder.AddSystem(DepthImageToPointCloud())

# TODO(russt) set camera pose.

meshlab_point_cloud = builder.AddSystem(MeshcatPointCloudVisualizer(meshcat))
builder.Connect()

diagram = builder.Build()
context = diagram.CreateDefaultContext()

meshcat.load()
diagram.Publish(context)