import os

import pydrake.multibody.parsing
import pydrake.multibody.plant
import pydrake.systems.framework
import pydrake.systems.meshcat_visualizer

import manipulation.utils

filename = manipulation.utils.FindResource("models/two_bins_w_cameras.yaml")

builder = pydrake.systems.framework.DiagramBuilder()
plant, scene_graph = pydrake.multibody.plant.AddMultibodyPlantSceneGraph(
    builder, time_step=0.001)
parser = pydrake.multibody.parsing.Parser(plant)
manipulation.utils.AddPackagePaths(parser)
pydrake.multibody.parsing.ProcessModelDirectives(
    pydrake.multibody.parsing.LoadModelDirectives(filename), plant, parser)

plant.Finalize()

meshcat = pydrake.systems.meshcat_visualizer.ConnectMeshcatVisualizer(
    builder, scene_graph, zmq_url="new", open_browser=False)
diagram = builder.Build()
context = diagram.CreateDefaultContext()

meshcat.load()
diagram.Publish(context)
