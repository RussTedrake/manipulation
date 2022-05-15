import os

import pydrake.multibody.parsing
import pydrake.multibody.plant
import pydrake.systems.framework

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

diagram = builder.Build()
context = diagram.CreateDefaultContext()
