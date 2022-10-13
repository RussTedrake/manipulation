import os

import pydrake.multibody.parsing
import pydrake.multibody.plant
import pydrake.systems.framework

import manipulation.utils

filename = manipulation.utils.FindResource("models/two_bins_w_cameras.dmd.yaml")

builder = pydrake.systems.framework.DiagramBuilder()
plant, scene_graph = pydrake.multibody.plant.AddMultibodyPlantSceneGraph(
    builder, time_step=0.001)
parser = pydrake.multibody.parsing.Parser(plant)
manipulation.utils.AddPackagePaths(parser)
parser.AddAllModelsFromFile(filename)
plant.Finalize()

diagram = builder.Build()
context = diagram.CreateDefaultContext()
