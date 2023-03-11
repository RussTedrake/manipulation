import pydrake.multibody.parsing
import pydrake.multibody.plant
import pydrake.systems.framework

from manipulation.utils import ConfigureParser

filename = "package://manipulation/two_bins_w_cameras.dmd.yaml"

builder = pydrake.systems.framework.DiagramBuilder()
plant, scene_graph = pydrake.multibody.plant.AddMultibodyPlantSceneGraph(
    builder, time_step=0.001
)
parser = pydrake.multibody.parsing.Parser(plant)
ConfigureParser(parser)
parser.AddModelsFromUrl(filename)
plant.Finalize()

diagram = builder.Build()
context = diagram.CreateDefaultContext()
