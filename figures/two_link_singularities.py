import numpy as np
from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    ConnectPlanarSceneGraphVisualizer,
    DiagramBuilder,
    Parser,
    PiecewisePolynomial,
)

from manipulation.utils import ConfigureParser

builder = DiagramBuilder()

plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
parser = Parser(plant)
ConfigureParser(parser)
parser.AddModelsFromUrl("package://manipulation/double_pendulum.urdf")
plant.Finalize()

viz = ConnectPlanarSceneGraphVisualizer(
    builder, scene_graph, show=False, xlim=[-0.2, 2.2], ylim=[-1.0, 1.0]
)
viz.fig.set_size_inches([3, 2.5])
diagram = builder.Build()
context = diagram.CreateDefaultContext()

T = 2.0
q = PiecewisePolynomial.FirstOrderHold(
    [0, T], np.array([[-np.pi / 2.0 + 1.0, -np.pi / 2.0 - 1.0], [-2.0, 2.0]])
)
plant_context = plant.GetMyContextFromRoot(context)

viz.start_recording()
for t in np.linspace(0, T, num=100):
    context.SetTime(t)
    plant.SetPositions(plant_context, q.value(t))
    diagram.ForcedPublish(context)
viz.stop_recording()
ani = viz.get_recording_as_animation(repeat=False)

f = open("two_link_singularities.html", "w")
f.write(ani.to_jshtml())
f.close()

# Then I edited the style in to
# <img style="height:200px" id="_anim
