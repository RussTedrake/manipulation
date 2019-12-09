import numpy as np
import os

from pydrake.common import FindResourceOrThrow
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import ConstantVectorSource
from pydrake.systems.planar_scenegraph_visualizer import PlanarSceneGraphVisualizer

builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlantSceneGraph(builder)

Parser(plant, scene_graph).AddModelFromFile(FindResourceOrThrow(
    "drake/manipulation/models/iiwa_description/urdf/planar_iiwa14_spheres_dense_elbow_collision.urdf"))  # noqa
plant.Finalize()

visualizer = builder.AddSystem(PlanarSceneGraphVisualizer(scene_graph))
builder.Connect(scene_graph.get_pose_bundle_output_port(),
                visualizer.get_input_port(0))

torque_command = builder.AddSystem(ConstantVectorSource(np.zeros(3)))
builder.Connect(torque_command.get_output_port(0), plant.get_actuation_input_port())

diagram = builder.Build()

simulator = Simulator(diagram)
simulator.set_target_realtime_rate(1.0)

simulator.AdvanceTo(5)
