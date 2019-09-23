import pydrake as drake
import os
from underactuated import PlanarSceneGraphVisualizer

builder = drake.systems.framework.DiagramBuilder()
plant, scene_graph = drake.multibody.plant.AddMultibodyPlantSceneGraph(builder)

drake.multibody.parsing.Parser(plant, scene_graph).AddModelFromFile(drake.common.FindResourceOrThrow("drake/manipulation/models/iiwa_description/urdf/planar_iiwa14_spheres_dense_elbow_collision.urdf"))
plant.Finalize()

visualizer = builder.AddSystem(PlanarSceneGraphVisualizer(scene_graph))
builder.Connect(scene_graph.get_pose_bundle_output_port(),
        visualizer.get_input_port(0))

diagram = builder.Build()

simulator = Simulator(diagram)
simulator.set_target_realtime_rate(1.0)

simulator.AdvanceTo(5)
