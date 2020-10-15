import numpy as np

from pydrake.common import FindResourceOrThrow
from pydrake.common.value import AbstractValue
from pydrake.geometry import Box
from pydrake.geometry.render import (DepthCameraProperties, MakeRenderEngineVtk,
                                     RenderEngineVtkParams)
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.multibody.parsing import (Parser, ProcessModelDirectives,
                                       LoadModelDirectives)
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.perception import BaseField, DepthImageToPointCloud
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.sensors import RgbdSensor
from pydrake.systems.primitives import ConstantValueSource

from manipulation.scenarios import AddRgbdSensors
from manipulation.utils import FindResource


def MustardExampleSystem():
    builder = DiagramBuilder()

    # Create the physics engine + scene graph.
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    parser = Parser(plant)
    parser.package_map().Add("manipulation", FindResource("models"))
    ProcessModelDirectives(
        LoadModelDirectives(FindResource("models/mustard_w_cameras.yaml")),
        plant, parser)

    plant.Finalize()

    # Add a visualizer just to help us see the object.
    use_meshcat = False
    if use_meshcat:
        meshcat = builder.AddSystem(MeshcatVisualizer(scene_graph))
        builder.Connect(scene_graph.get_pose_bundle_output_port(),
                        meshcat.get_input_port(0))

    AddRgbdSensors(builder, plant, scene_graph)

    diagram = builder.Build()
    diagram.set_name("depth_camera_demo_system")
    return diagram
