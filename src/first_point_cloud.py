import matplotlib.pyplot as plt
import numpy as np

from pydrake.common import FindResourceOrThrow
from pydrake.geometry.render import DepthCameraProperties, MakeRenderEngineVtk, RenderEngineVtkParams
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.meshcat_visualizer import (
  MeshcatVisualizer, MeshcatPointCloudVisualizer )
from pydrake.systems.sensors import RgbdSensor

builder = DiagramBuilder()

# Create the physics engine + scene graph.
plant, scene_graph = AddMultibodyPlantSceneGraph(builder)
# Add a single object into it.
Parser(plant, scene_graph).AddModelFromFile(FindResourceOrThrow(
  "drake/manipulation/models/ycb/sdf/006_mustard_bottle.sdf"))
# Add a rendering engine
renderer = "my_renderer"
scene_graph.AddRenderer(renderer, MakeRenderEngineVtk(RenderEngineVtkParams()))
plant.Finalize()

# Add a visualizer just to help us see the object.
use_meshcat = False
if use_meshcat:
  meshcat = builder.AddSystem(MeshcatVisualizer(scene_graph))
  builder.Connect(scene_graph.get_pose_bundle_output_port(),
      meshcat.get_input_port(0))

# Add a camera to the environment.
pose = RigidTransform(RollPitchYaw(-0.2, 0.2, 0), [-.1, -0.1, -.5])  
properties = DepthCameraProperties(width=640, height=480, fov_y=np.pi/4.0, renderer_name=renderer, z_near=0.1, z_far=10.0)
camera = builder.AddSystem(RgbdSensor(parent_id=scene_graph.world_frame_id(),
  X_PB=pose, properties=properties, show_window=False))
builder.Connect(scene_graph.get_query_output_port(),   
  camera.query_object_input_port())

diagram = builder.Build()

# Evaluate the camera output ports to get the images.
context = diagram.CreateDefaultContext()
camera_context = diagram.GetSubsystemContext(camera, context)
color_image = camera.color_image_output_port().Eval(camera_context)
depth_image = camera.depth_image_32F_output_port().Eval(camera_context)

if use_meshcat:
  meshcat.load()
  diagram.Publish(context)

# Plot the two images.
plt.subplot(121)
plt.imshow(color_image.data)
plt.title('Color image')
plt.subplot(122)
plt.imshow(np.squeeze(depth_image.data))
plt.title('Depth image')
plt.show()