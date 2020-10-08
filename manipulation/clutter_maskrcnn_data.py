
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
from tqdm import tqdm
from PIL import Image

import pydrake.all
from pydrake.all import RigidTransform, RollPitchYaw
from manipulation.scenarios import ycb
from manipulation.utils import colorize_labels

debug = False
path = '/tmp/clutter_maskrcnn_data'
num_images = 1 if debug else 10000

shutil.rmtree(path)
os.makedirs(path)
print(f'Creating dataset in {path} with {num_images} images')

rng = np.random.default_rng()  # this is for python
generator = pydrake.common.RandomGenerator(rng.integers(1000))  # for c++

pbar = tqdm(total=num_images)

# Note: Could (trivially) parallelize this.
for image_num in range(num_images):
    filename_base = os.path.join(path, f"{image_num:05d}")

    builder = pydrake.systems.framework.DiagramBuilder()
    plant, scene_graph = pydrake.multibody.plant.AddMultibodyPlantSceneGraph(builder, time_step=0.0005)
    parser = pydrake.multibody.parsing.Parser(plant)
    parser.AddModelFromFile(pydrake.common.FindResourceOrThrow(
        "drake/examples/manipulation_station/models/bin.sdf"))
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("bin_base"))
    inspector = scene_graph.model_inspector()

    instance_id_to_class_name = dict()

    for object_num in range(rng.integers(1,10)):
        this_object = ycb[rng.integers(len(ycb))]
        class_name = os.path.splitext(this_object)[0]
        sdf = pydrake.common.FindResourceOrThrow("drake/manipulation/models/ycb/sdf/" + this_object)
        instance = parser.AddModelFromFile(sdf, f"object{object_num}")

        frame_id = plant.GetBodyFrameIdOrThrow(
            plant.GetBodyIndices(instance)[0])
        geometry_ids = inspector.GetGeometries(
            frame_id, pydrake.geometry.Role.kPerception)
        for geom_id in geometry_ids:
            instance_id_to_class_name[int(inspector.GetPerceptionProperties(geom_id).GetProperty("label", "id"))] = class_name

    plant.Finalize()

    with open(filename_base + ".json", "w") as f:
        json.dump(instance_id_to_class_name, f)

    renderer = "my_renderer"
    scene_graph.AddRenderer(
        renderer, pydrake.geometry.render.MakeRenderEngineVtk(pydrake.geometry.render.RenderEngineVtkParams()))
    properties = pydrake.geometry.render.DepthCameraProperties(width=640,
                                        height=480,
                                        fov_y=np.pi / 4.0,
                                        renderer_name=renderer,
                                        z_near=0.1,
                                        z_far=10.0)
    camera = builder.AddSystem(
        pydrake.systems.sensors.RgbdSensor(parent_id=scene_graph.world_frame_id(),
                    X_PB=RigidTransform(
                        RollPitchYaw(np.pi, 0, np.pi/2.0),
                        [0, 0, .8]),
                    properties=properties,
                    show_window=False))
    camera.set_name("rgbd_sensor")
    builder.Connect(scene_graph.get_query_output_port(),
                    camera.query_object_input_port())
    builder.ExportOutput(camera.color_image_output_port(), "color_image")
    builder.ExportOutput(camera.label_image_output_port(), "label_image")

    diagram = builder.Build()

    while True:
        simulator = pydrake.systems.analysis.Simulator(diagram)
        context = simulator.get_mutable_context()
        plant_context = plant.GetMyContextFromRoot(context)

        z = 0.1
        for body_index in plant.GetFloatingBaseBodies():
            tf = RigidTransform(
                    pydrake.math.UniformlyRandomRotationMatrix(generator),  
                    [rng.uniform(-.15,.15), rng.uniform(-.2, .2), z])
            plant.SetFreeBodyPose(plant_context, 
                                  plant.get_body(body_index),
                                  tf)
            z += 0.1

        try:
            simulator.AdvanceTo(1.0)
            break
        except RuntimeError:
            # I've chosen an aggressive simulation time step which works most 
            # of the time, but can fail occasionally.
            pass

    color_image = diagram.GetOutputPort("color_image").Eval(context)
    Image.fromarray(color_image.data).save(f"{filename_base}.png")
    label_image = diagram.GetOutputPort("label_image").Eval(context)
    np.save(f"{filename_base}_mask", label_image.data)

    pbar.update(1)

    if debug: 
        plt.figure()
        plt.subplot(211)
        plt.imshow(color_image.data)
        plt.subplot(212)
        plt.imshow(colorize_labels(label_image.data))
        plt.axis('off')
