# TODO: Change this to the three camera setup.  Make sure to pull images from
# both bins.

# TODO: Compute mean and std for dataset and add the normalization transform.
# https://github.com/pytorch/examples/blob/97304e232807082c2e7b54c597615dc0ad8f6173/imagenet/main.py#L197-L198

# TODO: Check if I'm running as a unit test and only do a single image.

import argparse
import json
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
from PIL import Image
import os
import shutil
import warnings

from pydrake.all import (
    AddMultibodyPlantSceneGraph,
    DiagramBuilder,
    FindResourceOrThrow,
    Parser,
    RandomGenerator,
    RigidTransform,
    Role,
    RollPitchYaw,
    Simulator,
    UniformlyRandomRotationMatrix,
)
from manipulation.scenarios import ycb, AddRgbdSensor
from manipulation.utils import colorize_labels

parser = argparse.ArgumentParser(
    description='Install ToC and Navigation into book html files.')
parser.add_argument('--test', action='store_true')
args = parser.parse_args()

if args.test:
    from pydrake.common.deprecation import DrakeDeprecationWarning
    warnings.simplefilter("error", DrakeDeprecationWarning)

debug = True
path = '/tmp/clutter_maskrcnn_data'
num_images = 10000 if not args.test else 2

if not debug and not args.test:
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    print(f'Creating dataset in {path} with {num_images} images')

rng = np.random.default_rng()  # this is for python
generator = RandomGenerator(rng.integers(1000))  # for c++


def generate_image(image_num):
    filename_base = os.path.join(path, f"{image_num:05d}")

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0005)
    parser = Parser(plant)
    parser.AddModelFromFile(
        FindResourceOrThrow(
            "drake/examples/manipulation_station/models/bin.sdf"))
    plant.WeldFrames(plant.world_frame(), plant.GetFrameByName("bin_base"))
    inspector = scene_graph.model_inspector()

    instance_id_to_class_name = dict()

    for object_num in range(rng.integers(1, 10)):
        this_object = ycb[rng.integers(len(ycb))]
        class_name = os.path.splitext(this_object)[0]
        sdf = FindResourceOrThrow("drake/manipulation/models/ycb/sdf/"
                                  + this_object)
        instance = parser.AddModelFromFile(sdf, f"object{object_num}")

        frame_id = plant.GetBodyFrameIdOrThrow(
            plant.GetBodyIndices(instance)[0])
        geometry_ids = inspector.GetGeometries(frame_id, Role.kPerception)
        for geom_id in geometry_ids:
            instance_id_to_class_name[int(
                inspector.GetPerceptionProperties(geom_id).GetProperty(
                    "label", "id"))] = class_name

    plant.Finalize()

    if not debug and not args.test:
        with open(filename_base + ".json", "w") as f:
            json.dump(instance_id_to_class_name, f)

    camera = AddRgbdSensor(
        builder, scene_graph,
        RigidTransform(RollPitchYaw(np.pi, 0, np.pi / 2.0), [0, 0, .8]))
    camera.set_name("rgbd_sensor")
    builder.ExportOutput(camera.color_image_output_port(), "color_image")
    builder.ExportOutput(camera.label_image_output_port(), "label_image")

    diagram = builder.Build()

    while True:
        simulator = Simulator(diagram)
        context = simulator.get_mutable_context()
        plant_context = plant.GetMyContextFromRoot(context)

        z = 0.1
        for body_index in plant.GetFloatingBaseBodies():
            tf = RigidTransform(
                UniformlyRandomRotationMatrix(generator),
                [rng.uniform(-.15, .15),
                 rng.uniform(-.2, .2), z])
            plant.SetFreeBodyPose(plant_context, plant.get_body(body_index), tf)
            z += 0.1

        try:
            simulator.AdvanceTo(1.0)
            break
        except RuntimeError:
            # I've chosen an aggressive simulation time step which works most
            # of the time, but can fail occasionally.
            pass

    color_image = diagram.GetOutputPort("color_image").Eval(context)
    label_image = diagram.GetOutputPort("label_image").Eval(context)

    if args.test:
        pass
    elif debug:
        plt.figure()
        plt.subplot(121)
        plt.imshow(color_image.data)
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(colorize_labels(label_image.data))
        plt.axis('off')
        plt.show()
    else:
        Image.fromarray(color_image.data).save(f"{filename_base}.png")
        np.save(f"{filename_base}_mask", label_image.data)


if args.test or debug:
    for image_num in range(num_images):
        generate_image(image_num)
else:
    from tqdm import tqdm
    pool = multiprocessing.Pool(10)
    list(tqdm(pool.imap(generate_image, range(num_images)), total=num_images))
    pool.close()
    pool.join()
