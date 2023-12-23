import argparse
import sys

import bosdyn.client
import bosdyn.client.util
import numpy as np
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, get_a_tform_b
from bosdyn.client.image import ImageClient
from pydrake.common.yaml import yaml_dump
from pydrake.math import RigidTransform, RollPitchYaw

CAMERA_NAMES = [
    "frontleft",
    "frontright",
    "left",
    "right",
    "back",
    "hand",
]

COLOR_CAMERA_NAMES = [
    "frontleft_fisheye_image",
    "frontright_fisheye_image",
    "left_fisheye_image",
    "right_fisheye_image",
    "back_fisheye_image",
    "hand_color_image",
]


def main(argv):
    # Parse arguments.
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    options = parser.parse_args(argv)

    # Create robot object with an image client.
    sdk = bosdyn.client.create_standard_sdk("image_capture")
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)

    robot.sync_with_directory()
    robot.time_sync.wait_for_sync()

    image_client = robot.ensure_client(ImageClient.default_service_name)

    camera_configs = {}
    for camera_name, color_image_response in zip(
        CAMERA_NAMES, image_client.get_image_from_sources(COLOR_CAMERA_NAMES)
    ):
        # Get camera intrinsics.
        camera_configs[camera_name] = {}
        camera_configs[camera_name]["name"] = camera_name

        camera_configs[camera_name]["width"] = color_image_response.source.cols
        camera_configs[camera_name]["height"] = color_image_response.source.rows

        camera_configs[camera_name]["focal"] = {
            "x": color_image_response.source.pinhole.intrinsics.focal_length.x,
            "y": color_image_response.source.pinhole.intrinsics.focal_length.y,
        }
        camera_configs[camera_name][
            "center_x"
        ] = color_image_response.source.pinhole.intrinsics.principal_point.x
        camera_configs[camera_name][
            "center_y"
        ] = color_image_response.source.pinhole.intrinsics.principal_point.y

        # Obtain transformations to camera body and its sensor frame.
        if "hand" in camera_name:
            # The hand cameras body frames match the sensor frames.
            camera_body_frame_name = "hand_color_image_sensor"

            X_PB = RigidTransform(
                get_a_tform_b(
                    color_image_response.shot.transforms_snapshot,
                    "arm0.link_wr1",
                    camera_body_frame_name,
                ).to_matrix()
            )

            camera_configs[camera_name]["X_PB"] = {}
            camera_configs[camera_name]["X_PB"]["base_frame"] = "arm_link_wr1"
            camera_configs[camera_name]["X_PB"][
                "translation"
            ] = X_PB.translation().tolist()
            camera_configs[camera_name]["X_PB"]["rotation"] = {
                "deg": (RollPitchYaw(X_PB.rotation()).vector() * 180 / np.pi).tolist(),
                "_tag": "!Rpy",
            }
        else:
            camera_body_frame_name = camera_name
            X_PB = RigidTransform(
                get_a_tform_b(
                    color_image_response.shot.transforms_snapshot,
                    BODY_FRAME_NAME,
                    camera_body_frame_name,
                ).to_matrix()
            )

            camera_configs[camera_name]["X_PB"] = {}
            camera_configs[camera_name]["X_PB"]["base_frame"] = "body"
            camera_configs[camera_name]["X_PB"][
                "translation"
            ] = X_PB.translation().tolist()
            camera_configs[camera_name]["X_PB"]["rotation"] = {
                "deg": (RollPitchYaw(X_PB.rotation()).vector() * 180 / np.pi).tolist(),
                "_tag": "!Rpy",
            }

            # The image frame does not coincide with the camera housing frame.
            camera_sensor_frame_name = camera_name + "_fisheye"
            X_BC = RigidTransform(
                get_a_tform_b(
                    color_image_response.shot.transforms_snapshot,
                    camera_body_frame_name,
                    camera_sensor_frame_name,
                ).to_matrix()
            )

            camera_configs[camera_name]["X_BC"] = {}
            camera_configs[camera_name]["X_BC"][
                "translation"
            ] = X_BC.translation().tolist()
            camera_configs[camera_name]["X_BC"]["rotation"] = {
                "deg": (RollPitchYaw(X_BC.rotation()).vector() * 180 / np.pi).tolist(),
                "_tag": "!Rpy",
            }

    # Print yaml string to console.
    camera_scenario_data = {"cameras": camera_configs}
    yaml_str = yaml_dump(
        camera_scenario_data,
    )
    print(yaml_str)


if __name__ == "__main__":
    if not main(sys.argv[1:]):
        sys.exit(1)
