import argparse
import sys
import bosdyn.client
from bosdyn.client.frame_helpers import get_a_tform_b, BODY_FRAME_NAME
import bosdyn.client.util
from bosdyn.client.image import ImageClient
import numpy as np
from pydrake.math import RigidTransform, RollPitchYaw
from pydrake.common.yaml import yaml_load_file, yaml_dump

CAMERA_NAMES = [
    "frontleft_fisheye_image",
    "frontleft_depth",
    "frontright_fisheye_image",
    "frontright_depth",
    "left_fisheye_image",
    "left_depth",
    "right_fisheye_image",
    "right_depth",
    "back_fisheye_image",
    "back_depth",
    "hand_color_image",
    "hand_depth",
]


def main(argv):
    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-scenario-file",
        help="The base scenario file you would like to append the camera data to.",
        default="./spot_with_arm_and_floating_base_actuators.scenario.yaml",
    )
    parser.add_argument(
        "--output-scenario-file",
        help="The desired output filename",
        default="./spot_with_arm_and_cameras.scenario.yaml",
    )
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
    for camera_name, image_response in zip(
        CAMERA_NAMES, image_client.get_image_from_sources(CAMERA_NAMES)
    ):
        # Get camera intrinsics.
        camera_configs[camera_name] = {}
        camera_configs[camera_name]["name"] = camera_name
        camera_configs[camera_name]["depth"] = True if "depth" in camera_name else False
        camera_configs[camera_name]["rgb"] = True if "color" in camera_name else False

        camera_configs[camera_name]["width"] = image_response.source.cols
        camera_configs[camera_name]["height"] = image_response.source.rows
        camera_configs[camera_name][
            "focal_x"
        ] = image_response.source.pinhole.intrinsics.focal_length.x
        camera_configs[camera_name][
            "focal_y"
        ] = image_response.source.pinhole.intrinsics.focal_length.y
        camera_configs[camera_name][
            "center_x"
        ] = image_response.source.pinhole.intrinsics.principal_point.x
        camera_configs[camera_name][
            "center_y"
        ] = image_response.source.pinhole.intrinsics.principal_point.y

        # Obtain transformations to camera body and its sensor frame.
        if "hand" in camera_name:
            # The hand cameras body frames match the sensor frames.
            camera_body_frame_name = (
                "hand_color_image_sensor"
                if "color" in camera_name
                else "hand_depth_sensor"
            )
            X_PB = RigidTransform(
                get_a_tform_b(
                    image_response.shot.transforms_snapshot,
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
            camera_body_frame_name = camera_name.split("_")[0]
            X_PB = RigidTransform(
                get_a_tform_b(
                    image_response.shot.transforms_snapshot,
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

            if "image" in camera_name:
                # Unlike the depth camera, the image frame does not coincide with the camera housing frame.
                camera_sensor_frame_name = camera_name.replace("_image", "")
                X_BC = RigidTransform(
                    get_a_tform_b(
                        image_response.shot.transforms_snapshot,
                        camera_body_frame_name,
                        camera_sensor_frame_name,
                    ).to_matrix()
                )

                camera_configs[camera_name]["X_BC"] = {}
                camera_configs[camera_name]["X_BC"][
                    "translation"
                ] = X_BC.translation().tolist()
                camera_configs[camera_name]["X_BC"]["rotation"] = {
                    "deg": (
                        RollPitchYaw(X_BC.rotation()).vector() * 180 / np.pi
                    ).tolist(),
                    "_tag": "!Rpy",
                }

    # Load in the base scenario file and append the camera data.
    spot_with_arm_data = yaml_load_file(options.base_scenario_file)
    yaml_dump(
        {**spot_with_arm_data, **{"cameras": camera_configs}},
        filename=options.output_scenario_file,
    )


if __name__ == "__main__":
    if not main(sys.argv[1:]):
        sys.exit(1)
