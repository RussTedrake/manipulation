import argparse
import os
import re

import pymeshlab
from pydrake.all import PackageMap


def _convert_mesh(input_filename, output_filename, overwrite=False):
    if not overwrite and os.path.exists(output_filename):
        print(f"Note: {output_filename} already exists. Skipping conversion.")
        return

    # Create a new PyMeshLab mesh set
    ms = pymeshlab.MeshSet()

    # Load the mesh file
    ms.load_new_mesh(input_filename)

    # Save the mesh as an OBJ file
    ms.save_current_mesh(
        output_filename, save_face_color=False, save_vertex_color=False
    )
    print(f"Converted {input_filename} to {output_filename}")


def _convert_urdf(input_filename, output_filename, package_map):
    with open(input_filename, "r") as file:
        urdf_content = file.read()

    # Remove XML comments to avoid matching filenames inside them
    urdf_content_no_comments = re.sub(r"<!--.*?-->", "", urdf_content, flags=re.DOTALL)

    # Regex pattern to find resource names with .stl or .dae extensions
    resource_pattern = re.compile(
        r'filename=["\']([^"\']+\.(stl|dae))["\']', re.IGNORECASE
    )

    # Find all matches in the URDF content without comments
    matches = resource_pattern.findall(urdf_content_no_comments)

    modified_urdf_content = urdf_content

    for match, extension in matches:
        input_mesh_path = package_map.ResolveUrl(match)
        output_obj_path = input_mesh_path.rsplit(".", 1)[0] + ".obj"

        _convert_mesh(input_mesh_path, output_obj_path)

        obj_filename = match.rsplit(".", 1)[0] + ".obj"
        modified_urdf_content = modified_urdf_content.replace(match, obj_filename)

    # Regex pattern to find scale attributes
    scale_pattern = re.compile(
        r'scale=["\']([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)? [-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)? [-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)["\']'
    )

    # Until https://github.com/RobotLocomotion/drake/issues/22046 is resolved, I
    # need to just brute-force a uniform scale.

    # Find all scale attributes in the URDF content
    scale_matches = scale_pattern.findall(urdf_content_no_comments)

    for scale_match in scale_matches:
        # Split the scale values, convert them to float, and invert the sign on any negative values
        scale_values = [abs(float(value)) for value in scale_match.split()]
        # Calculate the average scale value
        average_scale = sum(scale_values) / len(scale_values)
        # Create a uniform scale string
        uniform_scale = f"{average_scale} {average_scale} {average_scale}"
        # Replace the original scale attribute with the uniform scale
        modified_urdf_content = modified_urdf_content.replace(
            f'scale="{scale_match}"', f'scale="{uniform_scale}"'
        )
        if scale_match != uniform_scale:
            print(
                f"Replaced scale '{scale_match}' with uniform scale "
                f"'{uniform_scale}' in URDF content."
            )

    with open(output_filename, "w") as file:
        file.write(modified_urdf_content)

    print(
        f"Converted URDF file '{input_filename}' to '{output_filename}' "
        f"to use OBJ instead of STL."
    )


def MakeDrakeCompatibleModel(
    input_filename: str,
    output_filename: str,
    populate_package_map_from_env: str = None,
    populate_package_map_from_folder: str = None,
    overwrite: bool = False,
) -> None:
    """Converts a model file to be compatible with the Drake multibody parsers.

    - Converts any .stl files to obj
      https://github.com/RobotLocomotion/drake/issues/19408
    - Converts any .dae files to obj
      https://github.com/RobotLocomotion/drake/issues/19109
    - Zaps any non-uniform scale attributes
      https://github.com/RobotLocomotion/drake/issues/22046

    Any new files will be created alongside the original files (e.g. .obj files will be created next to the existing .stl files); existing files will not be overwritten by default.

    Args:
        input_filename (str): The path to the input file to be converted.
        output_filename (str): The path where the converted file will be saved.
            Using the same string as input_filename is allowed.
        populate_package_map_from_env (str, optional): Environment variable to
            populate the package map from. Defaults to None.
        populate_package_map_from_folder (str, optional): Folder path to
            populate the package map from. Defaults to None.
        overwrite (bool, optional): Whether to overwrite existing files. Defaults
            to False.
    """
    package_map = PackageMap()

    if populate_package_map_from_env:
        package_map.PopulateFromEnvironment(populate_package_map_from_env)

    if populate_package_map_from_folder:
        package_map.PopulateFromFolder(populate_package_map_from_folder)

    # TODO: Support .sdf, .xml, etc.
    if input_filename.lower().endswith(".stl") or input_filename.lower().endswith(
        ".dae"
    ):
        _convert_mesh(input_filename, output_filename, overwrite)
    elif input_filename.lower().endswith(".urdf"):
        _convert_urdf(input_filename, output_filename, package_map)
    else:
        print(
            f"Warning: The file extension of '{input_filename}' is not "
            "supported yet; we currently support '.stl', '.dae', and '.urdf'."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert STL, DAE, or URDF files to use OBJ instead of STL or DAE."
    )
    parser.add_argument(
        "input_filename", type=str, help="The path to the input file to be converted."
    )
    parser.add_argument(
        "output_filename",
        type=str,
        help="The path where the converted file will be saved.",
    )
    parser.add_argument(
        "--populate_package_map_from_env",
        type=str,
        default=None,
        help="Environment variable to populate the package map from.",
    )
    parser.add_argument(
        "--populate_package_map_from_folder",
        type=str,
        default=None,
        help="Folder path to populate the package map from.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite existing files.",
    )

    args = parser.parse_args()

    MakeDrakeCompatibleModel(
        input_filename=args.input_filename,
        output_filename=args.output_filename,
        populate_package_map_from_env=args.populate_package_map_from_env,
        populate_package_map_from_folder=args.populate_package_map_from_folder,
    )
