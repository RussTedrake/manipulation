import argparse
import os
import re
import xml.etree.ElementTree as ET

import pymeshlab
from pydrake.all import PackageMap


def _convert_mesh(input_filename, output_filename, scale, overwrite):
    if not overwrite and os.path.exists(output_filename):
        print(f"Note: {output_filename} already exists. Skipping conversion.")
        return

    # Create a new PyMeshLab mesh set
    ms = pymeshlab.MeshSet()

    # Load the mesh file
    ms.load_new_mesh(input_filename)

    if scale is not None:
        scale = [float(s) for s in scale]
        assert len(scale) == 3, "Scale must be a 3-element array."

        # First apply absolute value scaling
        ms.apply_filter(
            "compute_matrix_from_translation_rotation_scale",
            scalex=scale[0],
            scaley=scale[1],
            scalez=scale[2],
            freeze=True,
        )

    # Save the mesh
    ms.save_current_mesh(
        output_filename, save_face_color=False, save_vertex_color=False
    )
    print(f"Converted {input_filename} to {output_filename}")


def _convert_urdf(input_filename, output_filename, package_map, overwrite):
    with open(input_filename, "r") as file:
        urdf_content = file.read()

    # Remove XML comments to avoid matching filenames inside them
    urdf_content_no_comments = re.sub(r"<!--.*?-->", "", urdf_content, flags=re.DOTALL)

    # Regex pattern to find entire XML nodes with .stl or .dae extensions in the filename attribute
    resource_pattern = re.compile(
        r'<(\w+)\s+[^>]*filename=["\']([^"\']+\.(stl|dae))["\'][^>]*>', re.IGNORECASE
    )

    # Find all matches in the URDF content without comments
    matches = resource_pattern.finditer(urdf_content_no_comments)

    modified_urdf_content = urdf_content

    for match in matches:
        node_text = match.group(0)
        node = ET.fromstring(node_text)
        filename = node.attrib["filename"]
        scale = node.attrib.get("scale")
        if scale:
            scale_suffix = "_scaled_" + scale.replace("-", "n").replace(" ", "_")
            scale = scale.split()
        else:
            scale = None
            scale_suffix = ""

        input_mesh_path = package_map.ResolveUrl(filename)
        output_obj_path = input_mesh_path.rsplit(".", 1)[0] + scale_suffix + ".obj"
        obj_filename = filename.rsplit(".", 1)[0] + scale_suffix + ".obj"

        _convert_mesh(
            input_mesh_path, output_obj_path, scale=scale, overwrite=overwrite
        )

        # Update the node's filename attribute
        node.attrib["filename"] = obj_filename

        # If scale is not None, update the scale attribute
        if scale:
            node.attrib["scale"] = "1 1 1"

        # Convert the node back to a string
        node_string = ET.tostring(node, encoding="unicode")

        # Replace the original match with the updated node string
        modified_urdf_content = modified_urdf_content.replace(node_text, node_string)

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
        _convert_urdf(input_filename, output_filename, package_map, overwrite=overwrite)
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
        overwrite=args.overwrite,
    )
