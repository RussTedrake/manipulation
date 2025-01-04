import argparse
import os
import re
import warnings

import pymeshlab
from lxml import etree
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
        node = etree.fromstring(node_text)
        filename = node.attrib["filename"]
        scale = node.attrib.get("scale")
        if scale:
            scale_suffix = "_scaled_" + scale.replace("-", "n").replace(" ", "_")
            scale = scale.split()
        else:
            scale = None
            scale_suffix = ""

        if mesh_filename.lower().startswith("package://"):
            input_mesh_path = package_map.ResolveUrl(mesh_filename)
        else:
            input_mesh_path = os.path.join(
                os.path.dirname(input_filename), mesh_filename
            )
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

    print(f"Converted URDF file '{input_filename}' to '{output_filename}'.")


def _convert_sdf(input_filename, output_filename, package_map, overwrite):
    raise NotImplementedError("SDF format is not supported yet, but it will be soon.")


def _process_includes(filename, processed_files=None):
    """Process include tags recursively and return complete DOM tree.

    Args:
        filename: Path to the input XML file
        processed_files: Set of already processed files to prevent cycles

    Returns:
        etree.ElementTree: Complete DOM tree with includes resolved
    """
    if processed_files is None:
        processed_files = set()

    if filename in processed_files:
        raise ValueError(
            f"Circular include detected: {filename} has already been processed"
        )

    processed_files.add(filename)

    # Parse the main file
    tree = etree.parse(filename)
    root = tree.getroot()

    # Find all include elements
    includes = root.findall(".//include")

    # Process each include
    for include in includes:
        # Get included file path relative to current file's directory
        include_file = include.get("file")
        print(f"Processing include file: {include_file}")
        if not include_file:
            raise ValueError("Include element missing required 'file' attribute")

        include_path = os.path.join(os.path.dirname(filename), include_file)

        # Process the included file recursively
        included_tree = _process_includes(include_path, processed_files)
        included_root = included_tree.getroot()

        # Insert all child elements from included file
        for child in included_root:
            # Insert before the include element
            include.addprevious(child)

        # Remove the include element
        include.getparent().remove(include)

    return tree


def _convert_mjcf(input_filename, output_filename, package_map, overwrite):
    # Process includes first to build complete DOM
    tree = _process_includes(input_filename)
    root = tree.getroot()

    # TODO(russt): Parse defaults, meshdir, and assetdir.
    has_meshdir = root.find(".//meshdir") is not None
    has_assetdir = root.find(".//assetdir") is not None

    # Find all <mesh> elements recursively using xpath
    mesh_elements = root.findall(".//mesh")
    for mesh_element in mesh_elements:
        mesh_filename = mesh_element.attrib["file"]
        if "class" in mesh_element.attrib:
            warnings.warn("Defaults are not being parsed yet.", UserWarning)
        if mesh_filename is None:
            raise ValueError("A 'file' attribute must be specified for mesh elements.")
        scale = mesh_element.attrib.get("scale")

        has_uniform_scale = True
        if scale:
            scale_suffix = "_scaled_" + scale.replace("-", "n").replace(" ", "_")
            scale = scale.split()
            has_uniform_scale = len(set(float(s) for s in scale)) == 1
        else:
            scale = None
            scale_suffix = ""

        if mesh_filename.lower().endswith(".obj") and has_uniform_scale:
            continue

        if mesh_filename.lower().startswith("package://"):
            input_mesh_path = package_map.ResolveUrl(mesh_filename)
        else:
            input_mesh_path = os.path.join(
                os.path.dirname(input_filename), mesh_filename
            )

        if not os.path.exists(input_mesh_path):
            if has_meshdir or has_assetdir:
                warnings.warn(
                    f"The file {input_mesh_path} does not exist in the expected location. This may be due to the fact that meshdir and assetdir are not being parsed yet.",
                    UserWarning,
                )
            raise FileNotFoundError(f"The file {input_mesh_path} does not exist.")

        output_obj_path = input_mesh_path.rsplit(".", 1)[0] + scale_suffix + ".obj"
        obj_filename = mesh_filename.rsplit(".", 1)[0] + scale_suffix + ".obj"

        _convert_mesh(
            input_mesh_path, output_obj_path, scale=scale, overwrite=overwrite
        )

        # Update the node's filename attribute
        mesh_element.attrib["filename"] = obj_filename

        # If scale is not None, update the scale attribute
        if scale:
            mesh_element.attrib["scale"] = "1 1 1"

    tree.write(output_filename, pretty_print=True)
    print(f"Converted MJCF file '{input_filename}' to '{output_filename}'.")


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

    if input_filename.lower().endswith(".stl") or input_filename.lower().endswith(
        ".dae"
    ):
        _convert_mesh(input_filename, output_filename, overwrite)
    elif input_filename.lower().endswith(".urdf"):
        _convert_urdf(input_filename, output_filename, package_map, overwrite=overwrite)
    elif input_filename.lower().endswith(".sdf"):
        _convert_sdf(input_filename, output_filename, package_map, overwrite=overwrite)
    elif input_filename.lower().endswith(".xml"):
        _convert_mjcf(input_filename, output_filename, package_map, overwrite=overwrite)
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
