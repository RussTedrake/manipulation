import argparse
import os
import re
import warnings
from typing import List, Optional

import pymeshlab
from lxml import etree
from pydrake.all import PackageMap


def _convert_mesh(
    url: str, path: str, scale: Optional[List[float]] = None, overwrite: bool = False
) -> str:
    if scale:
        scale_str = "_scaled_" + "_".join(
            [str(int(s) if s.is_integer() else s).replace("-", "n") for s in scale]
        )
    else:
        scale_str = ""
    suffix = "_from_" + path.rsplit(".", 1)[1] + scale_str + ".obj"

    output_url = url.rsplit(".", 1)[0] + suffix
    output_path = path.rsplit(".", 1)[0] + suffix

    if not overwrite and os.path.exists(output_path):
        print(f"Note: {output_path} already exists. Skipping conversion.")
        return output_url, output_path

    # Create a new PyMeshLab mesh set
    ms = pymeshlab.MeshSet()

    # Load the mesh file
    ms.load_new_mesh(path)

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
    ms.save_current_mesh(output_path, save_face_color=False, save_vertex_color=False)
    print(f"Converted {path} to {output_path}")
    return output_url, output_path


def _convert_urdf(input_filename, output_filename, package_map, overwrite):
    with open(input_filename, "r") as file:
        urdf_content = file.read()

    # Remove XML comments to avoid matching filenames inside them
    urdf_content_no_comments = re.sub(r"<!--.*?-->", "", urdf_content, flags=re.DOTALL)

    # Regex pattern to find entire XML nodes with .stl or .dae extensions in the filename attribute
    resource_pattern = re.compile(
        r'<(\w+)\s+[^>]*filename=["\']([^"\']+\.(stl|dae|obj))["\'][^>]*>',
        re.IGNORECASE,
    )

    # Find all matches in the URDF content without comments
    matches = resource_pattern.finditer(urdf_content_no_comments)

    modified_urdf_content = urdf_content

    for match in matches:
        node_text = match.group(0)
        node = etree.fromstring(node_text)
        mesh_url = node.attrib["filename"]
        scale = node.attrib.get("scale")
        if scale:
            scale = [float(s) for s in scale.split()]
            if len(set(scale)) == 1:
                # Uniform scaling is supported natively by Drake.
                scale = None
        if mesh_url.lower().endswith(".obj") and scale is None:
            # Don't need to convert .obj files with no scale or uniform scale.
            continue

        if mesh_url.lower().startswith("package://") or mesh_url.lower().startswith(
            "file://"
        ):
            mesh_path = package_map.ResolveUrl(mesh_url)
        else:
            mesh_path = os.path.join(os.path.dirname(input_filename), mesh_url)

        output_mesh_url, output_mesh_path = _convert_mesh(
            url=mesh_url, path=mesh_path, scale=scale, overwrite=overwrite
        )

        # Update the node's filename attribute
        node.attrib["filename"] = output_mesh_url

        # If scale is not None, update the scale attribute
        if scale:
            node.attrib["scale"] = "1 1 1"

        # Convert the node back to a string
        node_string = etree.tostring(node, encoding="unicode")

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

    # TODO(russt): Parse defaults.
    compiler_elements = root.findall(".//compiler")
    strippath = False
    meshdir = None
    for compiler_element in compiler_elements:
        if "strippath" in compiler_element.attrib:
            # Convert XML string "true"/"false" to Python bool
            strippath = compiler_element.attrib["strippath"].lower() == "true"
        if "assetdir" in compiler_element.attrib:
            meshdir = compiler_element.attrib["assetdir"]
            compiler_element.attrib["assetdir"]
        if "meshdir" in compiler_element.attrib:
            meshdir = compiler_element.attrib["meshdir"]
        if "texturedir" in compiler_element.attrib:
            compiler_element.attrib["texturedir"]

    # Find all <mesh> elements recursively using xpath
    mesh_elements = root.findall(".//mesh")
    for mesh_element in mesh_elements:
        mesh_url = mesh_element.attrib["file"]
        if "class" in mesh_element.attrib:
            warnings.warn("Defaults are not being parsed yet.", UserWarning)
        if mesh_url is None:
            raise ValueError("A 'file' attribute must be specified for mesh elements.")
        scale = mesh_element.attrib.get("scale")
        if scale:
            scale = [float(s) for s in scale.split()]
            if len(set(scale)) == 1:
                # Uniform scaling is supported natively by Drake.
                scale = None
        if mesh_url.lower().endswith(".obj") and scale is None:
            # Don't need to convert .obj files with no scale or uniform scale.
            continue

        # Get absolute path to mesh file according to MJCF rules
        if strippath:
            # Remove all path information, keeping only filename
            mesh_url = os.path.basename(mesh_url)

        # Check if mesh_url is already an absolute path
        if os.path.isabs(mesh_url):
            mesh_path = mesh_url
        # Check if meshdir is an absolute path
        elif meshdir is not None and os.path.isabs(meshdir):
            mesh_path = os.path.join(meshdir, mesh_url)
        # Use path relative to MJCF file
        else:
            base_path = os.path.dirname(input_filename)
            if meshdir is not None:
                base_path = os.path.join(base_path, meshdir)
            mesh_path = os.path.join(base_path, mesh_url)

        if not os.path.exists(mesh_path):
            if meshdir is not None:
                warnings.warn(
                    f"The file {mesh_path} does not exist in the expected location. This may be due to the fact that meshdir and assetdir are not being parsed yet.",
                    UserWarning,
                )
            raise FileNotFoundError(f"The file {mesh_path} does not exist.")

        output_mesh_url, output_mesh_path = _convert_mesh(
            url=mesh_url, path=mesh_path, scale=scale, overwrite=overwrite
        )

        # Update the node's filename attribute
        mesh_element.attrib["file"] = output_mesh_url

        # If scale is not None, update the scale attribute
        if scale:
            mesh_element.attrib["scale"] = "1 1 1"

    tree.write(output_filename, pretty_print=True)
    print(f"Converted MJCF file '{input_filename}' to '{output_filename}'.")


def MakeDrakeCompatibleModel(
    input_filename: str,
    output_filename: str,
    package_map: PackageMap = None,
    overwrite: bool = False,
) -> None:
    """Converts a model file (currently .urdf or .xml)to be compatible with the
    Drake multibody parsers.

    - Converts any .stl files to obj
      https://github.com/RobotLocomotion/drake/issues/19408
    - Converts any .dae files to obj
      https://github.com/RobotLocomotion/drake/issues/19109
    - Zaps any non-uniform scale attributes
      https://github.com/RobotLocomotion/drake/issues/22046

    Any new files will be created alongside the original files (e.g. .obj files
    will be created next to the existing .stl files); all new files will get a
    descriptive suffix, and existing files will not be overwritten by default.

    Args:
        input_filename (str): The path to the input file to be converted.
        output_filename (str): The path where the converted file will be saved.
            Using the same string as input_filename is allowed.
        package_map (PackageMap, optional): The package map to use. Defaults to None.
        overwrite (bool, optional): Whether to overwrite existing files. Defaults
            to False.
    """
    if package_map is None:
        package_map = PackageMap()

    if input_filename.lower().endswith(".urdf"):
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

    package_map = PackageMap()

    if args.populate_package_map_from_env:
        package_map.PopulateFromEnvironment(args.populate_package_map_from_env)

    if args.populate_package_map_from_folder:
        package_map.PopulateFromFolder(args.populate_package_map_from_folder)

    MakeDrakeCompatibleModel(
        input_filename=args.input_filename,
        output_filename=args.output_filename,
        package_map=package_map,
        overwrite=args.overwrite,
    )
