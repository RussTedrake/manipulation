import argparse
import copy
import os
import re
import warnings
from typing import Tuple

import numpy as np
import trimesh
from lxml import etree
from PIL import Image  # For image loading
from pydrake.all import PackageMap, Quaternion, RigidTransform


def _convert_mesh(
    url: str,
    path: str,
    scale: np.ndarray | None = None,
    X_GM: np.ndarray = np.eye(4),
    material: trimesh.visual.material.Material | None = None,
    overwrite: bool = False,
) -> Tuple[str, str]:
    """Convert a mesh file to be compatible with Drake using Trimesh.

    Args:
        url: The URL of the mesh file.
        path: The path to the mesh file.
        scale: The (optional) scale to apply to the mesh.
        X_GM: The (optional) transform from the geometry frame to the mesh frame.
        texture_path: The (optional) path to the texture file to apply to the mesh.
        overwrite: Whether to overwrite existing files.

    Returns:
        A tuple containing the URL and path of the converted mesh file.
    """
    # Create a compact matrix string for filename
    if scale is not None:
        scale_str = "_scaled_" + "_".join(
            [str(int(s) if s.is_integer() else s).replace("-", "n") for s in scale]
        )
    else:
        scale_str = ""
    if not np.allclose(X_GM, np.eye(4)):
        R = X_GM[:3, :3]
        p = X_GM[:3, 3]

        # Check if rotation is around a single axis
        rot_str = ""
        if np.allclose(R, np.eye(3)):
            pass
        elif np.allclose(R, np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])):
            rot_str = "_Rx180"
        elif np.allclose(R, np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])):
            rot_str = "_Ry180"
        elif np.allclose(R, np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])):
            rot_str = "_Rz180"
        else:
            # Use a hash of the rotation matrix for other cases
            rot_str = f"_R{hash(R.tobytes()) % 1000:03d}"

        # Add translation if non-zero
        trans_str = ""
        if not np.allclose(p, 0):
            trans_str = f"_T{hash(p.tobytes()) % 1000:03d}"
    else:
        rot_str = ""
        trans_str = ""
    X_GM_str = scale_str + rot_str + trans_str
    suffix = "_from_" + path.rsplit(".", 1)[1] + X_GM_str + ".obj"

    output_url = url.rsplit(".", 1)[0] + suffix
    output_path = path.rsplit(".", 1)[0] + suffix

    # Generate base name for material
    mtl_name = os.path.splitext(os.path.basename(output_path))[0] + ".mtl"

    if not overwrite and os.path.exists(output_path):
        return output_url, output_path

    # Load the mesh
    try:
        mesh_or_scene = trimesh.load(path)
    except Exception as e:
        raise ValueError(f"Failed to load mesh {path}:\n{e}")
    if isinstance(mesh_or_scene, trimesh.Scene):
        meshes = [mesh for mesh in mesh_or_scene.geometry.values()]
    else:
        meshes = [mesh_or_scene]

    for mesh in meshes:
        # Apply scaling if specified
        if not np.allclose(X_GM, np.eye(4)) or scale is not None:
            if scale is not None:
                if len(scale) != 3:
                    raise ValueError("Scale must be a 3-element array.")
                scale.append(1)
                X_GM = np.diag(scale) @ X_GM
            mesh.apply_transform(X_GM)

        if material is not None:
            # For STL files, we need to ensure UV coordinates exist
            if not hasattr(mesh.visual, "uv"):
                # Generate simple planar UV coordinates based on normalized vertex positions
                vertices = mesh.vertices - mesh.vertices.min(axis=0)
                vertices = vertices / vertices.max(axis=0)
                uv = vertices[:, [0, 1]]  # Use X and Y coordinates for UV mapping

                mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
            else:
                mesh.visual.material = material

    # Export as OBJ with specified material filename
    mesh_or_scene.export(output_path, include_texture=True, mtl_name=mtl_name)

    # If we specified a texture_path, clean up the generated texture and update MTL
    if material is not None and material.image is not None:
        image_path = material.image.filename
        # Remove the generated texture if it exists
        generated_texture = os.path.join(os.path.dirname(output_path), "material_0.png")
        if os.path.exists(generated_texture) and generated_texture != image_path:
            os.remove(generated_texture)

        # Update the MTL file to point to our texture
        mtl_path = os.path.join(os.path.dirname(output_path), mtl_name)
        if os.path.exists(mtl_path):
            with open(mtl_path, "r") as f:
                mtl_content = f.read()
            mtl_content = mtl_content.replace(
                "material_0.png", os.path.basename(image_path)
            )
            with open(mtl_path, "w") as f:
                f.write(mtl_content)

    print(f"Converted {path} to {output_path}")
    return output_url, output_path


def _convert_urdf(
    input_filename: str, output_filename: str, package_map: PackageMap, overwrite: bool
) -> None:
    """Convert an URDF file to be compatible with Drake.

    Args:
        input_filename: The path to the input URDF file.
        output_filename: The path where the converted URDF file will be saved.
        package_map: The PackageMap to use.
        overwrite: Whether to overwrite existing files.
    """
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
            if len(set(scale)) == 1 and all(s > 0 for s in scale):
                # Uniform positive scaling is supported natively by Drake.
                scale = None
            else:
                node.attrib["scale"] = "1 1 1"
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

        # Convert the node back to a string
        node_string = etree.tostring(node, encoding="unicode")

        # Replace the original match with the updated node string
        modified_urdf_content = modified_urdf_content.replace(node_text, node_string)

    with open(output_filename, "w") as file:
        file.write(modified_urdf_content)

    print(f"Converted URDF file '{input_filename}' to '{output_filename}'.")


def _convert_sdf(
    input_filename: str, output_filename: str, package_map: PackageMap, overwrite: bool
) -> None:
    """Convert an SDF file to be compatible with Drake.

    Args:
        input_filename: The path to the input SDF file.
        output_filename: The path where the converted SDF file will be saved.
        package_map: The PackageMap to use.
        overwrite: Whether to overwrite existing files.
    """
    with open(input_filename, "r") as file:
        sdf_content = file.read()

    # Remove XML comments to avoid matching filenames inside them
    sdf_content_no_comments = re.sub(r"<!--.*?-->", "", sdf_content, flags=re.DOTALL)

    # Parse the XML to properly handle SDF structure
    try:
        root = etree.fromstring(sdf_content_no_comments)
    except etree.XMLSyntaxError as e:
        raise ValueError(f"Invalid SDF XML syntax in {input_filename}: {e}")

    # Find all mesh elements with uri containing .stl, .dae, or .obj extensions
    mesh_elements = root.xpath(".//mesh[uri]")

    # Filter to only those with target file extensions
    target_extensions = (".stl", ".dae", ".obj")
    filtered_mesh_elements = []
    for mesh_element in mesh_elements:
        uri_element = mesh_element.find("uri")
        if uri_element is not None and uri_element.text:
            uri_text = uri_element.text.strip().lower()
            if any(uri_text.endswith(ext) for ext in target_extensions):
                filtered_mesh_elements.append(mesh_element)

    for mesh_element in filtered_mesh_elements:
        uri_element = mesh_element.find("uri")
        mesh_url = uri_element.text.strip()

        # Handle scale element
        scale_element = mesh_element.find("scale")
        scale = None
        if scale_element is not None:
            scale_values = scale_element.text.strip().split()
            if len(scale_values) == 3:
                scale = [float(s) for s in scale_values]
                if len(set(scale)) == 1 and all(s > 0 for s in scale):
                    # Uniform positive scaling is supported natively by Drake
                    scale = None
                else:
                    scale_element.text = "1 1 1"

        # Don't need to convert .obj files with no scale or uniform scale
        if mesh_url.lower().endswith(".obj") and scale is None:
            continue

        # Resolve the mesh path
        if mesh_url.lower().startswith("package://") or mesh_url.lower().startswith(
            "file://"
        ):
            mesh_path = package_map.ResolveUrl(mesh_url)
        else:
            mesh_path = os.path.join(os.path.dirname(input_filename), mesh_url)

        # Convert the mesh
        output_mesh_url, _ = _convert_mesh(
            url=mesh_url, path=mesh_path, scale=scale, overwrite=overwrite
        )

        # Update the URI element text
        uri_element.text = output_mesh_url

    # Convert back to string
    output_content = etree.tostring(root, encoding="unicode", pretty_print=True)

    with open(output_filename, "w") as file:
        file.write(output_content)

    print(f"Converted SDF file '{input_filename}' to '{output_filename}'.")


def _process_includes(
    filename: str, processed_files: set[str] | None = None
) -> etree.ElementTree:
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
    tree = etree.parse(filename, parser=etree.XMLParser(remove_comments=True))
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


def _apply_defaults(
    element: etree.Element,
    defaults_dict: dict[tuple[str, str], etree.Element],
    class_name_override: str | None = None,
) -> etree.Element:
    """Apply default attributes from a defaults class to an element.

    Args:
        element: The XML element to apply defaults to
        defaults_dict: Dictionary mapping (class_name, element_type) to default elements
        class_name_override: If not None, this will override the class name for the element.
    Returns:
        The element with defaults applied.
    """
    # If class_name_override not already set, check parent elements for childclass
    if class_name_override is None and "class" not in element.attrib:
        parent = element.getparent()
        while parent is not None:
            if "childclass" in parent.attrib:
                class_name_override = parent.attrib["childclass"]
                break
            parent = parent.getparent()
    if class_name_override or "class" in element.attrib:
        element = copy.deepcopy(
            element
        )  # make a copy so we don't write the applied default back to the exported model
        class_name = (
            element.attrib["class"]
            if class_name_override is None
            else class_name_override
        )
        # Try to find defaults for this specific element type and class
        key = (class_name, element.tag)
        if key in defaults_dict:
            for attr, value in defaults_dict[key].attrib.items():
                # Special case: only one rotation specification allowed
                rotation_attrs = {"quat", "axisangle", "xyaxes", "zaxis", "euler"}
                if attr in rotation_attrs:
                    # Check if element already has any rotation attributes
                    has_rotation = any(
                        rot_attr in element.attrib for rot_attr in rotation_attrs
                    )
                    if has_rotation:
                        continue
                if attr != "class":
                    if attr not in element.attrib:
                        element.attrib[attr] = value
    return element


def _convert_mjcf(
    input_filename: str,
    output_filename: str,
    overwrite: bool,
    remap_geometry_groups: dict[int, int] = {},
) -> None:
    """Convert an MJCF file to be compatible with Drake.

    Args:
        input_filename: The path to the input MJCF file.
        output_filename: The path where the converted MJCF file will be saved.
        package_map: The PackageMap to use.
        overwrite: Whether to overwrite existing files.
    """
    # Check if output path matches input path
    if os.path.dirname(os.path.abspath(output_filename)) != os.path.dirname(
        os.path.abspath(input_filename)
    ):
        warnings.warn(
            f"Output path {os.path.dirname(output_filename)} differs from input path "
            f"{os.path.dirname(input_filename)}. "
            "This may cause issues with relative paths in the MJCF file."
        )

    # Process includes first to build complete DOM
    tree = _process_includes(input_filename)
    root = tree.getroot()

    compiler_elements = root.findall(".//compiler")
    strippath = False
    meshdir = None
    texturedir = None
    for compiler_element in compiler_elements:
        if "strippath" in compiler_element.attrib:
            # Convert XML string "true"/"false" to Python bool
            strippath = compiler_element.attrib["strippath"].lower() == "true"
        if "assetdir" in compiler_element.attrib:
            meshdir = compiler_element.attrib["assetdir"]
            texturedir = meshdir
        if "meshdir" in compiler_element.attrib:
            meshdir = compiler_element.attrib["meshdir"]
        if "texturedir" in compiler_element.attrib:
            texturedir = compiler_element.attrib["texturedir"]
            del compiler_element.attrib["texturedir"]

    # Truncate all rgba attributes to [0, 1]. See Drake#22445.
    elements_with_rgba = root.findall(".//*[@rgba]")
    for element in elements_with_rgba:
        rgba = element.attrib["rgba"]
        rgba = [float(value) for value in rgba.split()]
        rgba = [min(max(value, 0), 1) for value in rgba]
        element.attrib["rgba"] = " ".join([str(value) for value in rgba])

    defaults = {}

    # Process all default elements recursively
    def process_defaults(element, parent_class="main"):
        """Process default elements recursively.

        Args:
            element: The default element to process
            parent_class: The parent class name for inheritance
        """
        # Get class name, default to parent_class if not specified
        class_name = element.get("class", parent_class)

        # Process this default element's children (which define defaults for specific
        # types)
        for child in element:
            if child.tag != "default":  # Only process non-default children
                # Create key from class name and element type
                key = (class_name, child.tag)

                # If parent class has defaults for this type, inherit them
                parent_key = (parent_class, child.tag)
                if parent_key in defaults:
                    defaults[key] = copy.deepcopy(defaults[parent_key])
                else:
                    # Create new default element
                    try:
                        defaults[key] = etree.Element(child.tag)
                    except:
                        raise RuntimeError(
                            f"Failed to create default element for {key}"
                        )

                # Update with new attributes
                for attr, value in child.attrib.items():
                    if attr != "class":
                        defaults[key].attrib[attr] = value

        # Process nested default elements recursively
        for child in element.findall("default"):
            process_defaults(child, class_name)

    # Find and process all top-level default elements
    for default_element in root.findall("default"):
        process_defaults(default_element)

    # Parse all textures
    texture_paths = {}
    texture_elements = root.findall(".//asset/texture")
    for texture_element in texture_elements:
        if "file" in texture_element.attrib:
            texture_url = texture_element.attrib["file"]
            if "name" in texture_element.attrib:
                texture_name = texture_element.attrib["name"]
            else:
                texture_name = os.path.splitext(os.path.basename(texture_url))[0]
            if strippath:
                # Remove all path information, keeping only filename
                texture_url = os.path.basename(texture_url)

            # Check if texture_url is already an absolute path
            if os.path.isabs(texture_url):
                texture_path = texture_url
            # Check if texturedir is an absolute path
            elif texturedir is not None and os.path.isabs(texturedir):
                texture_path = os.path.join(texturedir, texture_url)
            # Use path relative to MJCF file
            else:
                base_path = os.path.dirname(input_filename)
                if texturedir is not None:
                    base_path = os.path.join(base_path, texturedir)
                texture_path = os.path.join(base_path, texture_url)

            if not os.path.exists(texture_path):
                raise FileNotFoundError(f"The file {texture_path} does not exist.")

            texture_paths[texture_name] = texture_path
            texture_element.getparent().remove(texture_element)

    # Parse all materials that reference textures
    materials = {}
    for material_element in root.findall(".//asset/material"):
        if "name" not in material_element.attrib:
            raise AssertionError("Material element must have a 'name' attribute")
        # Create a SimpleMaterial with the specified properties
        props = {  # default values from the MujoCo XML documentation.
            "diffuse": [255, 255, 255, 255],
            "ambient": [255, 255, 255, 255],
            "specular": [0.5, 0.5, 0.5],
            "glossiness": 0.5,
        }
        if "rgba" in material_element.attrib:
            props["diffuse"] = [
                255 * float(value) for value in material_element.attrib["rgba"].split()
            ]
            props["ambient"] = props["diffuse"]  # Match ambient to diffuse
        if "specular" in material_element.attrib:
            props["specular"] = [float(material_element.attrib["specular"])] * 3
        if "shininess" in material_element.attrib:
            props["glossiness"] = float(material_element.attrib["shininess"])
        # Note: emission, metallic, and roughness aren't directly supported by
        # SimpleMaterial so we'll skip those properties
        if "texture" in material_element.attrib:
            texture_name = material_element.attrib["texture"]
            if texture_name in texture_paths:
                texture_path = texture_paths[texture_name]
                if not os.path.exists(texture_path):
                    raise FileNotFoundError(f"Texture file not found: {texture_path}")
                image = Image.open(texture_path)
                props["image"] = image
        materials[material_element.attrib["name"]] = (
            trimesh.visual.material.SimpleMaterial(**props)
        )

    mesh_to_material = {}
    # Loop through geoms to find the mesh => material mappings
    for geom_element in root.findall(".//geom"):
        geom_element_w_defaults = _apply_defaults(geom_element, defaults)
        if (
            "material" in geom_element_w_defaults.attrib
            and geom_element_w_defaults.attrib["material"] in materials
            and "mesh" in geom_element_w_defaults.attrib
        ):
            mesh_name = geom_element_w_defaults.attrib["mesh"]
            material_name = geom_element_w_defaults.attrib["material"]
            if (
                mesh_name in mesh_to_material
                and mesh_to_material[mesh_name] != material_name
            ):
                raise AssertionError(
                    f"Mesh {mesh_name} was already associated with "
                    f"{mesh_to_material[mesh_name]}. We don't handle multiple "
                    "materials assigned to the same mesh yet."
                )
            mesh_to_material[mesh_name] = material_name

    # Find all <mesh> elements recursively using xpath
    mesh_elements = root.findall(".//asset/mesh")
    for mesh_element in mesh_elements:
        mesh_element_w_defaults = _apply_defaults(mesh_element, defaults)

        mesh_url = mesh_element_w_defaults.attrib["file"]
        if mesh_url is None:
            raise ValueError("A 'file' attribute must be specified for mesh elements.")
        if "name" in mesh_element_w_defaults.attrib:
            mesh_name = mesh_element_w_defaults.attrib["name"]
        else:
            mesh_name = os.path.splitext(os.path.basename(mesh_url))[0]
            mesh_element_w_defaults.attrib["name"] = mesh_name
            mesh_element.attrib["name"] = mesh_name

        X_MG = RigidTransform()
        if "refpos" in mesh_element_w_defaults.attrib:
            refpos = [
                float(value)
                for value in mesh_element_w_defaults.attrib["refpos"].split()
            ]
            X_MG.set_translation(-refpos)
            mesh_element.attrib["refpos"] = "0 0 0"
        if "refquat" in mesh_element_w_defaults.attrib:
            refquat = np.array(
                [
                    float(value)
                    for value in mesh_element_w_defaults.attrib["refquat"].split()
                ]
            )
            refquat = refquat / np.linalg.norm(refquat)
            X_MG.set_rotation(
                Quaternion(refquat[0], refquat[1], refquat[2], refquat[3])
            )
            mesh_element.attrib["refquat"] = "1 0 0 0"
        X_GM = X_MG.inverse().GetAsMatrix4()

        scale = mesh_element_w_defaults.attrib.get("scale")
        if scale:
            scale = [float(s) for s in scale.split()]
            if len(set(scale)) == 1 and all(s > 0 for s in scale):
                # Uniform positive scaling is supported natively by Drake.
                scale = None
            else:
                mesh_element.attrib["scale"] = "1 1 1"

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
            raise FileNotFoundError(f"The file {mesh_path} does not exist.")

        material = None
        if mesh_name in mesh_to_material:
            material = materials[mesh_to_material[mesh_name]]

        if mesh_url.lower().endswith(".obj") and scale is None and material is None:
            # Don't need to convert .obj files with no scale or uniform scale.
            continue

        output_mesh_url, output_mesh_path = _convert_mesh(
            url=mesh_url,
            path=mesh_path,
            scale=scale,
            X_GM=X_GM,
            material=material,
            overwrite=overwrite,
        )

        # Update the node's filename attribute
        mesh_element.attrib["file"] = output_mesh_url

    geoms_with_type_plane = root.findall(".//geom[@type='plane']")
    for geom in geoms_with_type_plane:
        parent = geom.getparent()
        while parent is not None:
            if parent.tag == "worldbody":
                break
            elif parent.tag == "body":
                # Then the plane would have been parsed as a dynamic collision element
                # in Drake. Replace it with a large box.
                geom.attrib["size"] = "1000 1000 1"
                if "pos" in geom.attrib:
                    pos = [float(value) for value in geom.attrib["pos"].split()]
                else:
                    pos = [0, 0, 0]
                pos[2] -= 1
                geom.attrib["pos"] = " ".join([str(value) for value in pos])
                geom.attrib["type"] = "box"
                break
            parent = parent.getparent()

    # Remap geometry groups
    if remap_geometry_groups:
        geoms = root.findall(".//geom")
        for geom in geoms:
            if "group" in geom.attrib:
                group = int(geom.attrib["group"])
                if group in remap_geometry_groups:
                    geom.attrib["group"] = str(remap_geometry_groups[group])

    tree.write(output_filename, pretty_print=True)
    print(f"Converted MJCF file {input_filename} to {output_filename}")


def MakeDrakeCompatibleModel(
    input_filename: str,
    output_filename: str,
    package_map: PackageMap = PackageMap(),
    overwrite: bool = False,
    remap_mujoco_geometry_groups: dict[int, int] = {},
) -> None:
    """Converts a model file (currently .urdf, .sdf, or .xml) to be compatible with the
    Drake multibody parsers.

    For all models:

    - Converts any .stl files to obj
      https://github.com/RobotLocomotion/drake/issues/19408
    - Converts any .dae files to obj
      https://github.com/RobotLocomotion/drake/issues/19109
    - Resizes meshes to work around any non-uniform scale attributes
      https://github.com/RobotLocomotion/drake/issues/22046

    In addition, for MuJoCo .xml models:

    - Converts dynamic half-space collision geometries to (very) large boxes
      https://github.com/RobotLocomotion/drake/issues/19263
    - Truncates all rgba attributes to [0, 1].
      https://github.com/RobotLocomotion/drake/issues/22445
    - Applies refpos and refquat attributes to the mesh .obj files.
      https://github.com/RobotLocomotion/drake/issues/22488
    - Applies materials specified in mujoco .xml directly to the mesh .obj files.

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
        remap_mujoco_geometry_groups (dict[int, int], optional): Drake's mujoco
            parser registers visual geometry for geometry groups < 3 (the
            mujoco default), which is a common, but not universal, convention.
            This argument allows you to remap (substituting the value for the
            key).
    """
    if input_filename.lower().endswith(".urdf"):
        _convert_urdf(input_filename, output_filename, package_map, overwrite=overwrite)
    elif input_filename.lower().endswith(".sdf"):
        _convert_sdf(input_filename, output_filename, package_map, overwrite=overwrite)
    elif input_filename.lower().endswith(".xml"):
        _convert_mjcf(
            input_filename,
            output_filename,
            overwrite=overwrite,
            remap_geometry_groups=remap_mujoco_geometry_groups,
        )
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
