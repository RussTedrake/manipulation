"""
A script for creating a Drake-compatible SDFormat file for a triangle mesh.

Credits: Part of this code has been adopted from Greg Izatt
(https://github.com/gizatt/convex_decomp_to_sdf).
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
from lxml import etree as ET

try:
    import trimesh
except ImportError:
    print("trimesh not found.")
    print("Consider 'pip install trimesh'.")
    exit(code=1)


def _calc_mesh_com_and_inertia(
    mesh: trimesh.Trimesh,
    mass: float,
    frame: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculates the mesh's center of mass and moment of inertia by assuming a uniform
    mass density.

    Args:
        mesh (trimesh.Trimesh): The mesh.
        mass (float): The mass in kg of the object.
        frame (np.ndarray, optional): The frame that the moment of inertia should be
        expressed in. If None, the center of mass is used. This is a (4,4) homogenous
        transformation matrix.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple of (center_of_mass, moment_of_inertia):
            center_of_mass: The center of mass of shape (3,).
            moment_of_inertia: The moment of inertia of shape (3,3).
    """
    if not mesh.is_watertight:
        logging.warning(
            "The mesh is not watertight. This might lead to incorrect center of mass "
            + "and moment of inertia computations. However, in practice the values are "
            + "often reasonable regardless."
        )

    volume = mesh.volume
    mesh.density = mass / volume
    moment_of_inertia = (
        mesh.moment_inertia if frame is None else mesh.moment_inertia_frame(frame)
    )
    if frame is None:
        logging.info(f"Calculating the moment of inertia about {mesh.center_mass}")
    return mesh.center_mass, moment_of_inertia


def _perform_convex_decomposition(
    mesh: trimesh.Trimesh,
    mesh_name: str,
    mesh_dir: Path,
    preview_with_trimesh: bool,
    use_coacd: bool = False,
    coacd_kwargs: dict | None = None,
    vhacd_kwargs: dict | None = None,
) -> List[Path]:
    """Given a mesh, performs a convex decomposition of it with either VHACD or CoACD.
    The resulting convex parts are saved in a subfolder named `<mesh_filename>_parts`.

    Args:
        mesh (trimesh.Trimesh): The mesh to decompose.
        mesh_name (str): The name of the mesh. It is used for naming the mesh parts
        directory.
        mesh_dir (Path): The path to the directory that the mesh is stored in. This is
        used for creating the mesh parts directory.
        preview_with_trimesh (bool): Whether to open (and block on) a window to preview
        the decomposition.
        use_coacd (bool): Whether to use CoACD instead of VHACD for decomposition.
        coacd_kwargs (dict | None): The CoACD-specific parameters.
        vhacd_kwargs (dict | None): The VHACD-specific parameters.

    Returns:
        List[Path]: The paths of the convex pieces.
    """
    # Create a subdir for the convex parts
    out_dir = mesh_dir / (mesh_name + "_parts")
    os.makedirs(out_dir, exist_ok=True)

    if preview_with_trimesh:
        logging.info(
            "Showing mesh before convex decomposition. Close the window to proceed."
        )
        mesh.show()

    logging.info(
        "Performing convex decomposition. This might take a couple of minutes for "
        + "complicated meshes and fine resolution settings."
    )
    try:
        if use_coacd:
            try:
                import coacd
            except ImportError:
                print("coacd not found.")
                print("Consider 'pip install coacd'.")
                exit(code=1)

            coacd.set_log_level("error")
            coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
            coacd_result = coacd.run_coacd(coacd_mesh, **(coacd_kwargs or {}))
            # Convert CoACD result to trimesh objects.
            convex_pieces = []
            for vertices, faces in coacd_result:
                piece = trimesh.Trimesh(vertices, faces)
                convex_pieces.append(piece)
        else:
            try:
                import vhacdx  # noqa: F401
            except ImportError:
                print("vhacdx not found.")
                print("Consider 'pip install vhacdx'.")
                exit(code=1)

            vhacd_settings = vhacd_kwargs or {}
            convex_pieces = mesh.convex_decomposition(**vhacd_settings)
            if not isinstance(convex_pieces, list):
                convex_pieces = [convex_pieces]
    except Exception as e:
        logging.error(f"Problem performing decomposition: {e}")
        exit(1)

    if preview_with_trimesh:
        # Display the convex decomposition, giving each a random colors
        for part in convex_pieces:
            this_color = trimesh.visual.random_color()
            part.visual.face_colors[:] = this_color
        scene = trimesh.scene.scene.Scene()
        for part in convex_pieces:
            scene.add_geometry(part)

        logging.info(
            f"Showing the mesh convex decomposition into {len(convex_pieces)} parts. "
            + "Close the window to proceed."
        )
        scene.show()

    convex_piece_paths: List[Path] = []
    for i, part in enumerate(convex_pieces):
        piece_name = f"convex_piece_{i:03d}.obj"
        path = out_dir / piece_name
        part.export(path)
        convex_piece_paths.append(path)

    return convex_piece_paths


def create_sdf_from_mesh(
    mesh_path: Path,
    mass: float,
    scale: float,
    is_compliant: bool,
    hydroelastic_modulus: float,
    hunt_crossley_dissipation: Union[float, None],
    mu_dynamic: Union[float, None],
    mu_static: Union[float, None],
    preview_with_trimesh: bool,
    use_coacd: bool = False,
    coacd_kwargs: dict | None = None,
    vhacd_kwargs: dict | None = None,
) -> None:
    """Given a mesh, creates an SDFormat file in the same directory that:
    - Uses the mesh as its visual geometry
    - Performs a convex decomposition of the mesh, and uses those pieces
    as the collision geometry
    - Inserts the center of mass and moment of inertia for the object, calculated from
    the original mesh and a uniform density assumption. The density is computed from
    the mesh volume and the given mass

    Args:
        mesh_path (Path): The path to the mesh.
        mass (float): The mass in kg of the mesh.
        scale (float): Scale factor to convert the specified mesh's coordinates to
        meters.
        is_compliant (bool): Whether the SDFormat file will be used for compliant
        Hydroelastic simulations. The object will behave as rigid Hydroelastic if this
        is not specified.
        hydroelastic_modulus (float): The Hydroelastic Modulus. This is only used if
        `is_compliant` is True. The default value leads to low compliance. See
        https://drake.mit.edu/doxygen_cxx/group__hydroelastic__user__guide.html for how
        to pick a value.
        hunt_crossley_dissipation (Union[float, None]): The optional Hydroelastic
        Hunt-Crossley dissipation (s/m). See
        https://drake.mit.edu/doxygen_cxx/group__hydroelastic__user__guide.html for how
        to pick a value.
        mu_dynamic (Union[float, None]): The coefficient of dynamic friction.
        mu_static (Union[float, None]): The coefficient of static friction.
        preview_with_trimesh (bool): Whether to open (and block on) a window to preview
        the decomposition.
        use_coacd (bool): Whether to use CoACD instead of VHACD for convex decomposition.
        coacd_kwargs (dict | None): The CoACD-specific parameters.
        vhacd_kwargs (dict | None): The VHACD-specific parameters.
    """
    if (use_coacd and vhacd_kwargs is not None) or (
        not use_coacd and coacd_kwargs is not None
    ):
        raise ValueError("Cannot use both CoACD and VHACD.")

    # Construct SDF path
    dir_path = mesh_path.parent
    mesh_name = mesh_path.stem
    sdf_path = dir_path / (mesh_name + ".sdf")

    # Load and scale mesh
    mesh = trimesh.load(mesh_path, skip_materials=True, force="mesh")
    mesh.apply_scale(scale)

    # Generate the SDFormat headers
    root_item = ET.Element("sdf", version="1.7", nsmap={"drake": "drake.mit.edu"})
    model_item = ET.SubElement(root_item, "model", name=mesh_name)
    link_item = ET.SubElement(model_item, "link", name=f"{mesh_name}_body_link")
    pose_item = ET.SubElement(link_item, "pose")
    pose_item.text = "0 0 0 0 0 0"

    # Compute and add the physical properties
    com, inertia = _calc_mesh_com_and_inertia(mesh=mesh, mass=mass)
    inertial_item = ET.SubElement(link_item, "inertial")
    mass_item = ET.SubElement(inertial_item, "mass")
    mass_item.text = str(mass)
    com_item = ET.SubElement(inertial_item, "pose")
    com_item.text = f"{com[0]:.5f} {com[1]:.5f} {com[2]:.5f} 0 0 0"
    inertia_item = ET.SubElement(inertial_item, "inertia")
    for i in range(3):
        for j in range(i, 3):
            item = ET.SubElement(inertia_item, "i" + "xyz"[i] + "xyz"[j])
            item.text = f"{inertia[i, j]:.5e}"

    # Add the original mesh as the visual mesh
    visual_mesh_path = mesh_path.relative_to(dir_path)
    visual_item = ET.SubElement(link_item, "visual", name="visual")
    geometry_item = ET.SubElement(visual_item, "geometry")
    mesh_item = ET.SubElement(geometry_item, "mesh")
    uri_item = ET.SubElement(mesh_item, "uri")
    uri_item.text = visual_mesh_path.as_posix()
    scale_item = ET.SubElement(mesh_item, "scale")
    scale_item.text = f"{scale} {scale} {scale}"

    # Compute the VHACD convex decomposition and use it as the collision geometry
    mesh_piece_paths = _perform_convex_decomposition(
        mesh=mesh,
        mesh_name=mesh_name,
        mesh_dir=dir_path,
        preview_with_trimesh=preview_with_trimesh,
        use_coacd=use_coacd,
        coacd_kwargs=coacd_kwargs,
        vhacd_kwargs=vhacd_kwargs,
    )
    for i, mesh_piece_path in enumerate(mesh_piece_paths):
        mesh_piece_path = mesh_piece_path.relative_to(dir_path)
        collision_item = ET.SubElement(
            link_item, "collision", name=f"collision_{i:03d}"
        )
        geometry_item = ET.SubElement(collision_item, "geometry")
        mesh_item = ET.SubElement(geometry_item, "mesh")
        uri_item = ET.SubElement(mesh_item, "uri")
        uri_item.text = mesh_piece_path.as_posix()
        ET.SubElement(mesh_item, "{drake.mit.edu}declare_convex")

        # Add proximity properties
        proximity_item = ET.SubElement(
            collision_item, "{drake.mit.edu}proximity_properties"
        )
        if is_compliant:
            ET.SubElement(proximity_item, "{drake.mit.edu}compliant_hydroelastic")
            hydroelastic_moulus_item = ET.SubElement(
                proximity_item, "{drake.mit.edu}hydroelastic_modulus"
            )
            hydroelastic_moulus_item.text = f"{hydroelastic_modulus:.3e}"
        else:
            ET.SubElement(proximity_item, "{drake.mit.edu}rigid_hydroelastic")
        if hunt_crossley_dissipation is not None:
            hunt_crossley_dissipation_item = ET.SubElement(
                proximity_item, "{drake.mit.edu}hunt_crossley_dissipation"
            )
            hunt_crossley_dissipation_item.text = f"{hunt_crossley_dissipation:.3f}"
        if mu_dynamic is not None:
            mu_dynamic_item = ET.SubElement(proximity_item, "{drake.mit.edu}mu_dynamic")
            mu_dynamic_item.text = f"{mu_dynamic:.3f}"
        if mu_static is not None:
            mu_static_item = ET.SubElement(proximity_item, "{drake.mit.edu}mu_static")
            mu_static_item.text = f"{mu_static:.3f}"

    logging.info(f"Writing SDF to {sdf_path}")
    ET.ElementTree(root_item).write(sdf_path, pretty_print=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a Drake-compatible SDFormat file for a triangle mesh."
    )
    parser.add_argument(
        "--mesh",
        type=str,
        required=True,
        help="Path to mesh file.",
    )
    parser.add_argument(
        "--mass",
        type=float,
        required=True,
        help="The mass in kg of the object that is represented by the mesh. This is "
        + "used for computing the moment of inertia.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale factor to convert the specified mesh's coordinates to meters.",
    )
    parser.add_argument(
        "--compliant",
        action="store_true",
        help="Whether the SDFormat file will be used for compliant Hydroelastic "
        + "simulations. The object will behave as rigid Hydroelastic if this is not "
        + "specified.",
    )
    parser.add_argument(
        "--hydroelastic_modulus",
        type=float,
        default=1.0e8,
        help="The Hydroelastic Modulus. This is only used if --compliant is specified. "
        + "The default value leads to low compliance. See "
        + "https://drake.mit.edu/doxygen_cxx/group__hydroelastic__user__guide.html for "
        + "how to pick a value.",
    )
    parser.add_argument(
        "--hunt_crossley_dissipation",
        type=float,
        default=None,
        help="The Hydroelastic Hunt-Crossley dissipation (s/m). See "
        + "https://drake.mit.edu/doxygen_cxx/group__hydroelastic__user__guide.html for "
        + "how to pick a value.",
    )
    parser.add_argument(
        "--mu_dynamic",
        type=float,
        default=None,
        help="The coefficient of dynamic friction.",
    )
    parser.add_argument(
        "--mu_static",
        type=float,
        default=None,
        help="The coefficient of static friction.",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Whether to preview the decomposition.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Log level.",
    )

    # Create argument groups for VHACD and CoACD.
    vhacd_group = parser.add_argument_group("VHACD parameters")
    coacd_group = parser.add_argument_group("CoACD parameters")

    parser.add_argument(
        "--use_coacd",
        action="store_true",
        help="Use CoACD instead of VHACD for convex decomposition.",
    )

    # CoACD arguments.
    coacd_group.add_argument(
        "--threshold",
        type=float,
        help="CoACD threshold parameter for determining concavity.",
    )
    coacd_group.add_argument(
        "--preprocess_resolution",
        type=int,
        help="Resolution used in preprocessing step.",
    )
    coacd_group.add_argument(
        "--coacd_resolution",
        type=int,
        help="Main resolution parameter for decomposition.",
    )
    coacd_group.add_argument(
        "--mcts_nodes",
        type=int,
        help="Number of nodes for Monte Carlo Tree Search.",
    )
    coacd_group.add_argument(
        "--mcts_iterations",
        type=int,
        help="Number of iterations for Monte Carlo Tree Search.",
    )
    coacd_group.add_argument(
        "--mcts_max_depth",
        type=int,
        help="Maximum depth for Monte Carlo Tree Search.",
    )
    coacd_group.add_argument(
        "--preprocess_mode",
        type=str,
        default="auto",
        choices=["auto", "voxel", "sampling"],
        help="CoACD preprocess mode.",
    )
    coacd_group.add_argument(
        "--pca", action="store_true", help="Enable PCA pre-processing."
    )

    # VHACD arguments.
    vhacd_group.add_argument(
        "--vhacd_resolution",
        type=int,
        default=10000000,
        help="VHACD voxel resolution.",
    )
    vhacd_group.add_argument(
        "--maxConvexHulls",
        type=int,
        default=64,
        help="VHACD maximum number of convex hulls/ mesh pieces.",
    )
    vhacd_group.add_argument(
        "--minimumVolumePercentErrorAllowed",
        type=float,
        default=1.0,
        help="VHACD minimum allowed volume percentage error.",
    )
    vhacd_group.add_argument(
        "--maxRecursionDepth",
        type=int,
        default=10,
        help="VHACD maximum recursion depth.",
    )
    vhacd_group.add_argument(
        "--no_shrinkWrap",
        action="store_true",
        help="Whether or not to shrinkwrap the voxel positions to the source mesh on "
        + "output.",
    )
    vhacd_group.add_argument(
        "--fillMode",
        type=str,
        default="flood",
        choices=["flood", "raycast", "surface"],
        help="VHACD maximum recursion depth.",
    )
    vhacd_group.add_argument(
        "--maxNumVerticesPerCH",
        type=int,
        default=64,
        help="VHACD maximum number of triangles per convex hull.",
    )
    vhacd_group.add_argument(
        "--no_asyncACD",
        action="store_true",
        help="Whether or not to run VHACD asynchronously, taking advantage of "
        + "additional cores.",
    )
    vhacd_group.add_argument(
        "--minEdgeLength",
        type=int,
        default=2,
        help="VHACD minimum voxel patch edge length.",
    )

    args = parser.parse_args()
    logging.basicConfig(level=args.log_level)

    mesh_path = Path(os.path.abspath(args.mesh))
    if not mesh_path.exists():
        logging.error(f"No mesh found at {mesh_path}!")
        sys.exit(1)

    mass = args.mass
    if mass <= 0:
        logging.error(f"Got a mass of {mass} kg. Mass must be positive!")
        sys.exit(1)

    is_compliant = args.compliant
    hydroelastic_modulus = args.hydroelastic_modulus
    if is_compliant and hydroelastic_modulus is not None and hydroelastic_modulus < 0:
        logging.error(
            f"Got a Hydroelastic modulus of {hydroelastic_modulus:.3e} Pa. A "
            + "non-negative value is required!"
        )
        sys.exit(1)

    hunt_crossley_dissipation = args.hunt_crossley_dissipation
    if hunt_crossley_dissipation is not None and hunt_crossley_dissipation < 0:
        logging.error(
            f"Got a Hunt-Crossley dissipation of {hunt_crossley_dissipation:.3f} s/m. A "
            + "non-negative value is required!"
        )
        sys.exit(1)

    mu_dynamic = args.mu_dynamic
    if mu_dynamic is not None and mu_dynamic < 0:
        logging.error(
            f"Got a coefficient of dynamic friction of {mu_dynamic:.3f}. A "
            + "non-negative value is required!"
        )
        sys.exit(1)

    mu_static = args.mu_static
    if mu_static is not None and mu_static < 0:
        logging.error(
            f"Got a coefficient of static friction of {mu_static:.3f}. A "
            + "non-negative value is required!"
        )
        sys.exit(1)

    # Separate VHACD and CoACD parameters.
    vhacd_params = (
        {
            "resolution": args.vhacd_resolution,
            "maxConvexHulls": args.maxConvexHulls,
            "minimumVolumePercentErrorAllowed": args.minimumVolumePercentErrorAllowed,
            "maxRecursionDepth": args.maxRecursionDepth,
            "shrinkWrap": not args.no_shrinkWrap,
            "fillMode": args.fillMode,
            "maxNumVerticesPerCH": args.maxNumVerticesPerCH,
            "asyncACD": not args.no_asyncACD,
            "minEdgeLength": args.minEdgeLength,
        }
        if not args.use_coacd
        else None
    )
    coacd_params = {}
    for param in [
        "threshold",
        "preprocess_resolution",
        "coacd_resolution",
        "mcts_nodes",
        "mcts_iterations",
        "mcts_max_depth",
        "preprocess_mode",
    ]:
        value = getattr(args, param)
        if value is not None:
            key = "resolution" if param == "coacd_resolution" else param
            coacd_params[key] = value

    create_sdf_from_mesh(
        mesh_path=mesh_path,
        mass=mass,
        scale=args.scale,
        is_compliant=is_compliant,
        hydroelastic_modulus=hydroelastic_modulus,
        hunt_crossley_dissipation=hunt_crossley_dissipation,
        mu_dynamic=mu_dynamic,
        mu_static=mu_static,
        preview_with_trimesh=args.preview,
        use_coacd=args.use_coacd,
        coacd_kwargs=coacd_params,
        vhacd_kwargs=vhacd_params,
    )
