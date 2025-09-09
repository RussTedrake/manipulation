import logging
from pathlib import Path

from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath

from manipulation.create_sdf_from_mesh import create_sdf_from_mesh

try:
    import trimesh

    trimesh_available = True
except ImportError:
    trimesh_available = False
    print("trimesh not found.")
    print("Consider 'pip install trimesh'.")

try:
    from shapely.affinity import scale as shapely_scale
    from shapely.geometry import MultiPolygon, Polygon

    shapely_available = True
except ImportError:
    shapely_available = False
    print("shapely not found.")
    print("Consider 'pip install shapely'.")


def create_sdf_asset_from_letter(
    text: str,
    font_name: str = "Arial",
    letter_height_meters: float = 0.4,
    extrusion_depth_meters: float = 0.2,
    mass: float = 1.0,
    is_compliant: bool = False,
    hydroelastic_modulus: float = 1e8,
    hunt_crossley_dissipation: float | None = None,
    mu_dynamic: float | None = 1.0,
    mu_static: float | None = None,
    output_dir: str = ".",
    use_bbox_collision_geometry: bool = False,
) -> Path | None:
    """
    Creates a complete SDF asset (mesh + SDF file) from a single letter.

    Args:
        text (str): The character to convert (single letter only).
        font_name (str): The name of the font to use (e.g., 'Arial', 'Times New Roman').
        letter_height_meters (float): The physical height of the letter in meters.
        extrusion_depth_meters (float): The physical depth/thickness of the letter in meters.
        mass (float): The mass in kg of the object for inertia calculations.
        is_compliant (bool): Whether to use compliant hydroelastic contact.
        hydroelastic_modulus (float): The hydroelastic modulus (Pa).
        hunt_crossley_dissipation (float | None): Hunt-Crossley dissipation parameter (s/m).
        mu_dynamic (float | None): Coefficient of dynamic friction.
        mu_static (float | None): Coefficient of static friction.
        output_dir (str): Directory where the output files will be saved.
        use_bbox_collision_geometry (bool): Whether to use axis-aligned bbox or coacd as collision geometry

    Returns:
        Path | None: Path to the created SDF file, or None if creation failed.
    """
    if len(text) == 0 or len(text) > 1:
        raise ValueError("Letter must be a single character")

    if letter_height_meters <= 0:
        raise ValueError("Letter height must be positive")

    if extrusion_depth_meters <= 0:
        raise ValueError("Extrusion depth must be positive")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Generate the 3D mesh from the letter
        mesh = _create_mesh_from_letter(
            text=text,
            font_name=font_name,
            letter_height_meters=letter_height_meters,
            extrusion_depth_meters=extrusion_depth_meters,
        )

        if mesh is None:
            logging.error(f"Failed to create mesh for letter '{text}'")
            return None

        # Save the mesh as an OBJ file
        mesh_path = output_path / f"{text}.obj"
        mesh.export(mesh_path)

        if use_bbox_collision_geometry:
            decomposition_method = "aabb"
            decomp_kwargs = None

        else:
            decomposition_method = "coacd"
            decomp_kwargs = {
                "threshold": 0.05,  # ↓ concavity threshold (lower ⇒ more pieces)
                "preprocess_mode": "auto",
                # "auto" (default) tries manifold remeshing only if needed
                "merge": True,  # allow post-merge to minimise hull count
            }

        # Use create_sdf_from_mesh to generate the SDF file
        create_sdf_from_mesh(
            mesh_path=mesh_path,
            mass=mass,
            scale=1.0,  # Mesh is already in correct units
            is_compliant=is_compliant,
            hydroelastic_modulus=hydroelastic_modulus,
            hunt_crossley_dissipation=hunt_crossley_dissipation,
            mu_dynamic=mu_dynamic,
            mu_static=mu_static,
            preview_with_trimesh=False,  # Don't show preview in automated generation
            decomposition_method=decomposition_method,
            coacd_kwargs=decomp_kwargs,
        )

        # Return path to the created SDF file
        sdf_path = output_path / f"{text}.sdf"
        return sdf_path if sdf_path.exists() else None

    except Exception as e:
        logging.error(f"Error creating SDF asset for letter '{text}': {e}")
        return None


def _create_mesh_from_letter(
    text: str,
    font_name: str = "Arial",
    letter_height_meters: float = 0.4,
    extrusion_depth_meters: float = 0.2,
) -> "trimesh.Trimesh | None":
    """
    Internal function to create a 3D mesh from a letter.

    Args:
        text (str): The character to convert.
        font_name (str): The name of the font to use.
        letter_height_meters (float): The physical height of the letter in meters.
        extrusion_depth_meters (float): The physical depth/thickness of the letter in meters.

    Returns:
        trimesh.Trimesh: A 3D mesh object of the letter in meters, or None if creation failed.
    """
    if not trimesh_available or not shapely_available:
        logging.error("Both trimesh and shapely are required for mesh generation")
        return None
    try:
        # Use a fixed font size for good mesh resolution, we'll scale to physical size later
        internal_font_size = 100

        # Set up font properties
        font_prop = FontProperties(
            family=font_name, style="normal", weight="normal", size=internal_font_size
        )

        # Generate the 2D path of the text
        # A text path is a sequence of vertices and control codes (e.g., MOVETO, LINETO)
        path = TextPath((0, 0), text, size=internal_font_size, prop=font_prop)

    except Exception as e:
        logging.error(f"Error generating font path: {e}")
        logging.error(
            f"Please ensure the specified font: '{font_name}' is available on your system."
        )
        return None

    # Process the 2D Path into Shapely Polygons
    path_polygons = path.to_polygons(closed_only=True)

    if not path_polygons:
        logging.warning("No polygons were generated from the text path.")
        return None

    # Convert the raw vertex lists into Shapely Polygon objects
    shapely_polygons = [Polygon(p) for p in path_polygons]

    # Identify exteriors as polygons that are not contained by any other polygon
    exteriors = []
    for p1 in shapely_polygons:
        is_exterior = True
        for p2 in shapely_polygons:
            # Don't compare a polygon to itself
            if p1 is p2:
                continue
            # If p1 is inside p2, it's a hole, not an exterior
            if p2.contains(p1):
                is_exterior = False
                break
        if is_exterior:
            exteriors.append(p1)

    # Identify holes by checking which polygons are contained within exteriors
    final_polygons = []
    all_holes = [p for p in shapely_polygons if p not in exteriors]

    for ext in exteriors:
        contained_holes = [h.exterior for h in all_holes if ext.contains(h)]
        # A Shapely Polygon is created from an exterior shell and a list of interior hole shells.
        final_polygons.append(Polygon(ext.exterior, contained_holes))

    # If no valid polygons were created, exit.
    if not final_polygons:
        logging.warning("Could not construct valid geometry from paths.")
        return None

    # Combine the resulting polygons. If there's only one, it's a Polygon.
    # If there are more (like for the letter 'i'), they form a MultiPolygon.
    if len(final_polygons) == 1:
        final_shape = final_polygons[0]
    else:
        final_shape = MultiPolygon(final_polygons)

    if final_shape.is_empty:
        logging.warning("Resulting Shapely polygon is empty.")
        return None

    # Calculate the current bounds of the shape in font coordinates
    bounds = final_shape.bounds  # (minx, miny, maxx, maxy)
    current_height = bounds[3] - bounds[1]  # maxy - miny

    if current_height <= 0:
        logging.warning("Letter has no height in font coordinates.")
        return None

    # Calculate scale factor to achieve desired physical height
    scale_factor = letter_height_meters / current_height

    # Scale the shape to physical dimensions
    final_shape_scaled = shapely_scale(
        final_shape, xfact=scale_factor, yfact=scale_factor, origin=(0, 0)
    )

    # Extrude the 2D shape into a 3D mesh using physical depth
    mesh = trimesh.creation.extrude_polygon(
        final_shape_scaled, height=extrusion_depth_meters
    )

    return mesh


def _visualize_letter(sdf_path: Path, letter: str, output_dir: str) -> None:
    """
    Visualize the generated letter using Drake's ModelVisualizer.

    Args:
        sdf_path (Path): Path to the generated SDF file.
        letter (str): The letter character.
        output_dir (str): Output directory containing the model files.
    """
    from pydrake.visualization import ModelVisualizer

    try:
        logging.info("Starting Drake ModelVisualizer...")

        # Create ModelVisualizer with no contact visualization
        visualizer = ModelVisualizer(publish_contacts=False)

        # Load the SDF file directly
        visualizer.AddModels(str(sdf_path))

        logging.info("Model loaded. Close the visualizer window to continue...")

        # Run the visualizer (this will block until closed)
        visualizer.Run()

    except Exception as e:
        logging.error(f"Error during visualization: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create an SDF asset from a single letter."
    )
    parser.add_argument(
        "letter",
        type=str,
        help="The letter to convert (single character only)",
    )
    parser.add_argument(
        "--font",
        type=str,
        default="DejaVu Sans",
        help="Font name to use (default: DejaVu Sans)",
    )
    parser.add_argument(
        "--letter-height",
        type=float,
        default=0.4,
        help="Physical height of the letter in meters (default: 0.4)",
    )
    parser.add_argument(
        "--extrusion-depth",
        type=float,
        default=0.2,
        help="Physical depth/thickness of the letter in meters (default: 0.2)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: {letter}_model)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the generated letter using Drake's ModelVisualizer",
    )

    args = parser.parse_args()

    # Validate letter input
    if len(args.letter) != 1:
        logging.warning(
            f"Error: Letter must be a single character, got '{args.letter}'"
        )
        exit(1)

    # Set default output directory if not provided
    output_dir = args.output_dir or f"{args.letter}_model"

    logging.info(f"Attempting to convert the letter '{args.letter}' to an SDF asset.")
    logging.info(f"Using font: {args.font}")
    logging.info(f"Letter height: {args.letter_height} meters")
    logging.info(f"Extrusion depth: {args.extrusion_depth} meters")

    # Generate the complete SDF asset
    sdf_path = create_sdf_asset_from_letter(
        text=args.letter,
        font_name=args.font,
        letter_height_meters=args.letter_height,
        extrusion_depth_meters=args.extrusion_depth,
        output_dir=output_dir,
    )

    if sdf_path is not None:
        logging.info(f"Successfully created SDF asset at: {sdf_path}")

        # Visualize if requested
        if args.visualize:
            _visualize_letter(sdf_path, args.letter, output_dir)
    else:
        logging.info("Failed to create SDF asset")
