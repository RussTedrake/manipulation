from pathlib import Path

import trimesh
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from shapely.geometry import MultiPolygon, Polygon

from manipulation.create_sdf_from_mesh import create_sdf_from_mesh


def create_sdf_asset_from_letter(
    text: str,
    font_name: str = "Arial",
    font_size: int = 100,
    scale: float = 0.01,
    mass: float = 1.0,
    extrusion_height: float = 10.0,
    is_compliant: bool = False,
    hydroelastic_modulus: float = 1e8,
    hunt_crossley_dissipation: float | None = None,
    mu_dynamic: float | None = 1.0,
    mu_static: float | None = None,
    output_dir: str = ".",
):
    """
    Creates a complete SDF asset (mesh + SDF file) from a single letter.

    Args:
        text (str): The character to convert (single letter only).
        font_name (str): The name of the font to use (e.g., 'Arial', 'Times New Roman').
        font_size (int): The font size in points, affecting the mesh resolution.
        scale (float): Scale factor to convert mesh coordinates to meters.
        mass (float): The mass in kg of the object for inertia calculations.
        extrusion_height (float): The height to extrude the 2D letter shape into 3D.
        is_compliant (bool): Whether to use compliant hydroelastic contact.
        hydroelastic_modulus (float): The hydroelastic modulus (Pa).
        hunt_crossley_dissipation (float | None): Hunt-Crossley dissipation parameter (s/m).
        mu_dynamic (float | None): Coefficient of dynamic friction.
        mu_static (float | None): Coefficient of static friction.
        output_dir (str): Directory where the output files will be saved.

    Returns:
        Path | None: Path to the created SDF file, or None if creation failed.
    """
    assert len(text) == 1, "Only one letter can be converted at a time"

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # Generate the 3D mesh from the letter
        mesh = _create_mesh_from_letter(
            text=text,
            font_name=font_name,
            font_size=font_size,
            extrusion_height=extrusion_height,
        )

        if mesh is None:
            print(f"Failed to create mesh for letter '{text}'")
            return None

        # Save the mesh as an OBJ file
        mesh_path = output_path / f"{text}.obj"
        mesh.export(mesh_path)

        # Use create_sdf_from_mesh to generate the SDF file
        create_sdf_from_mesh(
            mesh_path=mesh_path,
            mass=mass,
            scale=scale,
            is_compliant=is_compliant,
            hydroelastic_modulus=hydroelastic_modulus,
            hunt_crossley_dissipation=hunt_crossley_dissipation,
            mu_dynamic=mu_dynamic,
            mu_static=mu_static,
            preview_with_trimesh=False,  # Don't show preview in automated generation
            use_coacd=True,
            coacd_kwargs={
                "threshold": 0.05,  # ↓ concavity threshold (lower ⇒ more pieces)
                "preprocess_mode": "auto",
                # "auto" (default) tries manifold remeshing only if needed
                "merge": True,  # allow post-merge to minimise hull count
            },
        )

        # Return path to the created SDF file
        sdf_path = output_path / f"{text}.sdf"
        return sdf_path if sdf_path.exists() else None

    except Exception as e:
        print(f"Error creating SDF asset for letter '{text}': {e}")
        return None


def _create_mesh_from_letter(
    text: str,
    font_name: str = "Arial",
    font_size: int = 100,
    extrusion_height: float = 10.0,
):
    """
    Internal function to create a 3D mesh from a letter.

    Returns:
        trimesh.Trimesh: A 3D mesh object of the letter, or None if creation failed.
    """
    try:
        # Set up font properties
        font_prop = FontProperties(
            family=font_name, style="normal", weight="normal", size=font_size
        )

        # Generate the 2D path of the text
        # A text path is a sequence of vertices and control codes (e.g., MOVETO, LINETO)
        path = TextPath((0, 0), text, size=font_size, prop=font_prop)

    except Exception as e:
        print(f"Error generating font path: {e}")
        print("Please ensure the specified font is available on your system.")
        return None

    # Process the 2D Path into Shapely Polygons
    path_polygons = path.to_polygons(closed_only=True)

    if not path_polygons:
        print("No polygons were generated from the text path.")
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
        print("Could not construct valid geometry from paths.")
        return None

    # Combine the resulting polygons. If there's only one, it's a Polygon.
    # If there are more (like for the letter 'i'), they form a MultiPolygon.
    if len(final_polygons) == 1:
        final_shape = final_polygons[0]
    else:
        final_shape = MultiPolygon(final_polygons)

    if final_shape.is_empty:
        print("Resulting Shapely polygon is empty.")
        return None

    # Extrude the 2D shape into a 3D mesh
    mesh = trimesh.creation.extrude_polygon(final_shape, height=extrusion_height)

    return mesh


if __name__ == "__main__":
    LETTER_TO_CONVERT = "P"
    FONT_TO_USE = "DejaVu Sans"
    EXTRUSION_DEPTH = 15.0

    print(f"Attempting to convert the letter '{LETTER_TO_CONVERT}' to an SDF asset.")
    print(f"Using font: {FONT_TO_USE}")

    # Generate the complete SDF asset
    sdf_path = create_sdf_asset_from_letter(
        text=LETTER_TO_CONVERT,
        font_name=FONT_TO_USE,
        extrusion_height=EXTRUSION_DEPTH,
        scale=0.01,
        output_dir=f"{LETTER_TO_CONVERT}_letter_model",
    )

    if sdf_path:
        print(f"Successfully created SDF asset at: {sdf_path}")
    else:
        print("Failed to create SDF asset")
