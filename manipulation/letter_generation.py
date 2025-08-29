import pathlib

import coacd
import trimesh
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextPath
from shapely.geometry import MultiPolygon, Polygon


def create_mesh_from_letter(
    text: str,
    font_name: str = "Arial",
    font_size: int = 100,
    extrusion_height: float = 10.0,
    model_dir: str = ".",
):
    """
    Converts a single letter into a 3D mesh.

    Args:
        text (str): The character(s) to convert.
        font_name (str): The name of the font to use (e.g., 'Arial', 'Times New Roman').
        font_size (int): The font size in points, affecting the mesh resolution.
        extrusion_height (float): The height to extrude the 2D letter shape into 3D.
        dpi (int): The dots per inch resolution for path generation.

    Returns:
        trimesh.Trimesh: A 3D mesh object of the letter.
                         Returns None if the font path cannot be generated.
    """
    assert len(text) == 1, "Only one letter can be converted at a time"
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

    # --- Process the 2D Path into Shapely Polygons ---
    path_polygons = path.to_polygons(closed_only=True)

    if not path_polygons:
        print("No polygons were generated from the text path.")
        return None

    # Convert the raw vertex lists into Shapely Polygon objects
    shapely_polygons = [Polygon(p) for p in path_polygons]

    # New robust logic: Identify exteriors as polygons that are not contained by any other polygon.
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

    # Identify holes by checking which of the remaining polygons are contained within our exteriors.
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

    # --- Extrude the 2D Polygons into a 3D Mesh ---
    if final_shape.is_empty:
        print("Resulting Shapely polygon is empty.")
        return None

    # Trimesh can directly extrude Shapely Polygons and MultiPolygons
    mesh = trimesh.creation.extrude_polygon(final_shape, height=extrusion_height)

    # 2) wrap it for CoACD
    cmesh = coacd.Mesh(mesh.vertices, mesh.faces)

    # 3) run the decomposition
    parts = coacd.run_coacd(
        cmesh,
        threshold=0.05,  # ↓ concavity threshold (lower ⇒ more pieces)
        preprocess_mode="auto",
        # "auto" (default) tries manifold remeshing only if needed
        merge=True,  # allow post-merge to minimise hull count
    )

    # 4) convert each returned (V, F) pair back to trimesh and save

    # Create directory if it doesn't exist
    out_dir = pathlib.Path(model_dir) / f"{text}_model"
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, (v, f) in enumerate(parts):
        trimesh.Trimesh(v, f, process=False).export(out_dir / f"convex_{i}.obj")


def create_urdf_from_mesh(text: str, scale=0.01, model_dir="."):
    import glob
    import xml.etree.ElementTree as ET

    model_path = f"{text}_model"
    # grab the OBJ paths you just exported ----------------------------------------
    obj_paths = sorted(
        glob.glob(str(pathlib.Path(model_dir) / model_path / "convex_*.obj"))
    )  # adjust the glob as needed

    # build the XML ----------------------------------------------------------------
    robot = ET.Element("robot", name=f"{text}_letter")

    link = ET.SubElement(robot, "link", name=f"{text}_body")

    # ── add dummy inertia ────────────────────────────────────────────
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(inertial, "mass", value="1.0")
    ET.SubElement(
        inertial,
        "inertia",
        ixx="0.01",
        ixy="0.0",
        ixz="0.0",
        iyy="0.01",
        iyz="0.0",
        izz="0.01",
    )

    for i, path in enumerate(obj_paths):
        relative_path = path.split("/")[-1]
        # --------------- collision element ---------------
        coll = ET.SubElement(link, "collision", name=f"part_{i}")
        ET.SubElement(coll, "origin", xyz="0 0 0", rpy="0 0 0")
        c_geom = ET.SubElement(coll, "geometry")
        ET.SubElement(
            c_geom,
            "mesh",
            filename=str(relative_path),
            scale=f"{scale} {scale} {scale}",
        )

        # --------------- *visual* element (same mesh) ----
        vis = ET.SubElement(link, "visual", name=f"part_{i}_vis")
        ET.SubElement(vis, "origin", xyz="0 0 0", rpy="0 0 0")
        v_geom = ET.SubElement(vis, "geometry")
        ET.SubElement(
            v_geom,
            "mesh",
            filename=str(relative_path),
            scale=f"{scale} {scale} {scale}",
        )

    # save -------------------------------------------------------------------------
    ET.indent(robot)  # Python ≥ 3.9 for pretty-printing
    ET.ElementTree(robot).write(
        str(pathlib.Path(model_dir) / model_path / f"{text}_convex.urdf"),
        xml_declaration=True,
        encoding="utf-8",
    )


if __name__ == "__main__":
    # --- Configuration ---
    LETTER_TO_CONVERT = "P"
    FONT_TO_USE = (
        "DejaVu Sans"  # A common font, change if not available. 'Arial' on Windows.
    )
    OUTPUT_FILENAME = "letter.stl"
    EXTRUSION_DEPTH = 15.0  # How "thick" the letter will be

    print(f"Attempting to convert the letter '{LETTER_TO_CONVERT}' to a 3D mesh.")
    print(f"Using font: {FONT_TO_USE}")

    # Generate the mesh
    create_mesh_from_letter(
        LETTER_TO_CONVERT, font_name=FONT_TO_USE, extrusion_height=EXTRUSION_DEPTH
    )

    create_urdf_from_mesh(LETTER_TO_CONVERT)
