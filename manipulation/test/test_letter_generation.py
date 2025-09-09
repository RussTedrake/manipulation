import shutil
import tempfile
import unittest
from pathlib import Path

try:
    import trimesh  # noqa: F401

    trimesh_available = True
except ImportError:
    trimesh_available = False
    print("trimesh not found.")
    print("Consider 'pip install trimesh'.")

try:
    from shapely.geometry import MultiPolygon, Polygon  # noqa: F401

    shapely_available = True
except ImportError:
    shapely_available = False
    print("shapely not found.")
    print("Consider 'pip install shapely'.")

try:
    import coacd  # noqa: F401

    coacd_available = True
except ImportError:
    coacd_available = False
    print("coacd not found.")
    print("Consider 'pip install coacd'.")

# Import the module under test only if dependencies are available
dependencies_available = all([trimesh_available, shapely_available, coacd_available])

if dependencies_available:
    from manipulation.letter_generation import create_sdf_asset_from_letter


@unittest.skipIf(
    not dependencies_available,
    "Requires trimesh, shapely, and coacd dependencies.",
)
class LetterGenerationTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmp_dir)

    def test_create_sdf_asset_from_letter(self):
        # Expected number of convex pieces for each letter
        letter_expected_pieces = {"I": 1, "A": 5}

        for letter, expected_pieces in letter_expected_pieces.items():
            with self.subTest(letter=letter):
                create_sdf_asset_from_letter(
                    text=letter,
                    font_name="DejaVu Sans",
                    letter_height_meters=0.4,
                    extrusion_depth_meters=0.15,
                    output_dir=f"{self._tmp_dir}/{letter}_model",
                )

                model_dir = Path(self._tmp_dir) / f"{letter}_model" / f"{letter}_parts"
                self.assertTrue(
                    model_dir.exists(),
                    f"Model directory should be created for letter {letter}",
                )

                # Check that the expected number of convex pieces was generated
                obj_files = list(model_dir.glob("convex_*.obj"))
                self.assertEqual(
                    len(obj_files),
                    expected_pieces,
                    f"Letter {letter} should generate exactly {expected_pieces} convex piece(s)",
                )

                # Check that the files are not empty
                for obj_file in obj_files:
                    self.assertGreater(
                        obj_file.stat().st_size,
                        0,
                        f"OBJ file {obj_file} should not be empty for letter {letter}",
                    )

    def test_create_sdf_asset_different_fonts(self):
        """Test SDF asset creation with different fonts."""
        fonts_to_test = ["DejaVu Sans", "DejaVu Serif"]
        letter = "B"

        for font in fonts_to_test:
            with self.subTest(font=font):
                sdf_path = create_sdf_asset_from_letter(
                    text=letter,
                    font_name=font,
                    letter_height_meters=0.4,
                    extrusion_depth_meters=0.15,
                    output_dir=self._tmp_dir,
                )

                self.assertIsNotNone(
                    sdf_path, f"SDF path should not be None for font {font}"
                )
                if sdf_path:
                    self.assertTrue(
                        sdf_path.exists(), f"SDF file should exist for font {font}"
                    )

    def test_create_sdf_asset_different_parameters(self):
        """Test SDF asset creation with different parameters."""
        letter = "C"

        # Test with different letter heights
        for letter_height in [0.2, 0.4, 0.6]:
            with self.subTest(letter_height=letter_height):
                sdf_path = create_sdf_asset_from_letter(
                    text=letter,
                    font_name="DejaVu Sans",
                    letter_height_meters=letter_height,
                    extrusion_depth_meters=0.15,
                    output_dir=self._tmp_dir,
                )

                self.assertIsNotNone(
                    sdf_path,
                    f"SDF path should not be None for letter_height {letter_height}",
                )
                if sdf_path:
                    self.assertTrue(
                        sdf_path.exists(),
                        f"SDF file should exist for letter_height {letter_height}",
                    )

        # Test with different extrusion depths
        for depth in [0.1, 0.15, 0.2]:
            with self.subTest(extrusion_depth=depth):
                sdf_path = create_sdf_asset_from_letter(
                    text=letter,
                    font_name="DejaVu Sans",
                    letter_height_meters=0.4,
                    extrusion_depth_meters=depth,
                    output_dir=self._tmp_dir,
                )

                self.assertIsNotNone(
                    sdf_path, f"SDF path should not be None for depth {depth}"
                )
                if sdf_path:
                    self.assertTrue(
                        sdf_path.exists(), f"SDF file should exist for depth {depth}"
                    )

        for use_bbox_collision_geometry in [True, False]:
            with self.subTest(extrusion_depth=depth):
                sdf_path = create_sdf_asset_from_letter(
                    text=letter,
                    font_name="DejaVu Sans",
                    letter_height_meters=0.4,
                    extrusion_depth_meters=depth,
                    output_dir=self._tmp_dir,
                    use_bbox_collision_geometry=use_bbox_collision_geometry,
                )

                self.assertIsNotNone(
                    sdf_path,
                    f"SDF path should not be None for use_bbox_collision_geometry {use_bbox_collision_geometry}",
                )
                if sdf_path:
                    self.assertTrue(
                        sdf_path.exists(),
                        f"SDF file should exist for use_bbox_collision_geometry {use_bbox_collision_geometry}",
                    )

    def test_create_sdf_asset_invalid_input(self):
        """Test error handling with invalid inputs."""
        # Test with multiple characters (should fail)
        with self.assertRaises(ValueError):
            create_sdf_asset_from_letter(
                text="AB",  # Multiple characters
                font_name="DejaVu Sans",
                letter_height_meters=0.4,
                extrusion_depth_meters=0.15,
                output_dir=self._tmp_dir,
            )

        # Test with empty string (should fail)
        with self.assertRaises(ValueError):
            create_sdf_asset_from_letter(
                text="",  # Empty string
                font_name="DejaVu Sans",
                letter_height_meters=0.4,
                extrusion_depth_meters=0.15,
                output_dir=self._tmp_dir,
            )


if __name__ == "__main__":
    unittest.main()
