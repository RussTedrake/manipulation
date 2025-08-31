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
        letter_expected_pieces = {"I": 1, "A": 6}

        for letter, expected_pieces in letter_expected_pieces.items():
            with self.subTest(letter=letter):
                create_sdf_asset_from_letter(
                    text=letter,
                    font_name="Arial",
                    font_size=100,
                    extrusion_height=10.0,
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
        fonts_to_test = ["Arial", "DejaVu Sans"]
        letter = "B"

        for font in fonts_to_test:
            with self.subTest(font=font):
                sdf_path = create_sdf_asset_from_letter(
                    text=letter,
                    font_name=font,
                    font_size=50,
                    extrusion_height=5.0,
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

        # Test with different font sizes
        for font_size in [50, 100, 200]:
            with self.subTest(font_size=font_size):
                sdf_path = create_sdf_asset_from_letter(
                    text=letter,
                    font_name="Arial",
                    font_size=font_size,
                    extrusion_height=10.0,
                    output_dir=self._tmp_dir,
                )

                self.assertIsNotNone(
                    sdf_path, f"SDF path should not be None for font_size {font_size}"
                )
                if sdf_path:
                    self.assertTrue(
                        sdf_path.exists(),
                        f"SDF file should exist for font_size {font_size}",
                    )

        # Test with different extrusion heights
        for height in [5.0, 15.0, 25.0]:
            with self.subTest(extrusion_height=height):
                sdf_path = create_sdf_asset_from_letter(
                    text=letter,
                    font_name="Arial",
                    font_size=100,
                    extrusion_height=height,
                    output_dir=self._tmp_dir,
                )

                self.assertIsNotNone(
                    sdf_path, f"SDF path should not be None for height {height}"
                )
                if sdf_path:
                    self.assertTrue(
                        sdf_path.exists(), f"SDF file should exist for height {height}"
                    )

    def test_create_sdf_asset_invalid_input(self):
        """Test error handling with invalid inputs."""
        # Test with multiple characters (should fail)
        with self.assertRaises(AssertionError):
            create_sdf_asset_from_letter(
                text="AB",  # Multiple characters
                font_name="Arial",
                font_size=100,
                extrusion_height=10.0,
                output_dir=self._tmp_dir,
            )

        # Test with empty string (should fail)
        with self.assertRaises(AssertionError):
            create_sdf_asset_from_letter(
                text="",  # Empty string
                font_name="Arial",
                font_size=100,
                extrusion_height=10.0,
                output_dir=self._tmp_dir,
            )


if __name__ == "__main__":
    unittest.main()
