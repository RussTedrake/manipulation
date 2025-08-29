import shutil
import tempfile
import unittest
from pathlib import Path

# Test for optional dependencies
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
    from matplotlib.font_manager import FontProperties  # noqa: F401
    from matplotlib.textpath import TextPath  # noqa: F401

    matplotlib_available = True
except ImportError:
    matplotlib_available = False
    print("matplotlib not found.")
    print("Consider 'pip install matplotlib'.")

try:
    import coacd  # noqa: F401

    coacd_available = True
except ImportError:
    coacd_available = False
    print("coacd not found.")
    print("Consider 'pip install coacd'.")

# Import the module under test only if dependencies are available
dependencies_available = all(
    [trimesh_available, shapely_available, matplotlib_available, coacd_available]
)

if dependencies_available:
    from manipulation.letter_generation import create_mesh_from_letter


@unittest.skipIf(
    not dependencies_available,
    "Requires trimesh, shapely, matplotlib, and coacd dependencies.",
)
class LetterGenerationTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp_dir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmp_dir)

    def test_create_mesh_from_letter(self):
        # Expected number of convex pieces for each letter
        letter_expected_pieces = {"I": 1, "A": 6}

        for letter, expected_pieces in letter_expected_pieces.items():
            with self.subTest(letter=letter):
                create_mesh_from_letter(
                    text=letter,
                    font_name="Arial",
                    font_size=100,
                    extrusion_height=10.0,
                    model_dir=self._tmp_dir,
                )

                # Check that the model directory was created
                model_dir = Path(self._tmp_dir) / f"{letter}_model"
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

    # def test_create_mesh_from_letter_different_fonts(self):
    #     """Test mesh creation with different fonts."""
    #     fonts_to_test = ["Arial", "DejaVu Sans"]
    #     letter = "B"

    #     for font in fonts_to_test:
    #         with self.subTest(font=font):
    #             create_mesh_from_letter(
    #                 text=letter,
    #                 font_name=font,
    #                 font_size=50,
    #                 extrusion_height=5.0,
    #                 model_dir=self._tmp_dir
    #             )

    #             model_dir = Path(self._tmp_dir) / f"{letter}_model"
    #             self.assertTrue(model_dir.exists(), f"Model directory should be created for font {font}")

    # def test_create_mesh_from_letter_different_parameters(self):
    #     """Test mesh creation with different parameters."""
    #     letter = "C"

    #     # Test with different font sizes
    #     for font_size in [50, 100, 200]:
    #         with self.subTest(font_size=font_size):
    #             create_mesh_from_letter(
    #                 text=letter,
    #                 font_name="Arial",
    #                 font_size=font_size,
    #                 extrusion_height=10.0,
    #                 model_dir=self._tmp_dir
    #             )

    #             model_dir = Path(self._tmp_dir) / f"{letter}_model"
    #             self.assertTrue(model_dir.exists(), f"Model directory should be created for font_size {font_size}")

    #     # Test with different extrusion heights
    #     for height in [5.0, 15.0, 25.0]:
    #         with self.subTest(extrusion_height=height):
    #             create_mesh_from_letter(
    #                 text=letter,
    #                 font_name="Arial",
    #                 font_size=100,
    #                 extrusion_height=height,
    #                 model_dir=self._tmp_dir
    #             )

    #             model_dir = Path(self._tmp_dir) / f"{letter}_model"
    #             self.assertTrue(model_dir.exists(), f"Model directory should be created for height {height}")

    # def test_create_mesh_from_letter_complex_letters(self):
    #     """Test mesh creation with letters that have holes (like 'O', 'P', 'R')."""
    #     complex_letters = ["O", "P", "R"]

    #     for letter in complex_letters:
    #         with self.subTest(letter=letter):
    #             create_mesh_from_letter(
    #                 text=letter,
    #                 font_name="Arial",
    #                 font_size=100,
    #                 extrusion_height=10.0,
    #                 model_dir=self._tmp_dir
    #             )

    #             model_dir = Path(self._tmp_dir) / f"{letter}_model"
    #             self.assertTrue(model_dir.exists(), f"Model directory should be created for letter {letter}")

    #             obj_files = list(model_dir.glob("convex_*.obj"))
    #             self.assertGreater(len(obj_files), 0, f"At least one convex piece should be generated for letter {letter}")

    # def test_create_mesh_from_letter_invalid_input(self):
    #     """Test error handling with invalid inputs."""
    #     # Test with multiple characters (should fail)
    #     with self.assertRaises(AssertionError):
    #         create_mesh_from_letter(
    #             text="AB",  # Multiple characters
    #             font_name="Arial",
    #             font_size=100,
    #             extrusion_height=10.0,
    #             model_dir=self._tmp_dir
    #         )

    #     # Test with empty string (should fail)
    #     with self.assertRaises(AssertionError):
    #         create_mesh_from_letter(
    #             text="",  # Empty string
    #             font_name="Arial",
    #             font_size=100,
    #             extrusion_height=10.0,
    #             model_dir=self._tmp_dir
    #         )

    # def test_create_mesh_from_letter_invalid_font(self):
    #     """Test handling of invalid font names."""
    #     # This should not crash but might return None or use a fallback font
    #     create_mesh_from_letter(
    #         text="D",
    #         font_name="NonExistentFont123",
    #         font_size=100,
    #         extrusion_height=10.0,
    #         model_dir=self._tmp_dir
    #     )
    #     # The function should handle this gracefully (either work with fallback or return None)
    #     # We don't assert a specific behavior here as it depends on the system's font handling

    # def test_create_urdf_from_mesh(self):
    #     """Test URDF creation from generated mesh."""
    #     letter = "E"

    #     # First create the mesh
    #     create_mesh_from_letter(
    #         text=letter,
    #         font_name="Arial",
    #         font_size=100,
    #         extrusion_height=10.0,
    #         model_dir=self._tmp_dir
    #     )

    #     # Then create the URDF
    #     create_urdf_from_mesh(
    #         text=letter,
    #         scale=0.01,
    #         model_dir=self._tmp_dir
    #     )

    #     # Check that URDF file was created
    #     model_dir = Path(self._tmp_dir) / f"{letter}_model"
    #     urdf_file = model_dir / f"{letter}_convex.urdf"
    #     self.assertTrue(urdf_file.exists(), "URDF file should be created")
    #     self.assertGreater(urdf_file.stat().st_size, 0, "URDF file should not be empty")

    #     # Check basic URDF structure by reading the file
    #     with open(urdf_file, 'r', encoding='utf-8') as f:
    #         urdf_content = f.read()

    #     # Basic checks for URDF structure
    #     self.assertIn('<?xml version="1.0"', urdf_content, "URDF should have XML declaration")
    #     self.assertIn('<robot', urdf_content, "URDF should have robot element")
    #     self.assertIn('<link', urdf_content, "URDF should have link element")
    #     self.assertIn('<collision', urdf_content, "URDF should have collision elements")
    #     self.assertIn('<visual', urdf_content, "URDF should have visual elements")
    #     self.assertIn('<inertial', urdf_content, "URDF should have inertial element")
    #     self.assertIn('convex_', urdf_content, "URDF should reference convex mesh files")

    # def test_create_urdf_from_mesh_different_scales(self):
    #     """Test URDF creation with different scales."""
    #     letter = "F"

    #     # Create the mesh first
    #     create_mesh_from_letter(
    #         text=letter,
    #         font_name="Arial",
    #         font_size=100,
    #         extrusion_height=10.0,
    #         model_dir=self._tmp_dir
    #     )

    #     # Test different scales
    #     scales = [0.005, 0.01, 0.02]
    #     for scale in scales:
    #         with self.subTest(scale=scale):
    #             create_urdf_from_mesh(
    #                 text=letter,
    #                 scale=scale,
    #                 model_dir=self._tmp_dir
    #             )

    #             model_dir = Path(self._tmp_dir) / f"{letter}_model"
    #             urdf_file = model_dir / f"{letter}_convex.urdf"
    #             self.assertTrue(urdf_file.exists(), f"URDF file should be created for scale {scale}")

    #             # Check that the scale is correctly set in the URDF
    #             with open(urdf_file, 'r', encoding='utf-8') as f:
    #                 urdf_content = f.read()

    #             scale_str = f"{scale} {scale} {scale}"
    #             self.assertIn(scale_str, urdf_content, f"URDF should contain scale {scale_str}")

    # def test_create_urdf_from_mesh_without_obj_files(self):
    #     """Test URDF creation when no OBJ files exist."""
    #     letter = "G"

    #     # Try to create URDF without first creating mesh files
    #     # This should handle the case gracefully
    #     create_urdf_from_mesh(
    #         text=letter,
    #         scale=0.01,
    #         model_dir=self._tmp_dir
    #     )

    #     # Check if URDF was created (it might be empty or minimal)
    #     model_dir = Path(self._tmp_dir) / f"{letter}_model"
    #     urdf_file = model_dir / f"{letter}_convex.urdf"

    #     if urdf_file.exists():
    #         # If URDF was created, it should at least have basic structure
    #         with open(urdf_file, 'r', encoding='utf-8') as f:
    #             urdf_content = f.read()
    #         self.assertIn('<robot', urdf_content, "URDF should have robot element even without meshes")

    # def test_numerical_and_special_characters(self):
    #     """Test with numerical and special characters that might be supported."""
    #     # Test with numbers (if supported by the font system)
    #     test_chars = ["1", "5", "8"]

    #     for char in test_chars:
    #         with self.subTest(char=char):
    #             try:
    #                 create_mesh_from_letter(
    #                     text=char,
    #                     font_name="Arial",
    #                     font_size=100,
    #                     extrusion_height=10.0,
    #                     model_dir=self._tmp_dir
    #                 )

    #                 model_dir = Path(self._tmp_dir) / f"{char}_model"
    #                 if model_dir.exists():
    #                     obj_files = list(model_dir.glob("convex_*.obj"))
    #                     self.assertGreater(len(obj_files), 0, f"At least one convex piece should be generated for character {char}")
    #             except (ValueError, TypeError, RuntimeError) as e:
    #                 # Some characters might not be supported, which is acceptable
    #                 print(f"Character {char} not supported: {e}")


if __name__ == "__main__":
    unittest.main()
