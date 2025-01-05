import os
import tempfile
import unittest

from pydrake.multibody.parsing import PackageMap

from manipulation.make_drake_compatible_model import MakeDrakeCompatibleModel
from manipulation.utils import FindResource


class TestMakeDrakeCompatibleModel(unittest.TestCase):
    def test_urdf(self):
        input_filename = FindResource("test/models/test.urdf")
        output_filename = tempfile.mktemp()
        package_map = PackageMap()
        package_map.AddPackageXml(filename=FindResource("test/models/package.xml"))
        MakeDrakeCompatibleModel(
            input_filename=input_filename,
            output_filename=output_filename,
            package_map=package_map,
        )
        self.assertTrue(os.path.exists(output_filename))
        with open(output_filename, "r") as f:
            output_content = f.read()
        self.assertIn('filename="cube_from_stl.obj"', output_content)
        self.assertIn('filename="cube_from_dae.obj"', output_content)
        self.assertIn('filename="cube.obj"', output_content)
        self.assertIn(
            'filename="package://manipulation_test_models/cube_from_obj_scaled_n1_1_1.obj"',
            output_content,
        )
        # Clean up the temp file
        os.remove(output_filename)

    def test_sdf(self):
        # TODO(russt): Implement this.
        pass

    def test_mjcf(self):
        input_filename = FindResource("test/models/test.xml")
        output_filename = tempfile.mktemp()
        package_map = PackageMap()
        package_map.AddPackageXml(filename=FindResource("test/models/package.xml"))
        MakeDrakeCompatibleModel(
            input_filename=input_filename,
            output_filename=output_filename,
            package_map=package_map,
        )
        self.assertTrue(os.path.exists(output_filename))
        with open(output_filename, "r") as f:
            output_content = f.read()
        self.assertIn('file="cube_from_stl.obj"', output_content)
        self.assertIn('file="cube_from_stl_scaled_n1_1_1.obj"', output_content)
        self.assertIn('file="cube_from_dae.obj"', output_content)
        self.assertIn('file="cube.obj"', output_content)
        self.assertIn('file="cube_from_obj_scaled_1.2_2.3_3.4.obj"', output_content)
        # Clean up the temp file
        os.remove(output_filename)


if __name__ == "__main__":
    unittest.main()
