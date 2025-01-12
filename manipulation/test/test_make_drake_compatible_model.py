import os
import tempfile
import unittest

from lxml import etree
from pydrake.multibody.parsing import PackageMap

from manipulation.utils import FindResource

try:
    from manipulation.make_drake_compatible_model import MakeDrakeCompatibleModel

    pymeshlab_available = True
except ImportError:
    pymeshlab_available = False
    print("pymeshlab not found.")
    print("Consider 'pip install pymeshlab'.")


@unittest.skipIf(not pymeshlab_available, "Requires pymeshlab dependency.")
class TestMakeDrakeCompatibleModel(unittest.TestCase):
    def test_urdf(self):
        original_filename = FindResource("test/models/test.urdf")
        input_filename = original_filename.replace(".urdf", "_modified.urdf")
        with open(original_filename, "r") as file:
            original_content = file.read()
        modified_content = original_content.replace(
            "replace_me_in_test_with_absolute_path", os.path.dirname(input_filename)
        )
        with open(input_filename, "w") as file:
            file.write(modified_content)
        output_filename = tempfile.mktemp(suffix=".urdf")
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
            f'filename="file://{os.path.dirname(input_filename)}/cube_from_obj_scaled_1_2_3.obj"',
            output_content,
        )
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
        output_filename = tempfile.mktemp(suffix=".xml")
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

        root = etree.parse(output_filename)
        planes_to_box = root.findall(".//body[@name='floor']/geom")
        self.assertEqual(len(planes_to_box), 2)
        for geom in planes_to_box:
            self.assertEqual(geom.attrib["size"], "1000 1000 1")
            self.assertEqual(geom.attrib["type"], "box")
            if geom.attrib["name"] == "wo_pos":
                self.assertEqual(geom.attrib["pos"], "0 0 -1")
            elif geom.attrib["name"] == "w_pos":
                self.assertEqual(geom.attrib["pos"], "1.0 2.0 2.0")
        # Clean up the temp file
        os.remove(output_filename)


if __name__ == "__main__":
    unittest.main()
