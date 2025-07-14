import os
import re
import tempfile
import unittest

from lxml import etree
from pydrake.multibody.parsing import PackageMap

from manipulation.utils import FindResource

try:
    from manipulation.make_drake_compatible_model import (
        MakeDrakeCompatibleModel,
        _convert_mesh,
    )
    from manipulation.remotes import AddMujocoMenagerie

    trimesh_available = True
except ImportError:
    trimesh_available = False
    print("trimesh not found.")
    print("Consider 'pip install trimesh'.")


@unittest.skipIf(not trimesh_available, "Requires trimesh dependency.")
class TestMakeDrakeCompatibleModel(unittest.TestCase):
    def test_obj_no_textures(self):
        input_url = FindResource("test/models/cube.obj")
        output_url, output_path = _convert_mesh(
            url=input_url, path=input_url, overwrite=True
        )
        self.assertTrue(os.path.exists(output_path))
        self.assertFalse(os.path.exists(output_path + ".mtl"))

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
        original_filename = FindResource("test/models/test.sdf")
        input_filename = original_filename.replace(".sdf", "_modified.sdf")
        with open(original_filename, "r") as file:
            original_content = file.read()
        modified_content = original_content.replace(
            "replace_me_in_test_with_absolute_path", os.path.dirname(input_filename)
        )
        with open(input_filename, "w") as file:
            file.write(modified_content)
        output_filename = tempfile.mktemp(suffix=".sdf")
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
        self.assertIn("<uri>cube_from_stl.obj</uri>", output_content)
        self.assertIn("<uri>cube_from_dae.obj</uri>", output_content)
        self.assertIn("<uri>cube.obj</uri>", output_content)
        self.assertIn(
            f"<uri>file://{os.path.dirname(input_filename)}/cube_from_obj_scaled_1_2_3.obj</uri>",
            output_content,
        )
        self.assertIn(
            "<uri>package://manipulation_test_models/cube_from_obj_scaled_n1_1_1.obj</uri>",
            output_content,
        )
        # Clean up the temp file
        os.remove(output_filename)

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

    def test_mjcf_meshdir(self):
        input_filename = FindResource("test/models/test_meshdir.xml")
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
        # Clean up the temp file
        os.remove(output_filename)

    def test_mjcf_defaults(self):
        input_filename = FindResource("test/models/test_defaults.xml")
        output_filename = tempfile.mktemp(suffix=".xml")
        package_map = PackageMap()
        package_map.AddPackageXml(filename=FindResource("test/models/package.xml"))
        MakeDrakeCompatibleModel(
            input_filename=input_filename,
            output_filename=output_filename,
            package_map=package_map,
            remap_mujoco_geometry_groups={0: 3},
        )
        self.assertTrue(os.path.exists(output_filename))
        with open(output_filename, "r") as f:
            output_content = f.read()
        self.assertIn(
            'file="cube_from_stl_scaled_0.001_0.002_0.003.obj"', output_content
        )
        self.assertIn('group="3"', output_content)
        # Clean up the temp file
        os.remove(output_filename)

    def test_mujoco_menagerie(self):
        """Test all files in the mujoco_menagerie package."""
        package_map = PackageMap()
        AddMujocoMenagerie(package_map)
        menagerie = package_map.GetPath("mujoco_menagerie")
        # Find all XML files recursively under the menagerie path
        results = ""
        for root, dirs, files in os.walk(menagerie):
            for file in files:
                if file.endswith(".drake.xml"):
                    continue
                if file.endswith("scene.xml"):
                    with self.subTest(file=file):
                        original_file = os.path.join(root, file)
                        drake_compatible_file = original_file.replace(
                            ".xml", ".drake.xml"
                        )
                        try:
                            MakeDrakeCompatibleModel(
                                original_file, drake_compatible_file
                            )
                            results += (
                                f"PASS: {os.path.relpath(root, menagerie)}/{file}\n"
                            )
                        except Exception as e:
                            rel_path = os.path.relpath(root, menagerie)
                            # Known type/message pairs that we expect to encounter
                            known_exceptions = [
                                # No more known exceptions (yeah!)... but the format is:
                                # (KeyError, r".*'file'.*", "Need to parse defaults"),
                            ]
                            known_failure = False
                            for exc_type, msg_pattern, note in known_exceptions:
                                if isinstance(e, exc_type) and re.match(
                                    msg_pattern, str(e)
                                ):
                                    results += f"FAIL: {os.path.join(rel_path, file)}: {note}\n"
                                    known_failure = True
                                    break
                            if not known_failure:
                                results += f"FAIL: {os.path.join(rel_path, file)}: Unregistered exception\n"
                                raise  # Re-raise if not a known exception
        print(results)


if __name__ == "__main__":
    unittest.main()
