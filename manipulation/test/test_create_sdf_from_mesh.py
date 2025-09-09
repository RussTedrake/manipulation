import fnmatch
import os
import shutil
import tempfile
import unittest
from pathlib import Path

try:
    import trimesh

    from manipulation.create_sdf_from_mesh import create_sdf_from_mesh

    trimesh_available = True
except ImportError:
    trimesh_available = False
    print("trimesh not found.")
    print("Consider 'pip install trimesh'.")


try:
    import coacd  # noqa: F401

    coacd_available = True
except ImportError:
    coacd_available = False
    print("coacd not found.")
    print("Consider 'pip install coacd'.")

try:
    import vhacdx  # noqa: F401

    vhacdx_available = True
except ImportError:
    vhacdx_available = False
    print("vhacdx not found.")
    print("Consider 'pip install vhacdx'.")


@unittest.skipIf(not trimesh_available, "Requires trimesh dependency.")
class CreateSDFFromMeshTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp_dir = tempfile.mkdtemp()
        cls._mesh_path = os.path.join(cls._tmp_dir, "box.obj")
        box = trimesh.primitives.Box([1.0, 1.0, 1.0])
        box.export(cls._mesh_path)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmp_dir)

    @unittest.skipIf(not vhacdx_available, "Requires vhacdx dependency.")
    def test_create_sdf_from_mesh(self):
        create_sdf_from_mesh(
            mesh_path=Path(self._mesh_path),
            mass=1.0,
            scale=1.5,
            is_compliant=True,
            hydroelastic_modulus=1e8,
            hunt_crossley_dissipation=None,
            mu_dynamic=1.0,
            mu_static=None,
            preview_with_trimesh=False,
            decomposition_method="vhacd",
        )
        self.assertTrue(
            os.path.exists(self._mesh_path.replace("obj", "sdf")),
            "SDFormat file does not exist",
        )
        mesh_pieces_dir = self._mesh_path.replace(".obj", "_parts")
        self.assertTrue(
            os.path.exists(mesh_pieces_dir), "Mesh pieces dir does not exist"
        )
        num_mesh_pieces = len(fnmatch.filter(os.listdir(mesh_pieces_dir), "*.*"))
        self.assertTrue(
            num_mesh_pieces == 1,
            f"An incorrect number of mesh pieces were created ({num_mesh_pieces} pieces)",
        )

    @unittest.skipIf(not coacd_available, "Requires coacd dependency.")
    def test_create_sdf_from_mesh_with_coacd_params(self):
        create_sdf_from_mesh(
            mesh_path=Path(self._mesh_path),
            mass=1.0,
            scale=1.5,
            is_compliant=True,
            hydroelastic_modulus=1e8,
            hunt_crossley_dissipation=None,
            mu_dynamic=1.0,
            mu_static=None,
            preview_with_trimesh=False,
            decomposition_method="coacd",
            coacd_kwargs={
                "threshold": 0.1,
                "resolution": 1000,
            },
        )
        mesh_pieces_dir = self._mesh_path.replace(".obj", "_parts")
        self.assertTrue(
            os.path.exists(mesh_pieces_dir), "Mesh pieces dir does not exist"
        )
        num_mesh_pieces = len(fnmatch.filter(os.listdir(mesh_pieces_dir), "*.*"))
        self.assertTrue(
            num_mesh_pieces == 1,
            f"An incorrect number of mesh pieces were created ({num_mesh_pieces} pieces)",
        )

    def test_create_sdf_from_mesh_with_aabb(self):
        create_sdf_from_mesh(
            mesh_path=Path(self._mesh_path),
            mass=1.0,
            scale=1.5,
            is_compliant=True,
            hydroelastic_modulus=1e8,
            hunt_crossley_dissipation=None,
            mu_dynamic=1.0,
            mu_static=None,
            preview_with_trimesh=False,
            decomposition_method="aabb",
        )
        mesh_pieces_dir = self._mesh_path.replace(".obj", "_parts")
        self.assertTrue(
            os.path.exists(mesh_pieces_dir), "Mesh pieces dir does not exist"
        )
        num_mesh_pieces = len(fnmatch.filter(os.listdir(mesh_pieces_dir), "*.*"))
        self.assertTrue(
            num_mesh_pieces == 1,
            f"An incorrect number of mesh pieces were created ({num_mesh_pieces} pieces)",
        )

    def test_create_sdf_from_mesh_invalid_decomposition(self):
        with self.assertRaises(SystemExit):
            # Test with unimplemented decomposition method (should fail)
            create_sdf_from_mesh(
                mesh_path=Path(self._mesh_path),
                mass=1.0,
                scale=1.5,
                is_compliant=True,
                hydroelastic_modulus=1e8,
                hunt_crossley_dissipation=None,
                mu_dynamic=1.0,
                mu_static=None,
                preview_with_trimesh=False,
                decomposition_method="obb",
            )


if __name__ == "__main__":
    unittest.main()
