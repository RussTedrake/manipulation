import fnmatch
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import trimesh

from manipulation.create_sdf_from_mesh import create_sdf_from_mesh


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


if __name__ == "__main__":
    unittest.main()
