import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight
from pydrake.all import Diagram

from manipulation.letter_generation import create_sdf_asset_from_letter


class TestHardwareStationBasic(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(10)
    @timeout_decorator.timeout(10.0)
    def test_create_IIWA14_with_hardware_station_structure(self):
        """Test basic IIWA14 HardwareStation creation and structure"""
        create_IIWA14_with_hardware_station = self.notebook_locals[
            "create_IIWA14_with_hardware_station"
        ]

        diagram = create_IIWA14_with_hardware_station()

        # Test return type
        self.assertIsInstance(diagram, Diagram, "Should return a Diagram")

        # Get the HardwareStation from the diagram
        systems = [diagram.GetSystems()[i] for i in range(len(diagram.GetSystems()))]
        self.assertEqual(
            len(systems),
            1,
            "Diagram should contain exactly one system (HardwareStation)",
        )

        station = systems[0]

        # Get the plant from the station and check model instances
        plant = station.GetSubsystemByName("plant")
        num_model_instances = plant.num_model_instances()
        # Should have: world, iiwa (2 total)
        self.assertGreaterEqual(
            num_model_instances,
            2,
            "Should have at least 2 model instances (world + 1 robot)",
        )


class TestHardwareStationBimanual(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(10)
    @timeout_decorator.timeout(15.0)
    def test_create_bimanual_IIWA14_structure(self):
        """Test bimanual IIWA14 HardwareStation creation and structure"""
        create_bimanual_IIWA14_with_hardware_station = self.notebook_locals[
            "create_bimanual_IIWA14_with_hardware_station"
        ]

        diagram = create_bimanual_IIWA14_with_hardware_station()

        # Test return type
        self.assertIsInstance(diagram, Diagram, "Should return a Diagram")

        # Get the HardwareStation from the diagram
        systems = [diagram.GetSystems()[i] for i in range(len(diagram.GetSystems()))]
        self.assertEqual(
            len(systems),
            1,
            "Diagram should contain exactly one system (HardwareStation)",
        )

        station = systems[0]

        # Get the plant from the station and check model instances
        plant = station.GetSubsystemByName("plant")
        num_model_instances = plant.num_model_instances()
        # Should have: world, iiwa_left, iiwa_right (3 total)
        self.assertGreaterEqual(
            num_model_instances,
            3,
            "Should have at least 3 model instances (world + 2 robots)",
        )


class TestHardwareStationFullSystem(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @classmethod
    def setUpClass(cls):
        cls._tmp_dir = tempfile.mkdtemp()
        cls._assets_generated = False

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmp_dir)

    def _setup_assets_once(self):
        """Generate assets once per test class instance"""
        if not self.__class__._assets_generated:
            # Generate test letters
            test_initials = "BPG"
            assets_dir = Path(self._tmp_dir) / "assets"

            for letter in test_initials:
                create_sdf_asset_from_letter(
                    text=letter,
                    font_name="Arial",
                    letter_height_meters=0.2,
                    extrusion_depth_meters=0.07,
                    output_dir=str(assets_dir / f"{letter}_model"),
                )

            # Create table SDF file
            table_sdf = self.notebook_locals["table_sdf"]
            table_sdf_path = assets_dir / "table.sdf"
            table_sdf_path.parent.mkdir(parents=True, exist_ok=True)
            with open(table_sdf_path, "w", encoding="utf-8") as f:
                f.write(table_sdf)

            self.__class__._assets_generated = True

    @weight(15)
    @timeout_decorator.timeout(30.0)
    def test_create_bimanual_with_assets_structure(self):
        """Test creation of full system with table and initials"""
        create_bimanual_IIWA14_with_table_and_initials_and_assets = (
            self.notebook_locals[
                "create_bimanual_IIWA14_with_table_and_initials_and_assets"
            ]
        )

        # Generate assets once
        self._setup_assets_once()

        try:
            original_cwd = os.getcwd()
            os.chdir(self._tmp_dir)

            diagram, station = (
                create_bimanual_IIWA14_with_table_and_initials_and_assets()
            )

            # Test return types
            self.assertIsInstance(diagram, Diagram, "Should return a Diagram")
            self.assertIsNotNone(station, "Should return a station")

            # Get the plant from the station and check model instances
            plant = station.GetSubsystemByName("plant")
            num_model_instances = plant.num_model_instances()
            # Should have: world, iiwa_left, iiwa_right, table, B_letter, P_letter, G_letter, mustard_bottle (8 total)
            self.assertGreaterEqual(
                num_model_instances, 8, "Should have at least 8 model instances"
            )

        finally:
            os.chdir(original_cwd)

    @weight(15)
    @timeout_decorator.timeout(45.0)
    def test_simulate_full_system_execution(self):
        """Test that simulate_full_system function runs without errors"""
        simulate_full_system = self.notebook_locals["simulate_full_system"]
        RigidTransform = self.notebook_locals["RigidTransform"]

        # Generate assets once
        self._setup_assets_once()

        try:
            original_cwd = os.getcwd()
            os.chdir(self._tmp_dir)

            simulate_full_system(
                iiwa_left_q0=np.array([0, 0, 0, 0, 0, 0, 0]),
                iiwa_right_q0=np.array([0, 0, 0, 0, 0, 0, 0]),
                letter_poses=[
                    RigidTransform([0.7, 0.0, 1.0]),
                    RigidTransform([0.9, 0.0, 1.0]),
                    RigidTransform([1.1, 0.0, 1.0]),
                ],
                object_poses=[
                    RigidTransform([0.5, 0.0, 0.75]),
                ],
                simulation_time=0.1,
                use_realtime=False,
            )
            # Test passes if no exception was raised
        except Exception as e:
            self.fail(f"simulate_full_system raised an exception: {e}")
        finally:
            os.chdir(original_cwd)
