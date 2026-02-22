import unittest

from pydrake.all import (
    DiagramBuilder,
    LcmImageArrayToImages,
    LcmSubscriberSystem,
    StartMeshcat,
)

from manipulation.station import LoadScenario, MakeHardwareStation


class StationCameraIdsTest(unittest.TestCase):
    def test_load_scenario_with_camera_ids(self):
        scenario_data = """
directives:
- add_frame:
    name: camera0_origin
    X_PF:
        base_frame: world
        rotation: !Rpy { deg: [-110, 0, 90]}
        translation: [1.2, 0.0, 0.54]

- add_model:
    name: camera_main
    file: package://manipulation/camera_box.sdf
cameras:
    main_camera:
        name: camera0
        lcm_bus: default
        depth: True
        X_PB:
            base_frame: camera_main::base
camera_ids:
    main_camera: DRAKE_RGBD_CAMERA_IMAGES_810512062206
        """

        meshcat = StartMeshcat()
        meshcat.ResetRenderMode()
        builder = DiagramBuilder()
        scenario = LoadScenario(data=scenario_data)
        station = builder.AddSystem(
            MakeHardwareStation(scenario, meshcat, hardware=True)
        )

        self.assertTrue(station.HasSubsystemNamed("camera0.data_subscriber"))
        self.assertTrue(station.HasSubsystemNamed("camera0.data_receiver"))

        connection_map = station.connection_map()
        has_camera_connection = any(
            isinstance(in_sys[0], LcmImageArrayToImages)
            and isinstance(out_sys[0], LcmSubscriberSystem)
            for in_sys, out_sys in connection_map.items()
        )
        self.assertTrue(
            has_camera_connection,
            "Expected at least one connection from LcmImageArrayToImages to "
            "LcmSubscriberSystem in connection_map",
        )


if __name__ == "__main__":
    unittest.main()
