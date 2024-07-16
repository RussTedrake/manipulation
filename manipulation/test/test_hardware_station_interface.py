import unittest

from pydrake.all import StartMeshcat

from manipulation.station import LoadScenario, MakeHardwareStation, Scenario


class HardwareStationInterfaceTest(unittest.TestCase):
    def get_scenario(self) -> Scenario:
        scenario_data = """
directives:
    # Add iiwa_left
    - add_model:
        name: iiwa_left
        file: package://drake_models/iiwa_description/urdf/iiwa14_spheres_collision.urdf

    - add_weld:
        parent: world
        child: iiwa_left::base

    # Add schunk_left
    - add_model:
        name: wsg_left
        file: package://drake_models/wsg_50_description/sdf/schunk_wsg_50_with_tip.sdf

    - add_weld:
        parent: iiwa_left::iiwa_link_7
        child: wsg_left::body
        X_PC:
            translation: [0, 0, 0.114]
            rotation: !Rpy { deg: [90.0, 0.0, 68.0 ]}

    # Add iiwa_right
    - add_model:
        name: iiwa_right
        file: package://drake_models/iiwa_description/urdf/iiwa14_no_collision.urdf

    - add_weld:
        parent: world
        child: iiwa_right::base
        X_PC:
            translation: [0, 0.765, 0]

    # Add schunk_right
    - add_model:
        name: wsg_right
        file: package://drake_models/wsg_50_description/sdf/schunk_wsg_50_with_tip.sdf

    - add_weld:
        parent: iiwa_right::iiwa_link_7
        child: wsg_right::body
        X_PC:
            translation: [0, 0, 0.114]
            rotation: !Rpy { deg: [90.0, 0.0, 68.0 ]}

    # Add table
    - add_model:
        name: table
        file: package://drake_models/manipulation_station/table_wide.sdf

    - add_frame:
        name: table_origin
        X_PF:
            base_frame: world
            translation: [0.4, 0.3825, 0.0]
            rotation: !Rpy { deg: [0., 0., 0.]}

    - add_weld:
        parent: table_origin
        child: table::table_body

lcm_buses:
    left_lcm:
        channel_suffix: _LEFT
    right_lcm:
        channel_suffix: _RIGHT

model_drivers:
    iiwa_left: !IiwaDriver
        control_mode: position_only
        hand_model_name: wsg_left
        lcm_bus: left_lcm
    iiwa_right: !IiwaDriver
        control_mode: position_only
        hand_model_name: wsg_right
        lcm_bus: right_lcm
    wsg_left: !SchunkWsgDriver {}
    wsg_right: !SchunkWsgDriver {}
"""
        return LoadScenario(data=scenario_data)

    def test_without_meshcat(self):
        scenario = self.get_scenario()
        station = MakeHardwareStation(scenario, hardware=True, meshcat=None)

        # Should not contain SceneGraph and MeshcatVisualizer.
        self.assertFalse(station.HasSubsystemNamed("scene_graph"))
        self.assertFalse(station.HasSubsystemNamed("meshcat_visualizer(illustration)"))
        self.assertFalse(station.HasSubsystemNamed("meshcat_visualizer(proximity)"))

    def test_with_meshcat(self):
        scenario = self.get_scenario()
        meshcat = StartMeshcat()
        station = MakeHardwareStation(scenario, hardware=True, meshcat=meshcat)

        # Should contain SceneGraph and MeshcatVisualizer.
        self.assertTrue(station.HasSubsystemNamed("scene_graph"))
        self.assertTrue(station.HasSubsystemNamed("meshcat_visualizer(illustration)"))
        self.assertTrue(station.HasSubsystemNamed("meshcat_visualizer(proximity)"))


if __name__ == "__main__":
    unittest.main()
