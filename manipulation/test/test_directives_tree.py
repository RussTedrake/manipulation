import unittest

from pydrake.all import StartMeshcat

from manipulation.station import LoadScenario, MakeHardwareStation


class DirectivesTreeTest(unittest.TestCase):
    def test_load_scenario(self):
        scenario_data = """
directives:
    # Add IIWA
    - add_model:
        name: iiwa
        file: package://drake_models/iiwa_description/urdf/iiwa14_no_collision.urdf

    - add_frame:
        name: iiwa_origin
        X_PF:
            base_frame: world
            translation: [0, 0.765, 0]
    
    - add_weld:
        parent: iiwa_origin
        child: iiwa::base

    # Add schunk
    - add_model:
        name: wsg
        file: package://drake_models/wsg_50_description/sdf/schunk_wsg_50_with_tip.sdf

    - add_frame:
        name: iiwa::wsg_attach
        X_PF:
            base_frame: iiwa::iiwa_link_7
            translation: [0, 0, 0.114]
            rotation: !Rpy { deg: [90.0, 0.0, 68.0 ]}

    - add_weld:
        parent: iiwa::wsg_attach
        child: wsg::body

    # Add object
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
    
model_drivers:
    iiwa: !IiwaDriver
        control_mode: position_only
        hand_model_name: wsg
"""

        meshcat = StartMeshcat()
        scenario = LoadScenario(data=scenario_data)
        station = MakeHardwareStation(scenario, meshcat, hardware=False)


if __name__ == "__main__":
    unittest.main()
