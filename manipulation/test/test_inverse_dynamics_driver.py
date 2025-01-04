import unittest

import numpy as np
from pydrake.all import Simulator

from manipulation.station import LoadScenario, MakeHardwareStation


class InverseDynamicsDriverTest(unittest.TestCase):
    def get_bimanual_directives_data(self):
        return """
directives:
- add_model:
    name: iiwa_left
    file: package://drake_models/iiwa_description/urdf/iiwa14_no_collision.urdf
- add_frame:
    name: iiwa_left_origin
    X_PF:
        base_frame: world
        translation: [0, 0.765, 0]
- add_weld:
    parent: iiwa_left_origin
    child: iiwa_left::base
- add_model:
    name: wsg_left
    file: package://drake_models/wsg_50_description/sdf/schunk_wsg_50_with_tip.sdf
- add_frame:
    name: iiwa_left::wsg_attach
    X_PF:
        base_frame: iiwa_left::iiwa_link_7
        translation: [0, 0, 0.114]
        rotation: !Rpy { deg: [90, 0, 68]}
- add_weld:
    parent: iiwa_left::iiwa_link_7
    child: wsg_left::body
    X_PC:
        translation: [0, 0, 0.09]
        rotation: !Rpy { deg: [90, 0, 90]}
- add_model:
    name: iiwa_right
    file: package://drake_models/iiwa_description/urdf/iiwa14_no_collision.urdf
- add_frame:
    name: iiwa_right_origin
    X_PF:
        base_frame: world
        translation: [0, -0.765, 0]
- add_weld:
    parent: iiwa_right_origin
    child: iiwa_right::base
- add_model:
    name: wsg_right
    file: package://drake_models/wsg_50_description/sdf/schunk_wsg_50_with_tip.sdf
- add_frame:
    name: iiwa_right::wsg_attach
    X_PF:
        base_frame: iiwa_right::iiwa_link_7
        translation: [0, 0, 0.114]
        rotation: !Rpy { deg: [90, 0, 68]}
- add_weld:
    parent: iiwa_right::iiwa_link_7
    child: wsg_right::body
    X_PC:
        translation: [0, 0, 0.09]
        rotation: !Rpy { deg: [90, 0, 90]}
        """

    def make_controller_test(self, model_name: str, model_dof: int):
        directives_data = self.get_bimanual_directives_data()
        model_drivers_data = f"""
model_drivers:
    {model_name}: !InverseDynamicsDriver {{}}
        """
        scenario = LoadScenario(data=directives_data + model_drivers_data)
        station = MakeHardwareStation(scenario)

        simulator = Simulator(station)
        context = simulator.get_mutable_context()
        fixed_value = np.zeros(2 * model_dof)
        station.GetInputPort(f"{model_name}.desired_state").FixValue(
            context, fixed_value
        )
        simulator.AdvanceTo(1.0)

    def test_controller_on_single_model(self):
        self.make_controller_test(
            model_name="iiwa_left",
            model_dof=7,
        )

    def test_controller_on_two_models(self):
        self.make_controller_test(
            model_name="iiwa_left+wsg_left",
            model_dof=9,
        )

    def test_controller_on_three_models(self):
        self.make_controller_test(
            model_name="iiwa_left+wsg_left+iiwa_right",
            model_dof=16,
        )

    def test_controller_on_all_models(self):
        self.make_controller_test(
            model_name="iiwa_left+iiwa_right+wsg_left+wsg_right",
            model_dof=18,
        )


if __name__ == "__main__":
    unittest.main()
