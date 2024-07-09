import typing
import unittest

from pydrake.all import (
    AddFrame,
    AddModel,
    AddWeld,
    IiwaDriver,
    ModelDirective,
    Rotation,
    StartMeshcat,
    Transform,
)

from manipulation.directives_tree import DirectivesTree
from manipulation.station import MakeHardwareStation, Scenario


class DirectivesTreeTest(unittest.TestCase):
    def GetFlattenedDirectives(self) -> typing.List[ModelDirective]:
        return [
            ModelDirective(
                add_model=AddModel(
                    name="iiwa",
                    file="package://drake_models/iiwa_description/urdf/iiwa14_no_collision.urdf",
                )
            ),
            ModelDirective(
                add_frame=AddFrame(
                    name="iiwa_origin",
                    X_PF=Transform(
                        base_frame="world",
                        translation=[0, 0.765, 0],
                    ),
                )
            ),
            ModelDirective(
                add_weld=AddWeld(
                    parent="iiwa_origin",
                    child="iiwa::base",
                )
            ),
            ModelDirective(
                add_model=AddModel(
                    name="wsg",
                    file="package://drake_models/wsg_50_description/sdf/schunk_wsg_50_with_tip.sdf",
                )
            ),
            ModelDirective(
                add_frame=AddFrame(
                    name="iiwa::wsg_attach",
                    X_PF=Transform(
                        base_frame="iiwa::iiwa_link_7",
                        translation=[0, 0, 0.114],
                        rotation=Rotation.Rpy(deg=[90.0, 0.0, 68.0]),
                    ),
                )
            ),
            ModelDirective(
                add_weld=AddWeld(
                    parent="iiwa::wsg_attach",
                    child="wsg::body",
                )
            ),
            ModelDirective(
                add_model=AddModel(
                    name="table",
                    file="package://drake_models/manipulation_station/table_wide.sdf",
                )
            ),
            ModelDirective(
                add_frame=AddFrame(
                    name="table_origin",
                    X_PF=Transform(
                        base_frame="world",
                        translation=[0.4, 0.3825, 0.0],
                        rotation=Rotation.Rpy(deg=[0.0, 0.0, 0.0]),
                    ),
                )
            ),
            ModelDirective(
                add_weld=AddWeld(
                    parent="table_origin",
                    child="table::table_body",
                )
            ),
        ]

    def LoadScenario(self):
        scenario = Scenario()
        scenario.directives = self.GetFlattenedDirectives()
        scenario.model_drivers = {
            "iiwa": IiwaDriver(
                control_mode="position_only",
                hand_model_name="wsg",
            )
        }
        return scenario

    def test_load_scenario(self):
        meshcat = StartMeshcat()
        scenario = self.LoadScenario()
        station = MakeHardwareStation(scenario, meshcat, hardware=False)

    def test_get_welded_descendants_and_directives(self):
        directives = self.GetFlattenedDirectives()
        tree = DirectivesTree(directives)

        children, iiwa_directives = tree.GetWeldedDescendantsAndDirectives(["iiwa"])
        self.assertEqual(children, {"wsg"})
        self.assertEqual(iiwa_directives, directives[3:6])  # wsg-related directives

    def test_get_weld_to_world_directives(self):
        directives = self.GetFlattenedDirectives()
        tree = DirectivesTree(directives)

        iiwa_directives = tree.GetWeldToWorldDirectives(["iiwa"])
        self.assertEqual(iiwa_directives, directives[:3])  # iiwa-related directives


if __name__ == "__main__":
    unittest.main()
