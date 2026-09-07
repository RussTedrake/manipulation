from pathlib import Path

import pytest
from pydrake.multibody.parsing import PackageMap

from manipulation.station import LoadScenario, MakeHardwareStation


@pytest.mark.parametrize("driver", ["IiwaDriver", "InverseDynamicsDriver"])
def test_controller_uses_station_package_overrides(tmp_path, driver):
    models = Path(PackageMap().GetPath("drake_models"))
    for child in models.iterdir():
        (tmp_path / child.name).symlink_to(child, target_is_directory=child.is_dir())
    # This directive only exists in the override, even if drake_models is cached.
    (tmp_path / "local_iiwa.yaml").write_text(
        """directives:
- add_model:
    name: iiwa
    file: package://drake_models/iiwa_description/urdf/iiwa14_no_collision.urdf
- add_weld:
    parent: world
    child: iiwa::base
"""
    )
    scenario = LoadScenario(
        data=f"""
directives:
- add_directives:
    file: package://drake_models/local_iiwa.yaml
model_drivers:
    iiwa: !{driver} {{}}
"""
    )
    calls = []

    def preload(parser):
        calls.append(parser)
        packages = parser.package_map()
        packages.Remove("drake_models")
        packages.Add("drake_models", str(tmp_path))
        # Copying the mappings must not resolve an unused remote package.
        packages.AddRemote(
            "unused",
            PackageMap.RemoteParams(
                urls=[(tmp_path / "nonexistent.zip").as_uri()], sha256="0" * 64
            ),
        )

    station = MakeHardwareStation(scenario, parser_preload_callback=preload)
    assert len(calls) == 1
    plant = station.GetSubsystemByName("plant")
    assert plant.num_positions(plant.GetModelInstanceByName("iiwa")) == 7
