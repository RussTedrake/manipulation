import pytest
from pydrake.all import StartMeshcat

from manipulation.station import LoadScenario, MakeHardwareStation


@pytest.mark.parametrize("with_meshcat", [False, True])
def test_station_exports_force_ports(with_meshcat):
    meshcat = StartMeshcat() if with_meshcat else None
    station = MakeHardwareStation(LoadScenario(data="{}"), meshcat=meshcat)
    context = station.CreateDefaultContext()
    plant = station.GetSubsystemByName("plant")
    plant_context = plant.GetMyContextFromRoot(context)

    station.GetInputPort("applied_generalized_force").FixValue(context, [])
    station.GetInputPort("applied_spatial_force").FixValue(context, [])
    assert (
        plant.get_applied_generalized_force_input_port().Eval(plant_context).size == 0
    )
    assert len(plant.get_applied_spatial_force_input_port().Eval(plant_context)) == 0
