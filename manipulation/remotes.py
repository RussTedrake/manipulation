from pydrake.multibody.parsing import PackageMap

from manipulation.make_drake_compatible_model import MakeDrakeCompatibleModel


def AddMujocoMenagerie(package_map: PackageMap) -> str:
    """Add the remote `mujoco_menagerie` package to the given PackageMap.
    https://github.com/google-deepmind/mujoco_menagerie"""
    package_name = "mujoco_menagerie"
    package_map.AddRemote(
        package_name=package_name,
        params=PackageMap.RemoteParams(
            # This repository doesn't have up-to-date tags/releases; the scary
            # hash in the url is the most recent commit sha at the time of my
            # writing.
            urls=[
                f"https://github.com/google-deepmind/mujoco_menagerie/archive/469893211c41d5da9c314f5ab58059fa17c8e360.tar.gz"
            ],
            sha256=("1cfe0ebde2c6dd80405977e0b3a6f72e1b062d8a79f9f0437ebebe463c9c85f7"),
            strip_prefix="mujoco_menagerie-469893211c41d5da9c314f5ab58059fa17c8e360/",
        ),
    )
    return package_name


def AddSpotRemote(package_map: PackageMap) -> str:
    """Add the remote `spot_description` package to the given PackageMap.
    https://github.com/rai-opensource/spot_description"""
    package_name = "spot_description"
    package_map.AddRemote(
        package_name=package_name,
        params=PackageMap.RemoteParams(
            # Asset submodule revision used by rai-opensource/spot_ros2 at
            # fc0eb245d47e146cf2a4c4fa47c9739f89bc79bd.
            urls=[
                "https://github.com/rai-opensource/spot_description/archive/156d1802bfb117f219dbfce7597d283d5fabc968.tar.gz"
            ],
            sha256=("ab68ae01f4ae23e63b00a3efb6dcf2cf68d1bf19d0f2f0e79a0cb3012fcc476f"),
            strip_prefix="spot_description-156d1802bfb117f219dbfce7597d283d5fabc968/spot_description/",
        ),
    )
    return package_name


def AddGymnasiumRobotics(package_map: PackageMap) -> str:
    """Add the remote `gymnasium_robotics` package to the given PackageMap.
    https://github.com/google-deepmind/gymnasium_robotics"""
    package_name = "gymnasium_robotics"
    package_map.AddRemote(
        package_name=package_name,
        params=PackageMap.RemoteParams(
            urls=[
                f"https://github.com/Farama-Foundation/Gymnasium-Robotics/archive/refs/tags/v1.3.1.tar.gz"
            ],
            sha256=("d274b3ee1d34337aa35d4686447fda6a6d20dfcb4d375eab91b4e49b1108afde"),
            strip_prefix="Gymnasium-Robotics-1.3.1/gymnasium_robotics/envs/assets/",
        ),
    )
    return package_name


def AddRby1Remote(package_map: PackageMap):
    package_name = "rby1"
    package_map.AddRemote(
        package_name=package_name,
        params=PackageMap.RemoteParams(
            urls=[
                f"https://github.com/RainbowRobotics/rby1-sdk/archive/refs/tags/v0.3.0.tar.gz"
            ],
            sha256=("f1cbebccc24ad2cb3d966a4816adc8394572949e2b41809b9dc1c6756e501f48"),
            strip_prefix="rby1-sdk-0.3.0/models/rby1a/",
        ),
    )
    model_path = package_map.ResolveUrl("package://rby1/urdf/model.urdf")
    drake_model_path = model_path.replace(".urdf", ".drake.urdf")
    try:
        MakeDrakeCompatibleModel(model_path, drake_model_path)
    except ImportError as e:
        print(f"RBY1 model conversion failed; models may not visualize properly: {e}")
    return package_name


def PrefetchAllRemotePackages():
    """Prefetch all remote packages in the given PackageMap.
    This is useful for CI, where the remote packages are downloaded as part of
    the build process.
    """
    package_map = PackageMap()

    def fetch(package_name):
        print(f"fetching {package_name}")
        package_map.GetPath(package_name)

    fetch("drake_models")
    fetch(AddMujocoMenagerie(package_map))
    fetch(AddSpotRemote(package_map))
    fetch(AddGymnasiumRobotics(package_map))
    fetch(AddRby1Remote(package_map))
