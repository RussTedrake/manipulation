"""Prefetch library and notebook packages using the test job's installation."""

from pydrake.multibody.parsing import PackageMap

from manipulation.remotes import PrefetchAllRemotePackages

if __name__ == "__main__":
    PrefetchAllRemotePackages()

    # book/trajectories/iris_builder.ipynb registers this package directly.
    # Keep this in the checkout so pip CI can also use older manipulation wheels.
    packages = PackageMap()
    packages.AddRemote(
        "gcs",
        PackageMap.RemoteParams(
            urls=[
                "https://github.com/mpetersen94/gcs/archive/refs/tags/arxiv_paper_version.tar.gz"
            ],
            sha256="6dd5e841c8228561b6d622f592359c36517cd3c3d5e1d3e04df74b2f5435680c",
            strip_prefix="gcs-arxiv_paper_version",
        ),
    )
    print("fetching gcs")
    packages.GetPath("gcs")
