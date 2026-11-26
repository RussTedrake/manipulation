"""Prefetch library and notebook packages using the test job's installation.

CI runs `python -m setup.prefetch_remotes` from the repository root before
pytest; see docs/testing.md. In the pip job, this imports the installed wheel.
"""

import sys
from pathlib import Path

from pydrake.multibody.parsing import PackageMap

from manipulation.remotes import AddMujocoMenagerie, AddRby1Remote, AddSpotRemote

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "book"))
from htmlbook.drake_remotes import prefetch_remotes  # noqa: E402

if __name__ == "__main__":
    packages = PackageMap()
    remotes = ["drake_models"]
    # Only prefetch packages used by tests. In particular, the Gymnasium
    # Robotics notebook deliberately skips its remote models during CI.
    remotes += [AddMujocoMenagerie(packages), AddSpotRemote(packages)]
    try:
        # This downloads immediately, to convert the model for Drake.
        remotes.append(AddRby1Remote(packages))
    except RuntimeError as error:
        # The package is registered before the download; retry it below.
        print(f"fetching rby1 failed: {error}", flush=True)
        remotes.append("rby1")

    # book/trajectories/iris_builder.ipynb registers this package directly.
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
    remotes.append("gcs")
    prefetch_remotes(packages, remotes)
