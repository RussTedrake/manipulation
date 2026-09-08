"""Install Poetry's locked dependencies with CPU PyTorch builds in Linux CI.

Restore the original manifest and lockfile even when resolution/install fails.
The caller installs the root package afterward using its original metadata.
"""

import subprocess
import sys
from pathlib import Path

import tomlkit


def main():
    paths = [Path("pyproject.toml"), Path("poetry.lock")]
    originals = {path: path.read_bytes() for path in paths}
    project = tomlkit.parse(originals[paths[0]].decode())
    locked = tomlkit.parse(originals[paths[1]].decode())
    versions = {p["name"]: p["version"] for p in locked["package"]}
    cpu_packages = {"torch", "torchvision"}
    source = tomlkit.table()
    source.update(
        name="pytorch-cpu",
        url="https://download.pytorch.org/whl/cpu",
        priority="explicit",
    )
    project["tool"]["poetry"]["source"].append(source)
    for name in cpu_packages:
        dependency = project["tool"]["poetry"]["dependencies"][name]
        dependency["version"] = versions[name].split("+")[0] + "+cpu"
        dependency["source"] = "pytorch-cpu"

    poetry = [sys.executable, "-m", "poetry"]
    try:
        paths[0].write_text(tomlkit.dumps(project))
        subprocess.run([*poetry, "lock"], check=True)
        resolved = tomlkit.parse(paths[1].read_text())
        for package in resolved["package"]:
            name = package["name"]
            if name not in cpu_packages and name in versions:
                if package["version"] != versions[name]:
                    raise RuntimeError(f"CPU override changed locked version of {name}")
        subprocess.run(
            [*poetry, "install", "--all-extras", "--with", "dev", "--no-root"],
            check=True,
        )
    finally:
        for path, content in originals.items():
            path.write_bytes(content)


if __name__ == "__main__":
    main()
