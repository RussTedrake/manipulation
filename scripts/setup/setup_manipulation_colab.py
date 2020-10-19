import importlib
import os
import subprocess
import sys
from urllib.request import urlretrieve


def setup_drake(*, version, build):
    urlretrieve(
        f"https://drake-packages.csail.mit.edu/drake/{build}/drake-{version}/setup_drake_colab.py",
        "setup_drake_colab.py")
    import setup_drake_colab
    setup_drake_colab.setup_drake(version=version, build=build)


def setup_manipulation(*, manipulation_sha, drake_version, drake_build):
    setup_drake(version=drake_version, build=drake_build)

    path = "/opt/manipulation"

    # Clone the repo (if necessary).
    if not os.path.isdir(path):
        subprocess.run([
            'git', 'clone', 'https://github.com/RussTedrake/manipulation.git',
            path
        ])

    # Checkout the sha.
    subprocess.run(['git', 'checkout', manipulation_sha], cwd=path)

    # Run install_prereqs.sh
    subprocess.run([f"{path}/scripts/setup/ubuntu/18.04/install_prereqs.sh"])

    # Run pip install
    subprocess.run([
        "pip3", "install", "--requirement", "/opt/manipulation/requirements.txt"
    ])

    # Install colab specific requirements
    subprocess.run(["apt", "install", "xvfb"])
    subprocess.run(["pip3", "install", "pyngrok", "pyvirtualdisplay"])

    # Set the path (if necessary).
    spec = importlib.util.find_spec('manipulation')
    if spec is None:
        sys.path.append(path)
        spec = importlib.util.find_spec('manipulation')

    # Confirm that we now have manipulation on the path.
    assert spec is not None, (
        "Installation failed.  find_spec('manipulation') returned None.")
    assert path in spec.origin, (
        "Installation failed.  find_spec is locating manipulation, but not "
        "in the expected path.")
