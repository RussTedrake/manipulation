import importlib
import os
import subprocess
import sys
from urllib.request import urlretrieve


#def setup_drake(*, version, build):
#    urlretrieve(
#        f"https://drake-packages.csail.mit.edu/drake/{build}/drake-{version}/#setup_drake_colab.py",
#        "setup_drake_colab.py")
#    import setup_drake_colab
#    setup_drake_colab.setup_drake(version=version, build=build)

import importlib
import json
import os
import shutil
import subprocess
import sys
import warnings
from urllib.request import urlretrieve


def run(cmd, **kwargs):
    cp = subprocess.run(cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        universal_newlines=True, **kwargs)
    if cp.stderr:
        print(cp.stderr)
    assert cp.returncode == 0, cp


def setup_drake(*, version, build='nightly'):
    """Installs drake on Google's Colaboratory and (if necessary) adds the
    installation location to `sys.path`.  This will take approximately two
    minutes, mostly to provision the machine with drake's prerequisites, but
    the server should remain provisioned for 12 hours. Colab may ask you to
    "Reset all runtimes"; say no to save yourself the reinstall.

    Args:
        version: A string to identify which revision of drake to install.
        build: An optional string to specify the hosted directory on
            https://drake-packages.csail.mit.edu/drake/ of the build
            identified by version.  Current options are 'nightly',
            'continuous', or 'experimental'.  Default is 'nightly', which is
            recommended.

    Note: Possible version names vary depending on the build.
        - Nightly builds are versioned by date, e.g., '20200725', and the date
          represents the *morning* (not the prior evening) of the build.  You
          can also use 'latest'.
        - Continuous builds are only available with the version 'latest'.
        - (Advanced) Experimental builds use the version name
          '<timestamp>-<commit>'. See
          https://drake.mit.edu/jenkins#building-binary-packages-on-demand for
          information on obtaining a binary from an experimental branch.

    See https://drake.mit.edu/from_binary.html for more information.

    Note: If you already have pydrake installed to the target location, this
        will confirm that the build/version are the same as the installed
        version, otherwise it will overwrite the previous installation.  If you
        have pydrake available on your ``sys.path`` in a location that is
        different than the target installation, this script will throw an
        Exception to avoid possible confusion.  If you had already imported
        pydrake, this script will throw an assertion to avoid promising that we
        can successfully reload the module.
    """

    assert 'google.colab' in sys.modules, (
        "This script is intended for use on Google Colab only.")
    assert 'pydrake' not in sys.modules, (
        "You have already imported a version of pydrake.  Please choose "
        "'Restart runtime' from the menu to restart with a clean environment.")

    print(f"The drake version {version} that you've specified will be ignored.  We've switched to a `pip install` workflow for colab, and are transitioning to that.")

    # TODO(russt): Deprecate this entire workflow, but I'll put this here now for compatibilty.
    run(['pip3', 'install', 'drake'])


def setup_manipulation(*, manipulation_sha, drake_version, drake_build):
    setup_drake(version=drake_version, build=drake_build)

    path = "/opt/manipulation"

    # Clone the repo (if necessary).
    if not os.path.isdir(path):
        run([
            'git', 'clone', 'https://github.com/RussTedrake/manipulation.git',
            path
        ])

    # Checkout the sha.
    run(['git', 'checkout', '--detach', manipulation_sha], cwd=path)

    # Run install_prereqs.sh
    # TODO: Update this from /scripts/setup to /setup and remove symlink.
    run([f"{path}/scripts/setup/ubuntu/18.04/install_prereqs.sh"])

    # Run pip install
    if os.path.isfile("/opt/manipulation/colab-requirements.txt"):
        run([
            "pip3", "install", "--requirement",
            "/opt/manipulation/colab-requirements.txt"
        ])
    else:
        run([
            "pip3", "install", "--requirement",
            "/opt/manipulation/requirements.txt"
        ])
        run([
            "pip3", "install",
            "pyngrok==4.2.2",
            "pyvirtualdisplay==1.3.2"
        ])

    # Install colab specific requirements
    run(["apt", "install", "xvfb"])

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
