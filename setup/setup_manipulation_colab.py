import importlib
import os
import subprocess
import sys
from urllib.request import urlretrieve

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
    assert 'google.colab' in sys.modules, (
        "This script is intended for use on Google Colab only.")
    assert 'pydrake' not in sys.modules, (
        "You have already imported a version of pydrake.  Please choose "
        "'Restart runtime' from the menu to restart with a clean environment.")

    print(
        f"setup_drake() is deprecated and will be removed after 2022-03-01.  Use `pip3 install drake` instead."
    )

    run(['pip3', 'install', 'drake'])


def setup_manipulation(*, manipulation_sha, drake_version, drake_build):
    assert 'google.colab' in sys.modules, (
        "This script is intended for use on Google Colab only.")

    print(
        f"setup_manipulation() is deprecated and will be removed after 2022-03-01.  Use `pip3 install manipulation` instead."
    )

    run(['pip3', 'install', 'manipulation'])
