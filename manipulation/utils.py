import os
import sys
from datetime import date
from pathlib import Path
from urllib.request import urlretrieve
from warnings import warn

import numpy as np
from IPython import get_ipython
from pydrake.all import GetDrakePath
from pydrake.common import GetDrakePath
from pydrake.geometry import RenderLabel

# Use a global variable here because some calls to IPython will actually case an
# interpreter to be created.  This file needs to be imported BEFORE that
# happens.
running_as_notebook = (
    "COLAB_TESTING" not in os.environ
    and get_ipython()
    and hasattr(get_ipython(), "kernel")
)

running_as_test = False


def set_running_as_test(value):
    global running_as_test
    running_as_test = value


def pyplot_is_interactive():
    # import needs to happen after the backend is set.
    import matplotlib.pyplot as plt
    from matplotlib.rcsetup import interactive_bk

    return plt.get_backend() in interactive_bk


def FindResource(filename):
    return os.path.join(os.path.dirname(__file__), filename)


# A filename from the data directory.  Will download if necessary.
def LoadDataResource(filename):
    data = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    if not os.path.exists(data):
        os.makedirs(data)
    path = os.path.join(data, filename)
    if not os.path.exists(path):
        print(f"{path} was not found locally; downloading it now...")
        urlretrieve(
            f"https://manipulation.csail.mit.edu/data/{filename}", path
        )
    return path


def ConfigureParser(parser):
    """Add the manipulation/package.xml index to the given Parser."""
    package_xml = os.path.join(os.path.dirname(__file__), "models/package.xml")
    parser.package_map().AddPackageXml(filename=package_xml)
    AddPackagePaths(parser)


def AddPackagePaths(parser):
    # Remove once https://github.com/RobotLocomotion/drake/issues/10531 lands.
    parser.package_map().PopulateFromFolder(FindResource(""))
    parser.package_map().Add(
        "manipulation_station",
        os.path.join(
            GetDrakePath(),
            "examples/manipulation_station/models",
        ),
    )
    parser.package_map().Add(
        "ycb",
        os.path.join(GetDrakePath(), "manipulation/models/ycb"),
    )
    parser.package_map().Add(
        "wsg_50_description",
        os.path.join(
            GetDrakePath(),
            "manipulation/models/wsg_50_description",
        ),
    )


reserved_labels = [
    RenderLabel.kDoNotRender,
    RenderLabel.kDontCare,
    RenderLabel.kEmpty,
    RenderLabel.kUnspecified,
]


def colorize_labels(image):
    # import needs to happen after backend is set up.
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    """Colorizes labels."""
    # TODO(eric.cousineau): Revive and use Kuni's palette.
    cc = mpl.colors.ColorConverter()
    color_cycle = plt.rcParams["axes.prop_cycle"]
    colors = np.array([cc.to_rgb(c["color"]) for c in color_cycle])
    bg_color = [0, 0, 0]
    image = np.squeeze(image)
    background = np.zeros(image.shape[:2], dtype=bool)
    for label in reserved_labels:
        background |= image == int(label)
    image[np.logical_not(background)]
    color_image = colors[image % len(colors)]
    color_image[background] = bg_color
    return color_image


def SetupMatplotlibBackend(wishlist=["notebook"]):
    """
    Helper to support multiple workflows:
        1) nominal -- running locally w/ jupyter notebook
        2) unit tests (no ipython, backend is template)
        3) binder -- does have notebook backend
        4) colab -- claims to have notebook, but it doesn't work
    Puts the matplotlib backend into notebook mode, if possible,
    otherwise falls back to inline mode.
    Returns True iff the final backend is interactive.
    """
    # To find available backends, one can access the lists:
    # matplotlib.rcsetup.interactive_bk
    # matplotlib.rcsetup.non_interactive_bk
    # matplotlib.rcsetup.all_backends
    if running_as_notebook:
        ipython = get_ipython()
        # Short-circuit for google colab.
        if "google.colab" in sys.modules:
            ipython.run_line_magic("matplotlib", "inline")
            return False
        # TODO: Find a way to detect vscode, and use inline instead of notebook
        for backend in wishlist:
            try:
                ipython.run_line_magic("matplotlib", backend)
                return pyplot_is_interactive()
            except KeyError:
                continue
        ipython.run_line_magic("matplotlib", "inline")
    return False


def DrakeVersionGreaterThan(minimum_date: date):
    drake_version_txt = (
        Path(GetDrakePath()).parent / "doc" / "drake" / "VERSION.TXT"
    )
    # If the file doesn't exist, then we should pass. A source install won't
    # have VERSION.TXT
    if drake_version_txt.is_file():
        with open(drake_version_txt, "r") as f:
            drake_version = f.read()
        drake_version = drake_version.split()[0]
        if len(drake_version) == 14:
            drake_date = date(
                year=int(drake_version[:4]),
                month=int(drake_version[4:6]),
                day=int(drake_version[6:8]),
            )
        elif drake_version == "1.13.0":
            drake_date = date(year=2023, month=2, day=14)
        elif drake_version == "1.14.0":
            drake_date = date(year=2023, month=3, day=15)
        else:
            warn(f"Unrecognized drake version {drake_version}")
            return
        if drake_date < minimum_date:
            raise (
                RuntimeError(
                    f"You need to update your Drake version. Python is using the Drake installation in {GetDrakePath()}. This installation was from a nightly build on {drake_date}, but this method requires Drake from at least {minimum_date}."
                )
            )


# Adapted from Drake's system_doxygen.py.  Just make an html rendering of the
# system block with its name and input/output ports (even if it is a Diagram).
def SystemHtml(system):
    input_port_html = ""
    for p in range(system.num_input_ports()):
        input_port_html += (
            f'<tr><td align=right style="padding:5px 0px 5px 0px">'
            f"{system.get_input_port(p).get_name()} &rarr;</td></tr>"
        )
    output_port_html = ""
    for p in range(system.num_output_ports()):
        output_port_html += (
            '<tr><td align=left style="padding:5px 0px 5px 0px">'
            f"&rarr; {system.get_output_port(p).get_name()}</td></tr>"
        )
    # Note: keeping this on a single line avoids having to handle comment line
    # markers (e.g. * or ///)
    html = (
        f"<table align=center cellpadding=0 cellspacing=0><tr align=center>"
        f'<td style="vertical-align:middle">'
        f"<table cellspacing=0 cellpadding=0>{input_port_html}</table>"
        f"</td>"
        f'<td align=center style="border:2px solid black;padding-left:20px;'
        f'padding-right:20px;vertical-align:middle" bgcolor=#F0F0F0>'
        f"{system.get_name()}</td>"
        f'<td style="vertical-align:middle">'
        f"<table cellspacing=0 cellpadding=0>{output_port_html}</table>"
        f"</td></tr></table>"
    )

    return html
