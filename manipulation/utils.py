import os
from datetime import date
from pathlib import Path
from urllib.request import urlretrieve
from warnings import warn

import numpy as np
import pydot
from IPython import get_ipython
from IPython.display import SVG, display
from pydrake.all import GetDrakePath
from pydrake.common import GetDrakePath
from pydrake.geometry import RenderLabel
from pydrake.multibody.parsing import Parser
from pydrake.systems.framework import System
from pydrake.systems.sensors import ImageLabel16I

# Use a global variable here because some calls to IPython will actually cause
# an interpreter to be created.  This file needs to be imported BEFORE that
# happens.
running_as_notebook = (
    "COLAB_TESTING" not in os.environ
    and get_ipython()
    and hasattr(get_ipython(), "kernel")
)

running_as_test = False


def _set_running_as_test(value: bool):
    """[INTERNAL USE ONLY]: Set the global variable `running_as_test` to
    `value`.

    This method is used by the build system; it is not intended for general
    use.
    """
    global running_as_test
    running_as_test = value


def FindResource(filename: str):
    """Returns the absolute path to the given filename relative to the
    manipulation module."""
    return os.path.join(os.path.dirname(__file__), filename)


def LoadDataResource(filename: str):
    warn("`LoadDataResource` is deprecated. Use `FindDataResource` instead.")
    FindDataResource(filename)


def FindDataResource(filename: str):
    """
    Returns the absolute path to the given filename relative to the book data
    directory; fetching it from a remote host if necessary.
    """
    if "MANIPULATION_DATA_DIR" in os.environ:
        data = os.environ["MANIPULATION_DATA_DIR"]
    else:
        data = os.path.join(os.path.dirname(os.path.dirname(__file__)), "book/data")
        if not os.path.exists(data):
            os.makedirs(data)
    path = os.path.join(data, filename)
    if not os.path.exists(path):
        if running_as_test:
            raise FileNotFoundError(
                f"{path} was not found locally; it is required for testing."
            )  # because the urlretrieve defeats bazel's hermetic testing.
        print(f"{path} was not found locally; downloading it now...")
        urlretrieve(f"https://manipulation.csail.mit.edu/data/{filename}", path)
    return path


def ConfigureParser(parser: Parser):
    """Add the `manipulation` module packages to the given Parser."""
    package_xml = os.path.join(os.path.dirname(__file__), "models/package.xml")
    parser.package_map().AddPackageXml(filename=package_xml)


def colorize_labels(image: ImageLabel16I):
    """Given a label image, replace the integer labels with color values that display nicely in matplotlib."""

    # import needs to happen after backend is set up.
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    reserved_labels = [
        RenderLabel.kDoNotRender,
        RenderLabel.kDontCare,
        RenderLabel.kEmpty,
        RenderLabel.kUnspecified,
    ]

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


def DrakeVersionGreaterThan(minimum_date: date):
    """Check that the Drake version is at least `minimum_data`."""
    drake_version_txt = Path(GetDrakePath()).parent / "doc" / "drake" / "VERSION.TXT"
    version_dates = {
        "1.13.0": date(year=2023, month=2, day=14),
        "1.14.0": date(year=2023, month=3, day=15),
        "1.15.0": date(year=2023, month=4, day=18),
        "1.16.0": date(year=2023, month=5, day=18),
        "1.17.0": date(year=2023, month=5, day=23),
        "1.18.0": date(year=2023, month=6, day=20),
        "1.19.0": date(year=2023, month=7, day=13),
        "1.20.0": date(year=2023, month=8, day=16),
        "1.21.0": date(year=2023, month=9, day=14),
        "1.22.0": date(year=2023, month=10, day=16),
        "1.23.0": date(year=2023, month=11, day=17),
        "1.24.0": date(year=2023, month=12, day=18),
    }
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
        elif drake_version in version_dates:
            drake_date = version_dates[drake_version]
        else:
            warn(f"Unrecognized drake version {drake_version}")
            return
        if drake_date < minimum_date:
            raise (
                RuntimeError(
                    f"You need to update your Drake version. Python is using the Drake installation in {GetDrakePath()}. This installation was from a nightly build on {drake_date}, but this method requires Drake from at least {minimum_date}."
                )
            )


def RenderDiagram(system: System, max_depth: int | None = None):
    """Use pydot to render the GraphViz diagram of the given system.

    Args:
        system (System): The Drake system (or diagram) to render.
        max_depth (int, optional): Sets a limit to the depth of nested diagrams
            to visualize. Use zero to render a diagram as a single system
            block. Defaults to 1.
    """
    display(
        SVG(
            pydot.graph_from_dot_data(system.GetGraphvizString(max_depth=max_depth))[
                0
            ].create_svg()
        )
    )


# Adapted from Drake's system_doxygen.py.  Just make an html rendering of the
# system block with its name and input/output ports (even if it is a Diagram).
def SystemHtml(system: System):
    """Generates an HTML string for `system` of the style seen in the Drake
    doxygen documentation."""
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
