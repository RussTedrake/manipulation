import os
import sys
from urllib.request import urlretrieve

import numpy as np
from IPython import get_ipython
from pydrake.common import GetDrakePath
from pydrake.common.containers import namedview
from pydrake.geometry.render import RenderLabel
from pydrake.multibody.tree import JointIndex

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


# TODO(russt): promote these to drake (and make a version with model_instance)


def MakeNamedViewPositions(
    mbp, view_name, add_suffix_if_single_position=False
):
    names = [None] * mbp.num_positions()
    for ind in range(mbp.num_joints()):
        joint = mbp.get_joint(JointIndex(ind))
        if joint.num_positions() == 1 and not add_suffix_if_single_position:
            names[joint.position_start()] = joint.name()
        else:
            for i in range(joint.num_positions()):
                names[
                    joint.position_start() + i
                ] = f"{joint.name()}_{joint.position_suffix(i)}"
    for ind in mbp.GetFloatingBaseBodies():
        body = mbp.get_body(ind)
        start = body.floating_positions_start()
        for i in range(7):
            names[
                start + i
            ] = f"{body.name()}_{body.floating_position_suffix(i)}"
    return namedview(view_name, names)


def MakeNamedViewVelocities(
    mbp, view_name, add_suffix_if_single_velocity=False
):
    names = [None] * mbp.num_velocities()
    for ind in range(mbp.num_joints()):
        joint = mbp.get_joint(JointIndex(ind))
        if joint.num_velocities() == 1 and not add_suffix_if_single_velocity:
            names[joint.velocity_start()] = joint.name()
        else:
            for i in range(joint.num_velocities()):
                names[
                    joint.velocity_start() + i
                ] = f"{joint.name()}_{joint.velocity_suffix(i)}"
    for ind in mbp.GetFloatingBaseBodies():
        body = mbp.get_body(ind)
        start = body.floating_velocities_start() - mbp.num_positions()
        for i in range(6):
            names[
                start + i
            ] = f"{body.name()}_{body.floating_velocity_suffix(i)}"
    return namedview(view_name, names)


def MakeNamedViewState(mbp, view_name):
    # TODO: this could become a nested named view, pending
    # https://github.com/RobotLocomotion/drake/pull/14973
    pview = MakeNamedViewPositions(mbp, f"{view_name}_pos", True)
    vview = MakeNamedViewVelocities(mbp, f"{view_name}_vel", True)
    return namedview(view_name, pview.get_fields() + vview.get_fields())


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
