import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest


@pytest.mark.parametrize(
    "eigenvalues, expected_title",
    [
        ([0.0, 0.0], None),
        ([1.0, 2.0], "Local Minima"),
        ([-1.0, -2.0], "Local Maxima"),
        ([1.0, -2.0], "Saddle Point"),
        ([-1.0, 2.0], "Saddle Point"),
    ],
)
def test_visualization_terminates(eigenvalues, expected_title, capsys):
    notebook = (
        Path(__file__).resolve().parents[2]
        / "book/clutter/exercises/analytic_antipodal_grasps.ipynb"
    )
    cells = json.loads(notebook.read_text())["cells"]
    source = "".join(
        next(c for c in cells if c["metadata"].get("id") == "MF16oJya8-WM")["source"]
    )
    calls = 0

    def find_antipodal_pts(shape):
        nonlocal calls
        calls += 1
        # Fail promptly if a visualization search becomes unbounded again.
        assert calls <= 300
        return np.array([1.0, 2.0]), np.array(eigenvalues)

    plt = Mock()
    plotted_titles = []
    namespace = dict(
        running_as_notebook=True,
        np=np,
        plt=plt,
        shape=Mock(),
        plot_gear=Mock(),
        find_antipodal_pts=find_antipodal_pts,
        plot_antipodal_pts=lambda *args: plotted_titles.append(
            plt.title.call_args.args[0]
        ),
    )
    random_state = np.random.get_state()
    try:
        exec(source, namespace)
    finally:
        np.random.set_state(random_state)

    assert plotted_titles == ([] if expected_title is None else [expected_title])
    output = capsys.readouterr().out
    assert output.count("No matching grasp found") == (
        3 if expected_title is None else 2
    )
