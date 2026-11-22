"""Check that every notebook in the book is executed by some test.

A notebook is covered when an `ipynb_test(...)` call in a `test_*.py` file in
the notebook's directory registers it. Exercise notebooks are generated from
the private solutions checkout, so a registration in
`solutions/notebooks/<chapter>/` also covers `book/<chapter>/exercises/`; that
check is skipped when the checkout is absent.

Notebooks that cannot run in CI must be listed in `UNTESTED` with a reason, so
that disabling a test leaves a visible record.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
BOOK = REPO_ROOT / "book"
SOLUTIONS_NOTEBOOKS = REPO_ROOT / "solutions" / "notebooks"

# Paths are relative to book/.
UNTESTED = {
    "mobile/robocasa.ipynb": "Downloads the RoboCasa kitchen asset archives.",
    "segmentation/segmentation_inference.ipynb": (
        "Needs the clutter_maskrcnn_data model and images, which CI does not fetch."
    ),
    "segmentation/segmentation_train.ipynb": (
        "Colab only: shells out to pip and git for the torchvision reference "
        "scripts, then trains Mask R-CNN."
    ),
}


def _book_notebooks() -> list[Path]:
    """Tracked notebooks in book/, excluding submodules."""
    output = subprocess.check_output(
        ["git", "ls-files", "--", "*.ipynb"], cwd=BOOK, text=True
    )
    return sorted(BOOK / line for line in output.splitlines())


def _registered_notebooks(directory: Path) -> set[str]:
    """Notebook names registered via ipynb_test() in directory/test_*.py."""
    names = set()
    for test_file in directory.glob("test_*.py"):
        tree = ast.parse(test_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "ipynb_test"
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                names.add(node.args[0].value)
    return names


def _is_exercise(notebook: Path) -> bool:
    return notebook.parent.name == "exercises"


def _publicly_covered(notebook: Path) -> bool:
    return notebook.name in _registered_notebooks(notebook.parent)


def _relative(notebook: Path) -> str:
    return notebook.relative_to(BOOK).as_posix()


def test_book_notebooks_are_tested() -> None:
    missing = [
        _relative(nb)
        for nb in _book_notebooks()
        if not _is_exercise(nb)
        and not _publicly_covered(nb)
        and _relative(nb) not in UNTESTED
    ]
    assert not missing, (
        "These notebooks are not run by any test. Register each one with "
        "ipynb_test() in a test_*.py file in its directory, or add it to "
        f"UNTESTED in {Path(__file__).name} with the reason it cannot run:\n  "
        + "\n  ".join(missing)
    )


def test_untested_list_is_current() -> None:
    notebooks = {_relative(nb): nb for nb in _book_notebooks()}
    stale = []
    for path in UNTESTED:
        if path not in notebooks:
            stale.append(f"{path}: no such tracked notebook")
        elif _publicly_covered(notebooks[path]):
            stale.append(f"{path}: now registered with ipynb_test()")
    assert not stale, "Remove these entries from UNTESTED:\n  " + "\n  ".join(stale)


@pytest.mark.skipif(
    not SOLUTIONS_NOTEBOOKS.is_dir(), reason="needs the solutions checkout"
)
def test_exercise_notebooks_are_tested() -> None:
    missing = []
    for nb in _book_notebooks():
        if not _is_exercise(nb) or _publicly_covered(nb):
            continue
        chapter = nb.parent.parent.name
        if nb.name not in _registered_notebooks(SOLUTIONS_NOTEBOOKS / chapter):
            missing.append(_relative(nb))
    assert not missing, (
        "These exercise notebooks are not run by any public or solutions test. "
        "Register each one with ipynb_test() in "
        "solutions/notebooks/<chapter>/test_*.py:\n  " + "\n  ".join(missing)
    )
