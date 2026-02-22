from __future__ import annotations

import contextlib
import os
import runpy
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@contextlib.contextmanager
def _chdir(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _run(script_rel: str) -> None:
    with tempfile.TemporaryDirectory(prefix="script_cwd_") as temp_cwd:
        with _chdir(Path(temp_cwd)):
            runpy.run_path(str(ROOT / script_rel), run_name="__main__")


def test_scaling_spatial_velocity_script() -> None:
    _run("book/figures/scaling_spatial_velocity.py")


def test_two_link_singularities_script() -> None:
    _run("book/figures/two_link_singularities.py")
