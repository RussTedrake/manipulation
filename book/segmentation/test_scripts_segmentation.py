from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_segmentation_data_script() -> None:
    cmd = [
        sys.executable,
        str(ROOT / "book/segmentation/segmentation_data.py"),
        "--test",
    ]
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    with tempfile.TemporaryDirectory(prefix="segmentation_script_") as run_cwd:
        result = subprocess.run(
            cmd, cwd=run_cwd, env=env, capture_output=True, text=True
        )
    if result.returncode != 0:
        message = [f"segmentation_data.py failed with exit code {result.returncode}"]
        if result.stdout:
            message.append(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            message.append(f"STDERR:\n{result.stderr}")
        raise AssertionError("\n\n".join(message))
