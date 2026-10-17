import platform
import sys

import pytest
from htmlbook.ipynb_test import ipynb_test

ipynb_test("label_generation.ipynb")
ipynb_test("segmentation_and_grasp.ipynb")

# Preserve the former Bazel target's Linux x86-64 restriction.
test_label_generation = pytest.mark.skipif(
    sys.platform != "linux" or platform.machine().lower() not in {"x86_64", "amd64"},
    reason="label_generation is only supported on Linux x86-64",
)(test_label_generation)

ipynb_test("segmentation_sam.ipynb")
