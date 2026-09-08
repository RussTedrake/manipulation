# Remote packages in tests

Local `pytest` runs can download Drake remote packages on demand and reuse the
normal user cache (or `XDG_CACHE_HOME`, when set).

Each CI test job runs `python -m setup.prefetch_remotes` before pytest,
using the same Python installation, user, and `XDG_CACHE_HOME=/tmp` as the tests.
The pip job prefetches using the installed manipulation wheel, matching its tests.
During pytest, `DRAKE_ALLOW_NETWORK=lcm:meshcat` disables PackageMap downloads
while allowing Meshcat and LCM. Notebook subprocesses inherit these settings.
An uncached remote package therefore fails instead of silently downloading.
This restricts Drake networking; it is not a general network sandbox for Python
or external programs.

To reproduce this policy locally, run from the repository root:

```sh
.venv/bin/python -m setup.prefetch_remotes
DRAKE_ALLOW_NETWORK=lcm:meshcat .venv/bin/python -m pytest -ra
```

If a test reports that PackageMap networking is disabled and no matching cache
entry exists, check that prefetch and pytest use the same user and cache directory.
If a test adds a new remote package, add it to `setup/prefetch_remotes.py`, which
also works with the pip job's published wheel. This list covers packages used
by tests; it deliberately excludes models such as Gymnasium Robotics that
notebooks only download in interactive mode.

See Drake's [network policy documentation](https://drake.mit.edu/doxygen_cxx/group__allow__network.html).

# CI timings

Each test job prints its 30 slowest pytest phases and uploads a JUnit report as
`pytest-<job-id>`, retained for 14 days and uploaded even on test failure when
available. Compare per-test times alongside Actions step timings. Test execution
remains serial.

Download caching was evaluated in CI, including archive restoration overhead.
Linux restoration cost more than the installation time it saved, and macOS did
not show a consistent setup improvement. The workflow therefore keeps fresh
dependency installation and model prefetch without download cache actions.

The Linux pip job installs CPU-only PyTorch and torchvision wheels from PyTorch's
CPU index. A temporary constraints file preserves those builds during the
`manipulation[all]` install, and the job verifies their build tags afterward.
This choice is confined to CI: the published dependency requirements and Poetry
lockfile are unchanged, and downstream users can continue using GPU builds.

# Menagerie conversion coverage

The default conversion test uses Panda (STL meshes, includes, and defaults) and
ANYmal B (textures and materials). It copies each model to a temporary directory,
converts its scene, checks mesh references, and loads the result with Drake.
[Drake's Menagerie tests](https://github.com/RobotLocomotion/drake/blob/master/multibody/parsing/test/detail_mujoco_parser_examples_test.cc)
exercise raw MJCF parsing, not this conversion code.

When updating the Menagerie revision in `manipulation/remotes.py`, run the full
conversion sweep locally:

```sh
TEST_ALL_MENAGERIE=1 .venv/bin/python -m pytest \
  manipulation/test/test_make_drake_compatible_model.py -k mujoco_menagerie
```

This checks conversion for all matching scenes. Asset references and Drake loading
are asserted for the two representatives; other upstream scenes can contain
dangling material references or use unsupported MJCF features. Generated files stay in temporary directories, outside the model cache.

Ubuntu Poetry CI sets `INSTALL_JUPYTER=0` when installing system prerequisites because
its Python environment supplies the notebook dependencies. The prerequisite
script still installs system Jupyter by default for local users and pip CI,
whose test harness relies on those packages. CI uses
Ubuntu's Git package and retains apt lists through setup to avoid redundant
repository setup and downloads.
