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
