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
If a test adds a new remote package, add it to `PrefetchAllRemotePackages` in
`manipulation/remotes.py`. Packages registered directly by notebooks belong in
`setup/prefetch_remotes.py`, which also works with the pip job’s published wheel.

See Drake's [network policy documentation](https://drake.mit.edu/doxygen_cxx/group__allow__network.html).
