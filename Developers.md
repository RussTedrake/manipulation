## To install the pre-commit hooks

```
pip3 install pre-commit
pre-commit install
```

## To Run the Unit Tests

Make sure that you have done a recursive checkout in this repository, or have run

```bash
git submodule update --init --recursive
```
Then run
```bash
bazel test //...
```

If you would like to `bazel` to use a local installation of drake, you can set
the `DRAKE_INSTALL_DIR` environment variable. Otherwise it will look in
`/opt/drake`.


## To update the pip wheels

Note: This should really only happen when drake publishes new wheels (since I'm
testing on drake master, not on the drake release).

Update the version number in `pyproject.toml`, and the drake version, then from the
root directory, run:
```
python3 -m pip install --upgrade build twine
rm -rf dist/*
python3 -m build
python3 -m twine upload dist/* -u __token__
```

## Tips for developers

These are things that I often add to my preamble of the notebook (ever since vs code broke my pythonpath importing)
```
%load_ext autoreload
%autoreload 2
import sys
sys.path.append('/home/russt/drake-install/lib/python3.6/site-packages')
sys.path.append('/home/russt/manipulation')
```
