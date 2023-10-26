## To update poetry
```
poetry lock
poetry export --without-hashes --with dev > requirements.txt
sed 's/matplotlib==3.7.3 ; python_version >= "3.8"/matplotlib==3.5.1 ; sys_platform == "linux"\nmatplotlib==3.7.3 ; sys_platform == "darwin"/' requirements.txt > requirements.txt.tmp && mv requirements.txt.tmp requirements.txt
awk '
/torch==[0-9]+\.[0-9]+\.[0-9]+ ; python_version >= "3.8"/ {
    version=$1; sub(/^torch==/, "", version); sub(/ ;.*/, "", version);
    print "--find-links https://download.pytorch.org/whl/torch_stable.html";
    print "torch==" version "+cpu ; python_version >= \"3.8\" and sys_platform == \"linux\"";
    print "torch==" version " ; python_version >= \"3.8\" and sys_platform == \"darwin\"";
    next;
}
1' requirements.txt > requirements.txt.tmp && mv requirements.txt.tmp requirements.txt
```
One may want to also run
```
poetry install
```
- Note that the requirements.txt file is only used now for bazel and docker.
- Hopefully [direct poetry
support](https://github.com/bazelbuild/rules_python/issues/340) will land soon, or I can use [rules_python_poetry](https://github.com/AndrewGuenther/rules_python_poetry) directly; but it looks like it will still require poetry to fix [their issue](# https://github.com/python-poetry/poetry-plugin-export/issues/176).
- The awk command forces [torch to be cpu-only for bazel](https://drakedevelopers.slack.com/archives/C2PMBJVAN/p1697855405335329).

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

Update the version number in `pyproject.toml`, and the drake version, then from
the root directory, run:
```
rm -rf dist/*
poetry publish --build
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
