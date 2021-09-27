# Robot Manipulation

_Perception, Planning, and Control_

<http://manipulation.csail.mit.edu/>

![](https://github.com/RussTedrake/manipulation/workflows/CI/badge.svg)

## To Cite

Russ Tedrake. _Robot Manipulation: Perception, Planning, and Control (Course
Notes for MIT 6.800/6.843)._ Downloaded on [date] from <http://manipulation.mit.edu/>.

## Installation

Please follow the installation instructions in `drake.html`.  Make sure that you have done a recursive checkout in this repository, or have run

```bash
git submodule update --init --recursive
```

## To Run the Unit Tests

```
bazel test //...
```

If you would like to `bazel` to use a local installation of drake, you can set
the `DRAKE_INSTALL_DIR` environment variable. Otherwise it will look in
`/opt/drake`.

## Tips for developers

If you find yourself making modifications to the supporting .py files in addition to the notebook, then adding 
```
%load_ext autoreload
%autoreload 2
```
to the top of the notebook will force those changes to get reloaded automatically.