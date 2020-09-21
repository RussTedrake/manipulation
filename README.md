# Robot Manipulation

_Perception, Planning, and Control_

<http://manipulation.csail.mit.edu/>

![](https://github.com/RussTedrake/manipulation/workflows/CI/badge.svg)

## Installation

Please follow the installation instructions in drake.html.  Make sure that you have done a recursive checkout in this repository, or have run

```bash
git submodule update --init --recursive
```

## To Run the Unit Tests

_macOS Mojave (10.14) and macOS Catalina (10.15)_

```zsh
bazel test //...
```

_Ubuntu 18.04 (Bionic)_

```bash
bazel test //...
```

## To Cite

Russ Tedrake. _Robot Manipulation: Perception, Planning, and Control (Course
Notes for MIT 6.881)._ Downloaded on [date] from <http://manipulation.mit.edu/>.


## Notes / tips / tricks

In most notebooks I launch a single meshcat server (as a subprocess) in the first cell.  Especially when I'm working in VS Code, I end up with an accumulation of meshcat server instances hanging around.  Running an occasional
```bash
pkill -f meshcat
```
will clean them up.
