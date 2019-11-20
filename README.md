# Intelligent Robot Manipulation

_A Systems Theory Perspective on Perception, Planning, and Control_

<http://manipulation.csail.mit.edu/>

![](https://github.com/RussTedrake/manipulation/workflows/ci/badge.svg)

## To Download and Install Drake

_macOS Mojave (10.14) and macOS Catalina (10.15)_

```zsh
curl -O https://drake-packages.csail.mit.edu/drake/nightly/drake-latest-mac.tar.gz
tar -xf drake-latest-mac.tar.gz
mv drake /path/to/drake
/path/to/drake/share/drake/setup/install_prereqs
```

_Ubuntu 18.04 (Bionic)_

```bash
wget https://drake-packages.csail.mit.edu/drake/nightly/drake-latest-bionic.tar.gz
tar -xf drake-latest-bionic.tar.gz
mv drake /path/to/drake
sudo /path/to/drake/share/drake/setup/install_prereqs
```

## To Run the Unit Tests

_macOS Mojave (10.14) and macOS Catalina (10.15)_

```zsh
./setup/mac/install_prereqs.sh
mkdir build
cd build
cmake -DCMAKE_SYSTEM_PREFIX_PATH=/path/to/drake ..
ctest
```

_Ubuntu 18.04 (Bionic)_

```bash
sudo ./setup/ubuntu/install_prereqs.sh
mkdir build
cd build
cmake -DCMAKE_SYSTEM_PREFIX_PATH=/path/to/drake ..
ctest
```

## To Cite

Russ Tedrake. _Intelligent Robot Manipulation: A Systems Theory Perspective on
Perception, Planning, and Control (Course Notes for MIT 6.881)._ Downloaded on
[date] from <http://manipulation.mit.edu/>.
