# -*- python -*-

# This file marks a workspace root for the Bazel build system. see
# https://bazel.build/ .

workspace(name = "manipulation")

# Load drake (recipe is from https://github.com/RobotLocomotion/drake-external-examples/blob/master/drake_bazel_installed/WORKSPACE)

# Choose which nightly build of Drake to use.
DRAKE_RELEASE = "latest"  # Can also use YYYYMMDD here, e.g., "20191026".
DRAKE_CHECKSUM = ""       # When using YYYYMMDD, best to add a checksum here.

# To use a local unpacked Drake binary release instead of an http download, set
# the INSTALLED_DRAKE_DIR environment variable to the correct path, e.g., 
# "/opt/drake".
load("//:environ.bzl", "environ_repository")
environ_repository(name = "environ", vars = ["INSTALLED_DRAKE_DIR"])
load("@environ//:environ.bzl", INSTALLED_DRAKE_DIR = "INSTALLED_DRAKE_DIR")

# This is only relevant when INSTALLED_DRAKE_DIR is set.
new_local_repository(
    name = "drake_artifacts",
    path = INSTALLED_DRAKE_DIR,
    build_file_content = "#",
) if INSTALLED_DRAKE_DIR else None

# This is only relevant when INSTALLED_DRAKE_DIR is unset.
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
http_archive(
    name = "drake_artifacts",
    url = "https://drake-packages.csail.mit.edu/drake/nightly/drake-{}-bionic.tar.gz".format(DRAKE_RELEASE),
    sha256 = DRAKE_CHECKSUM,
    strip_prefix = "drake/",
    build_file_content = "#",
) if not INSTALLED_DRAKE_DIR else None

# Load and run the repository rule that knows how to provide the @drake
# repository based on a Drake binary release.
load("@drake_artifacts//:share/drake/repo.bzl", "drake_repository")

drake_repository(name = "drake")



# TODO(russt): set MPLBACKEND=Template environment variable for tests"
# TODO(russt): add python lint tests
# TODO(russt): tests fail on drake deprecated warning
