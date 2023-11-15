#!/bin/bash

# For more details see Developers.md.

poetry lock
poetry export --without-hashes --with dev > requirements.txt
sed 's/matplotlib==3.7.3 ; python_version >= "3.8"/matplotlib==3.5.1 ; sys_platform == "linux"\nmatplotlib==3.7.3 ; sys_platform == "darwin"/' requirements.txt > requirements.txt.tmp && mv requirements.txt.tmp requirements.txt
# Force torch to be cpu-only for bazel.
awk '
/torch==[0-9]+\.[0-9]+\.[0-9]+ ; python_version >= "3.8"/ {
    version=$1; sub(/^torch==/, "", version); sub(/ ;.*/, "", version);
    print "--find-links https://download.pytorch.org/whl/torch_stable.html";
    print "torch==" version "+cpu ; python_version >= \"3.8\" and sys_platform == \"linux\"";
    print "torch==" version " ; python_version >= \"3.8\" and sys_platform == \"darwin\"";
    next;
}
1' requirements.txt > requirements-bazel.txt
