#!/bin/bash

# Copyright (c) 2019, Massachusetts Institute of Technology.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

set -euxo pipefail

if [[ "${EUID}" -ne 0 ]]; then
  echo 'This script must be run as root' >&2
  exit 1
fi

apt-get update -qq
apt-get install -o Dpkg::Use-Pty=0 -qy --no-install-recommends lsb-release

if [[ "$(lsb_release -cs)" != 'bionic' ]]; then
  echo 'ERROR: This script requires Ubuntu 18.04 (Bionic)' >&2
  exit 2
fi

# NOTE: Assumes that the drake (binary) install_prereqs script will also be
# run. Only install what we need to download drake and any additional
# requirements for this project.

apt-get update -qq

apt-get install -o Dpkg::Use-Pty=0 -qy --no-install-recommends $(tr '\n' ' ' <<EOF
pycodestyle
python3
python3-ipywidgets
python3-matplotlib
python3-notebook
python3-numpy
python3-pip
python3-setuptools
python3-wheel
python3-widgetsnbextension
EOF
)

pip3 install --disable-pip-version-check $(tr '\n' ' ' <<EOF
open3d
EOF
)
