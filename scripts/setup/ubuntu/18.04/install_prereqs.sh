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
  echo 'ERROR: This script must be run as root' >&2
  exit 1
fi

if command -v conda &>/dev/null; then
  echo 'WARNING: Anaconda is NOT supported. Please remove the Anaconda bin directory from the PATH.' >&2
fi

apt-get update -qq || (sleep 15; apt-get update -qq)

apt-get install -o APT::Acquire::Retries=4 -o Dpkg::Use-Pty=0 -qy \
  --no-install-recommends lsb-release

if [[ "$(lsb_release -cs)" != 'bionic' ]]; then
  echo 'ERROR: This script requires Ubuntu 18.04 (Bionic)' >&2
  exit 2
fi

apt-get install -o APT::Acquire::Retries=4 -o Dpkg::Use-Pty=0 -qy \
  --no-install-recommends ca-certificates gnupg

APT_KEY_DONT_WARN_ON_DANGEROUS_USAGE=1 apt-key adv \
  -q --fetch-keys https://bazel.build/bazel-release.pub.gpg

echo 'deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8' \
  > /etc/apt/sources.list.d/bazel.list

apt-get update -qq || (sleep 15; apt-get update -qq)

apt-get install -o APT::Acquire::Retries=4 -o Dpkg::Use-Pty=0 -qy \
  --no-install-recommends $(cat <<EOF
bazel
jupyter
jupyter-nbconvert
jupyter-notebook
locales
python3
python3-entrypoints
python3-future
python3-ipywidgets
python3-jinja2
python3-jsonschema
python3-matplotlib
python3-numpy
python3-pandas
python3-pip
python3-retrying
python3-setuptools
python3-six
python3-snowballstemmer
python3-toolz
python3-wheel
python3-widgetsnbextension
tidy
wget
EOF
)

locale-gen en_US.UTF-8

jupyter nbextension enable --system --py widgetsnbextension

if [[ -z "${LANG:-}" && -z "${LC_ALL:-}" ]]; then
  echo 'WARNING: LANG and LC_ALL environment variables are NOT set. Please export LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8.' >&2
fi
