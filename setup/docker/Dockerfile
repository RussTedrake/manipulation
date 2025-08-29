# -*- mode: dockerfile -*-
# vi: set ft=dockerfile :

# Copyright 2020 Massachusetts Institute of Technology.
# Licensed under the BSD 3-Clause License. See LICENSE.TXT for details.

# See Deepnote_docker.sh for usage.
#
# IMPORTANT: This image must be built for multiple platforms to support CI:
# docker buildx build --platform linux/amd64,linux/arm64 -t russtedrake/manipulation:latest --push .

FROM robotlocomotion/drake:noble
ARG DEBIAN_FRONTEND=noninteractive
ENV SHELL=/bin/bash
# Open ports for meshcat:
EXPOSE 7000-7099/tcp
EXPOSE 8888/tcp
LABEL org.opencontainers.image.authors="Russ Tedrake"
LABEL org.opencontainers.image.description="Perception, Planning, and Control"
LABEL org.opencontainers.image.licenses="BSD-3-Clause"
LABEL org.opencontainers.image.source="https://github.com/RussTedrake/manipulation.git"
LABEL org.opencontainers.image.title="Robot Manipulation"
LABEL org.opencontainers.image.url="https://manipulation.mit.edu/"
LABEL org.opencontainers.image.vendor="Massachusetts Institute of Technology"
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
COPY requirements.txt /tmp/requirements.txt
COPY setup/docker/pip.conf /root/.config/pip/pip.conf
COPY setup/ubuntu/24.04/install_prereqs.sh /tmp/
RUN /tmp/install_prereqs.sh
RUN apt-get -q update && apt-get -q install -y --no-install-recommends \
    curl git nginx-light python-is-python3 python3-pip python3-venv xvfb \
    && apt-get remove -y python3-notebook && apt-get autoremove -y && apt-get -q clean
# We still pip install the requirements, because otherwise deepnote will
# install old versions of e.g. scipy which cannot be upgraded in the
# Init notebook.  Drake is also installed in /opt/drake, but is not used (since 
# I've not added it to the path / pythonpath).
RUN pip3 install -r /tmp/requirements.txt pyvirtualdisplay manipulation[all] --ignore-installed --break-system-packages
# Bake model data into the image up-front, instead of fetching on demand.
RUN python3 -c 'from pydrake.all import PackageMap; PackageMap().GetPath("drake_models")'
RUN python3 -c 'from manipulation.prefetch_remotes import PrefetchAllRemotePackages; PrefetchAllRemotePackages()'

RUN rm -rf /root/.cache /root/.config
RUN rm -f /tmp/install_prereqs.sh
RUN rm -f /tmp/requirements.txt
RUN rm -f /var/cache/debconf/*-old
RUN rm -rf /var/lib/apt/lists/*
RUN rm -f /var/log/*.log
RUN rm -f /var/log/apt/*
