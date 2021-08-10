# -*- mode: dockerfile -*-
# vi: set ft=dockerfile :

# Copyright 2020 Massachusetts Institute of Technology.
# Licensed under the BSD 3-Clause License. See LICENSE.TXT for details.

# cd $(git rev-parse --show-toplevel)
# docker build -f setup/docker/Dockerfile -t russtedrake/manipulation:latest .
#
# docker run -i -p 7000:7000 -p 8888:8888 -t -w /root/manipulation russtedrake/manipulation:latest
#
# docker login
# docker push russtedrake/manipulation:latest
#
# To run w/ entire directory mounted (useful e.g. for running requirements.sh):
# docker run -i -p 7000:7000 -p 8888:8888 -t -v $(pwd):/root/repo_mount -w /root/repo_mount russtedrake/manipulation:latest

FROM robotlocomotion/drake:focal
ARG DEBIAN_FRONTEND=noninteractive
ENV SHELL /bin/bash
EXPOSE 7000/tcp
EXPOSE 8888/tcp
LABEL org.opencontainers.image.authors="Russ Tedrake"
LABEL org.opencontainers.image.description="Perception, Planning, and Control"
LABEL org.opencontainers.image.licenses="BSD-3-Clause"
LABEL org.opencontainers.image.source="https://github.com/RussTedrake/manipulation.git"
LABEL org.opencontainers.image.title="Robot Manipulation"
LABEL org.opencontainers.image.url="https://manipulation.mit.edu/"
LABEL org.opencontainers.image.vendor="Massachusetts Institute of Technology"
COPY *.ipynb /root/manipulation/
COPY manipulation/ /opt/manipulation/
COPY exercises/ /root/manipulation/exercises/
COPY requirements.txt /tmp/requirements.txt
## Delete torch requirement until I figure out how to enable it.
RUN sed -i '/torch/d' /tmp/requirements.txt
COPY setup/docker/pip.conf /root/.config/pip/pip.conf
COPY setup/ubuntu/20.04/install_prereqs.sh /tmp/
RUN /tmp/install_prereqs.sh
RUN LC_ALL=en_US.UTF-8 pip3 install -r /tmp/requirements.txt
RUN rm -rf /root/.cache /root/.config
RUN rm -f /tmp/install_prereqs.sh
RUN rm -f /tmp/requirements.txt
RUN rm -f /var/cache/debconf/*-old
RUN rm -rf /var/lib/apt/lists/*
RUN rm -f /var/log/*.log
RUN rm -f /var/log/apt/*
RUN export PYTHONPATH=/opt/manipulation:${PYTHONPATH}
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8
ENV LC_ALL en_US.UTF-8
# Make python and pip available to meet the requirements for deepnote.com:
# https://docs.deepnote.com/environment/custom-environments#custom-image-requirements
# TODO: Remove these pending drake #15586.
RUN ln -s /usr/bin/python3 /opt/drake/bin/python
RUN ln -s /usr/bin/pip3 /opt/drake/bin/pip
