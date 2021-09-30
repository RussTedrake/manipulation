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

FROM russtedrake/drake:meshcat_pointcloud
#FROM robotlocomotion/drake:focal
ARG DEBIAN_FRONTEND=noninteractive
ENV SHELL /bin/bash
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
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8
ENV LC_ALL en_US.UTF-8
COPY *.ipynb /root/manipulation/
COPY manipulation/ /opt/manipulation/manipulation/
COPY exercises/ /root/manipulation/exercises/
COPY requirements.txt /tmp/requirements.txt
## Delete torch requirement until I figure out how to enable it.
RUN sed -i '/torch/d' /tmp/requirements.txt
COPY setup/docker/pip.conf /root/.config/pip/pip.conf
COPY setup/ubuntu/20.04/install_prereqs.sh /tmp/
RUN /tmp/install_prereqs.sh
RUN apt -y install xvfb
RUN pip3 install -r /tmp/requirements.txt pyngrok pyvirtualdisplay
RUN rm -rf /root/.cache /root/.config
RUN rm -f /tmp/install_prereqs.sh
RUN rm -f /tmp/requirements.txt
RUN rm -f /var/cache/debconf/*-old
RUN rm -rf /var/lib/apt/lists/*
RUN rm -f /var/log/*.log
RUN rm -f /var/log/apt/*
ENV PYTHONPATH="/opt/manipulation:${PYTHONPATH}"
