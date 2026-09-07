#!/bin/bash

set -e

docker pull robotlocomotion/drake:noble
docker buildx build --platform linux/amd64,linux/arm64 -f setup/docker/Dockerfile -t russtedrake/manipulation:latest --push .
# Note: This could require `docker login`
docker buildx build --platform linux/amd64,linux/arm64 -f setup/docker/Dockerfile -t russtedrake/manipulation:$(git rev-parse --short HEAD) --push .
