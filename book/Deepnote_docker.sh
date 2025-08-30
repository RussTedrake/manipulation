#!/bin/bash

set -e

docker pull robotlocomotion/drake:noble
docker buildx build --platform linux/amd64,linux/arm64 -f setup/docker/Dockerfile -t russtedrake/manipulation:latest --push .
# Note: This could require `docker login`
docker buildx build --platform linux/amd64,linux/arm64 -f setup/docker/Dockerfile -t russtedrake/manipulation:$(git rev-parse --short HEAD) --push .
git rev-parse --short HEAD > book/Deepnote_docker_sha.txt
echo "Remember to run Deepnote.sh to actually push to Deepnote"
echo "Remember to log on to deepnote and build the dockerfile in any one of the notebooks"
echo "https://deepnote.com/workspace/$(cat book/Deepnote_workspace.txt)"
