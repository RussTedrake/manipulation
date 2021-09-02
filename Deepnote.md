# For users

You should access Deepnote notebooks by following the chapter links.  To run the notebook and/or make changes, you will need to:
- Log in
- "Duplicate" the project, but clicking on the icon in the top right next to Login.

You will want to duplicate the project again whenever the staff pushes new content; you can copy over any changes from your previous duplicate.  The notebooks are versioned by the docker environment.

# For developers

Any changes to .ipynb in this repo must be "released" (manually, for now) on Deepnote.  We have a separate Deepnote project for each chapter under the Deepnote [Manipulation team](https://deepnote.com/dashboard/Manipulation/projects).  This workflow was recommended by the Deepnote team; it allows users to duplicate small pieces at a time and for developers to version each chapter with the most recent/relevant dockerfile.

This requires:
- Pushing a new dockerfile to `russtedrake/manipulation:tagname`.
  - Instructions for (manually) updating the dockerhub instance are in [setup/docker/Dockerfile](setup/docker/Dockerfile).
  - Push to tagname with the date (format `20210831`) and also to `latest`.
  - I am working on adding github integration for the docker.  See [#147](https://github.com/RussTedrake/manipulation/issues/147).
- Open the deepnote project, and
  - Update the environment dockerfile to point to the new tag.
  - Manually copy the files from `/root/manipulation` to `~/work`.
  - Make sure that the "Allow incoming connections" is enabled in the Environment settings.

The manipulation dockerfile builds on the drake dockerfile.  Ideally this would be `robotlocomotion/drake:focal`.  This is updated nightly with binaries representing the master branch.  If we need changes to drake that are not yet in the nightly binaries then we can:
- [Create experimental drake binaries](https://drake.mit.edu/jenkins.html#building-binary-packages-on-demand)
- [Create a drake dockerfile](https://github.com/RobotLocomotion/drake/tree/master/tools/install/dockerhub/focal)
- Push the experimental drake dockerfile with e.g. `docker push russtedrake/drake:tagname` .
