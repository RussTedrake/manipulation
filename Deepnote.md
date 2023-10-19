# For users

You should access Deepnote notebooks by following the chapter links.  To run the notebook and/or make changes, you will need to:
- Log in
- "Duplicate" the project, but clicking on the icon in the top right next to Login.

You will want to duplicate the project again whenever the staff pushes new content; you can copy over any changes from your previous duplicate.  The notebooks are versioned by the docker environment.

# For developers

Any changes to .ipynb in this repo must be "released" on Deepnote.  We have a separate Deepnote project for each chapter under the Deepnote [Manipulation team](https://deepnote.com/dashboard/Manipulation/projects).  This workflow was recommended by the Deepnote team; it allows users to duplicate small pieces at a time and for developers to version each chapter with the most recent/relevant dockerfile.

This requires:
- Pushing a new dockerfile to `russtedrake/manipulation:tagname` *from a branch of master*. Run `Deepnote.sh`.
Note that the git sha will change if I merge a branch into master.  Pegging to a branch sha is almost certainly not a stable reference.

The manipulation dockerfile builds on the drake dockerfile.  Ideally this would be `robotlocomotion/drake:jammy`.  This is updated nightly with binaries representing the master branch.  If we need changes to drake that are not yet in the nightly binaries then we can:
- [Create experimental drake binaries](https://drake.mit.edu/jenkins.html#building-packages-on-demand)
- [Create a drake dockerfile](https://github.com/RobotLocomotion/drake/tree/master/tools/install/dockerhub/jammy) with e.g.
```
docker build . -t russtedrake/drake:meshcat_contact_visualizer
docker push russtedrake/drake:meshcat_contact_visualizer
```
- Push the experimental drake dockerfile with e.g. `docker push russtedrake/drake:tagname` .


# Latex

- https://community.deepnote.com/c/ask-anything/aligning-latex-equations-in-markdown-cells

# Design notes

The current design is to have one deepnote project per notebook.  (We originally started by having one project per chapter)  Here are some of the pros and cons:
- when a student duplicates the project will reset the view back to the main notebook (I'm actually not sure how the "main" notebook is determined; probably the first one I created?).  so students have to figure out how to navigate back to an exercise notebook.
- each user only fires up one cloud instance per project.  this is good and bad.  it's faster to load consecutive notebooks, but we only get to open one port on the machine directly to meshcat, so the second notebook uses ngrok... and the free ngrok tier only allows two connections per machine.  So currently if we try to open 4 meshcat instances on deepnote, that 4th one will fail to open a public port.
- having a more granular set of projects allows us to version each notebook on a potentially different docker image.  again, good and bad.  it's more work to setup and to maintain.  but it's probably also more robust.  (e.g. I won't update a docker image, test on just the chapter, and find out later that I broke some exercise).
- students need to duplicate projects again to get any updates.  it's probably better for them to do that in narrower slices.
  