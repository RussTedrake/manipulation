# Robot Manipulation

_Perception, Planning, and Control_

<http://manipulation.csail.mit.edu/>

![](https://github.com/RussTedrake/manipulation/workflows/CI/badge.svg)

## Installation

Please follow the installation instructions in drake.html.  Make sure that you have done a recursive checkout in this repository, or have run

```bash
git submodule update --init --recursive
```

## To Run the Unit Tests

_macOS Mojave (10.14) and macOS Catalina (10.15)_

```zsh
bazel test //...
```

_Ubuntu 18.04 (Bionic)_

```bash
bazel test //...
```

## To Cite

Russ Tedrake. _Robot Manipulation: Perception, Planning, and Control (Course
Notes for MIT 6.881)._ Downloaded on [date] from <http://manipulation.mit.edu/>.


## Notes / tips / tricks

In most notebooks I launch a single meshcat server (as a subprocess) in the first cell.  Especially when I'm working in VS Code, I end up with an accumulation of meshcat server instances hanging around.  Running an occasional
```bash
pkill -f meshcat
```
will clean them up.


## For lecturing

My current (proposed) solution:
- OBS on Ubuntu
  - Connect ipad via this Airplay recevier on Ubuntu: https://github.com/antimof/UxPlay
  - Share with Zoom via "Windowed Projector (Source)".  Drag the window to make it big (full resolution) and to fit the content nicely.  Then I can minimize it. 
  - Google Hangouts meet speaker
  - Brio for video (through OBS?  or directly?)

I've tried a number of solutions.
- OBS
  - Never worked well for me on mac; at least not on the macbook air.  It might have been starved of resources.
  - I got things working beautifully on ubuntu, and could push to YouTube at a nice screen quality.
  - The sharing to Zoom, both with the virtual camera and the desktop sharing, resulted in a dramatic loss of resolution.  To the point where it was unusable.  Update: Seems to be resolved with version 26.0.2
- Mmhmm
  - Again, resolution of the screen share with zoom was a bottleneck
- Zoom (only) on mac
  - Worked for me for a while.  Painful switch between ipad and slides.  Fan is always on and loud; the machine sounds like it's dying.  Active dis-incentive to have meshcat open.
  - Airplay sharing of ipad stopped working when I upgraded zoom to 5.3.1 .  I spent some time trying to downgrade but gave up.
  - Have been sharing ipad via usb, with a usb3 splitter sharing a port between Brio and ipad.
- Zoom (only) on Ubuntu.
  - Brio video
  - Google Hangouts meet speaker
  - Connect ipad via this Airplay recevier on Ubuntu: https://github.com/antimof/UxPlay
  - Couldn't find the gstreamer window as an option for window sharing.