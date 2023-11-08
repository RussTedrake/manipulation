
The models in this directory were obtained from https://github.com/bdaiinstitute/spot_ros2.

They were modified in the following ways:
- use package paths to reference the assets
- add collision filter groups

spot_with_arm_and_floating_base_actuators.urdf was forked from spot_with_arm, and modified in the following ways:
- changed the leg joints from revolute to fixed (at reasonable angles)
- add actuators for the arm joints (there were none!!)
- added x, y, theta floating base joints + actuators