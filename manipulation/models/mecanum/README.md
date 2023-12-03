
The mecanum_base.urdf was created with a procedural generator from Calder Phillips-Grafflin at the Toyota Research Institute.

This particular model was generated with the following command (in Anzu):
```
bazel run //models/robomaster:parametric_mecanum_chassis_urdf_generator -- --wheelbase=0.2 --track=0.205 --chassis_tube_radius=0.02 --chassis_tube_mass=0.5 --axle_radius=0.02 --axle_mass=0.5 --suspension_joint_range=0.5 --hub_radius=0.045 --hub_thickness=0.04 --hub_mass=0.125 --wheel_effort_limit=1.0 --roller_length=0.06 --roller_diameter=0.015 --roller_mass=0.01 --num_rollers=12 --model_name=mecanum_base --filename=/tmp/mecanum_base.urdf
```

The joystick mappings used in Anzu are [here](https://github.com/ToyotaResearchInstitute/tri_hardware_drivers/blob/master/dji_robomaster_ep_driver/src/dji_robomaster_ep_driver/joystick_controller_mappings.cpp).

The rest of the models (for the environment) where created by Hayden Tedrake.