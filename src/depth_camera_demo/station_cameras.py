import matplotlib.pyplot as plt
import numpy as np

from demo import DepthCameraDemoSystem  # needed to get the image types, somehow.  TODO: clean this up.

from pydrake.examples.manipulation_station import ManipulationStation, ManipulationStationHardwareInterface
from pydrake.systems.drawing import plot_system_graphviz

station = ManipulationStation()
station.SetupManipulationClassStation()
#station.SetupClutterClearingStation()
station.Finalize()
context = station.CreateDefaultContext()

camera_names = station.get_camera_names()
index = 1
for name in camera_names:
    color_image = station.GetOutputPort("camera_" + name + "_rgb_image").Eval(context)
    depth_image = station.GetOutputPort("camera_" + name + "_depth_image").Eval(context)

    plt.subplot(len(camera_names), 2, index)
    plt.imshow(color_image.data)
    index += 1
    plt.title('Color image')
    plt.subplot(len(camera_names), 2, index)
    plt.imshow(np.squeeze(depth_image.data))
    index += 1
    plt.title('Depth image')

plt.show()
