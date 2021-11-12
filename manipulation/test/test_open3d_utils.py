"""
An example of converting from the Drake RGBD image representation to the
Open3D RGBD image representation.
"""

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from pydrake.all import Meshcat
from manipulation.mustard_depth_camera_example import MustardExampleSystem
from manipulation.meshcat_cpp_utils import draw_open3d_point_cloud
from manipulation.open3d_utils import *

system = MustardExampleSystem()

# Evaluate the camera output ports to get the images.
context = system.CreateDefaultContext()

meshcat = Meshcat()

for i in range(3):
    color_image = system.GetOutputPort(f"camera{i}_rgb_image").Eval(context)
    depth_image = system.GetOutputPort(f"camera{i}_depth_image").Eval(context)
    rgbd_image = create_open3d_rgbd_image(color_image, depth_image)

    # Plot the two images.
    plt.subplot(3, 2, 2 * i + 1)
    plt.imshow(rgbd_image.color)
    plt.title('Color image')
    plt.subplot(3, 2, 2 * i + 2)
    plt.imshow(rgbd_image.depth)
    plt.title('Depth image')

    point_cloud = system.GetOutputPort(f"camera{i}_point_cloud").Eval(context)
    pcd = create_open3d_point_cloud(point_cloud)
    draw_open3d_point_cloud(meshcat, f"camera{i}_point_cloud", pcd)

plt.show()
