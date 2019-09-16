import matplotlib.pyplot as plt
import numpy as np

import open3d as o3d

from demo import DepthCameraDemoSystem

system = DepthCameraDemoSystem()

# Evaluate the camera output ports to get the images.
context = system.CreateDefaultContext()
color_image = o3d.geometry.Image(system.GetOutputPort("color_image").Eval(context).data)
depth_image = o3d.geometry.Image(system.GetOutputPort("depth_image").Eval(context).data)
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_image, depth_image)

# Plot the two images.
plt.subplot(121)
plt.imshow(rgbd_image.color)
plt.title('Color image')
plt.subplot(122)
plt.imshow(rgbd_image.depth)
plt.title('Depth image')
plt.show()