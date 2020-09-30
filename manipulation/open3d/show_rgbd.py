"""
An example of converting from the Drake RGBD image representation to the
Open3D RGBD image representation.
"""

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from example import DepthCameraExampleSystem


def create_open3d_rgbd_image(color_image, depth_image):
    color_image = o3d.geometry.Image(np.copy(
        color_image.data[:, :, :3]))  # No alpha
    depth_image = o3d.geometry.Image(np.squeeze(np.copy(depth_image.data)))
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=color_image,
        depth=depth_image,
        depth_scale=1.0,
        depth_trunc=3.0,
        convert_rgb_to_intensity=False)
    return rgbd_image


system = DepthCameraExampleSystem()

# Evaluate the camera output ports to get the images.
context = system.CreateDefaultContext()
color_image = system.GetOutputPort("color_image").Eval(context)
depth_image = system.GetOutputPort("depth_image").Eval(context)
rgbd_image = create_open3d_rgbd_image(color_image, depth_image)

# Plot the two images.
plt.subplot(121)
plt.imshow(rgbd_image.color)
plt.title('Color image')
plt.subplot(122)
plt.imshow(rgbd_image.depth)
plt.title('Depth image')
plt.show()
