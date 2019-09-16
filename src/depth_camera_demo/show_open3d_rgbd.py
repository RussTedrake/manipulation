import matplotlib.pyplot as plt
import numpy as np

from demo import DepthCameraDemoSystem, create_open3d_rgbd_image

system = DepthCameraDemoSystem()

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

#point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, )
