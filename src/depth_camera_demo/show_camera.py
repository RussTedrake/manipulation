import matplotlib.pyplot as plt
import numpy as np

from demo import DepthCameraDemoSystem

system = DepthCameraDemoSystem()

# Evaluate the camera output ports to get the images.
context = system.CreateDefaultContext()
color_image = system.GetOutputPort("color_image").Eval(context)
depth_image = system.GetOutputPort("depth_image").Eval(context)

# Plot the two images.
plt.subplot(121)
plt.imshow(color_image.data)
plt.title('Color image')
plt.subplot(122)
plt.imshow(np.squeeze(depth_image.data))
plt.title('Depth image')
plt.show()
