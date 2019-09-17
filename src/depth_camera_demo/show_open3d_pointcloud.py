import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from demo import DepthCameraDemoSystem

system = DepthCameraDemoSystem()

# Evaluate the camera output ports to get the images.
context = system.CreateDefaultContext()
point_cloud = system.GetOutputPort("point_cloud").Eval(context)

from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

indices = np.isfinite(np.sum(point_cloud.xyzs(), axis=0))
ax.scatter(point_cloud.xyzs()[0,indices], point_cloud.xyzs()[1,indices], point_cloud.xyzs()[2,indices], c=point_cloud.rgbs()[:, indices])
ax.set_xlim(-.1,.1)
ax.set_ylim(-.1,.1)
ax.set_zlim(.5+-.1,.5+.1)
plt.show()

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud.xyzs()[:,indices].T)
pcd.colors = o3d.utility.Vector3dVector(point_cloud.rgbs()[:,indices].T)

o3d.visualization.draw_geometries([pcd])
