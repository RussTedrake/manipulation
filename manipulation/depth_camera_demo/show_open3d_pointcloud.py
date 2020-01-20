import numpy as np
import open3d as o3d

from demo import DepthCameraDemoSystem

system = DepthCameraDemoSystem()

# Evaluate the camera output ports to get the images.
context = system.CreateDefaultContext()
point_cloud = system.GetOutputPort("point_cloud").Eval(context)

indices = np.all(np.isfinite(point_cloud.xyzs()), axis=0)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud.xyzs()[:, indices].T)
pcd.colors = o3d.utility.Vector3dVector(
    point_cloud.rgbs()[:, indices].T / 255.)

o3d.visualization.draw_geometries([pcd])
