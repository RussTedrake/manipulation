import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from demo import DepthCameraDemoSystem

system = DepthCameraDemoSystem()

# Evaluate the camera output ports to get the images.
context = system.CreateDefaultContext()
point_cloud = system.GetOutputPort("point_cloud").Eval(context)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(point_cloud.xyzs().T)

o3d.visualization.draw_geometries([pcd])
