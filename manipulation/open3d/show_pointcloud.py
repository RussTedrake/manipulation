"""
An example of converting from the Drake point cloud representation to the
Open3D point cloud representation.
"""

import meshcat
import numpy as np
import open3d as o3d
import os

from example import DepthCameraExampleSystem
from manipulation.meshcat_utils import DrawOpen3dPointCloud

system = DepthCameraExampleSystem()
plant = system.GetSubsystemByName("plant")

# Evaluate the camera output ports to get the images.
context = system.CreateDefaultContext()
plant_context = plant.GetMyContextFromRoot(context)

v = meshcat.Visualizer()
v["/Background"].set_property("visible", False)

pcd = []
for i in range(3):
    point_cloud = system.GetOutputPort(f"point_cloud{i}").Eval(context)
    indices = np.all(np.isfinite(point_cloud.xyzs()), axis=0)

    pcd.append(o3d.geometry.PointCloud())
    pcd[i].points = o3d.utility.Vector3dVector(point_cloud.xyzs()[:, indices].T)
    pcd[i].colors = o3d.utility.Vector3dVector(point_cloud.rgbs()[:, indices].T
                                               / 255.)

    pcd[i].estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1, max_nn=30))

    camera = plant.GetBodyByName(f"camera{i}")
    X_C = plant.EvalBodyPoseInWorld(plant_context, camera)
    pcd[i].orient_normals_towards_camera_location(X_C.translation())
    DrawOpen3dPointCloud(v[f"pointcloud{i}"], pcd[i])
    v[f"pointcloud{i}"].set_property("visible", False)

# Merge point clouds.  (Note: You might need something more clever here for
# noisier point clouds; but this can often work!)
merged_pcd = pcd[0] + pcd[1] + pcd[2]
DrawOpen3dPointCloud(v["merged"], merged_pcd)
v["merged"].set_property("visible", False)

# Crop to region of interest.
cropped_pcd = merged_pcd.crop(
    o3d.geometry.AxisAlignedBoundingBox(min_bound=[-.3, -.3, -.3],
                                        max_bound=[.3, .3, .3]))
DrawOpen3dPointCloud(v["cropped"], cropped_pcd)

# Voxelize down-sample.  (Note that the normls still look reasonable)
down_sampled_pcd = cropped_pcd.voxel_down_sample(voxel_size=0.005)
DrawOpen3dPointCloud(v["down_sampled"], down_sampled_pcd, normals_scale=0.01)
# Let the normals be drawn, only turn off the object...
v["down_sampled"]["<object>"].set_property("visible", False)

show_in_open3d_visualizer = False
if show_in_open3d_visualizer and "TEST_TIMEOUT" not in os.environ:
    print("Use 'n' to show normals, and '+/-' to change their size.")
    o3d.visualization.draw_geometries([down_sampled_pcd])
