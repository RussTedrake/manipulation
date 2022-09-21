import numpy as np
import open3d as o3d

from pydrake.all import BaseField, Fields, PointCloud


def drake_cloud_to_open3d(point_cloud):
    indices = np.all(np.isfinite(point_cloud.xyzs()), axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud.xyzs()[:, indices].T)

    if point_cloud.has_rgbs():
        pcd.colors = o3d.utility.Vector3dVector(point_cloud.rgbs()[:, indices].T
                                                / 255.)

    # TODO(russt): add normals (and test) if needed


#    if point_cloud.has_normals():
#        pcd.normals = o3d.uility.Vector3dVector(
#            point_cloud.normals()[:, indices].T)

    return pcd


def open3d_cloud_to_drake(cloud):
    fields = BaseField.kXYZs
    if cloud.has_colors():
        fields |= BaseField.kRGBs

    drake_cloud = PointCloud(new_size=len(cloud.points), fields=Fields(fields))
    drake_cloud.mutable_xyzs()[:] = np.asarray(cloud.points).T

    if cloud.has_colors():
        drake_cloud.mutable_rgbs()[:] = np.asarray(cloud.colors).T

    return drake_cloud


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


def draw_open3d_point_cloud(meshcat,
                            path,
                            pcd,
                            normals_scale=0.0,
                            point_size=0.001):
    pts = np.asarray(pcd.points)
    if pcd.has_colors():
        cloud = PointCloud(pts.shape[0],
                           Fields(BaseField.kXYZs | BaseField.kRGBs))
        cloud.mutable_rgbs()[:] = 255 * np.asarray(pcd.colors).T
    else:
        cloud = PointCloud(pts.shape[0], Fields(BaseField.kXYZs))
    cloud.mutable_xyzs()[:] = pts.T
    meshcat.SetObject(path, cloud, point_size=point_size)
    if pcd.has_normals() and normals_scale > 0.0:
        assert ('need to implement LineSegments in meshcat c++')
        normals = np.asarray(pcd.normals)
        vertices = np.hstack(
            (pts, pts + normals_scale * normals)).reshape(-1, 3).T
        meshcat["normals"].set_object(
            g.LineSegments(g.PointsGeometry(vertices),
                           g.MeshBasicMaterial(color=0x000000)))
