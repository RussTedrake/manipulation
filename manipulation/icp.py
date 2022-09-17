import numpy as np
import open3d as o3d
from pydrake.all import RigidTransform, RotationMatrix


def PoseEstimationGivenCorrespondences(p_Om, p_s, chat):
    """Returns optimal X_O given the correspondences"""
    # Apply correspondences, and transpose data to support numpy broadcasting
    p_Omc = p_Om[:, chat].T
    p_s = p_s.T

    # Calculate the central points
    p_Ombar = p_Omc.mean(axis=0)
    p_sbar = p_s.mean(axis=0)

    # Calculate the "error" terms, and form the data matrix
    merr = p_Omc - p_Ombar
    serr = p_s - p_sbar
    W = np.matmul(serr.T, merr)

    # Compute R
    U, Sigma, Vt = np.linalg.svd(W)
    R = np.matmul(U, Vt)
    if np.linalg.det(R) < 0:
        print("fixing improper rotation")
        Vt[-1, :] *= -1
        R = np.matmul(U, Vt)

    # Compute p
    p = p_sbar - np.matmul(R, p_Ombar)

    return RigidTransform(RotationMatrix(R), p)


def FindClosestPoints(point_cloud_A, point_cloud_B):
    """
    Finds the nearest (Euclidean) neighbor in point_cloud_B for each
    point in point_cloud_A.
    @param point_cloud_A A 3xN numpy array of points.
    @param point_cloud_B A 3xN numpy array of points.
    @return indices An (N, ) numpy array of the indices in point_cloud_B of each
        point_cloud_A point's nearest neighbor.
    """
    indices = np.empty(point_cloud_A.shape[1], dtype=int)

    # TODO(russt): Replace this with a direct call to flann
    # https://pypi.org/project/flann/
    kdtree = o3d.geometry.KDTreeFlann(point_cloud_B)
    for i in range(point_cloud_A.shape[1]):
        nn = kdtree.search_knn_vector_3d(point_cloud_A[:, i], 1)
        indices[i] = nn[1][0]

    return indices


def PrintResults(X_O, Xhat_O):
    p = X_O.translation()
    aa = X_O.rotation().ToAngleAxis()
    print(f"True position: {p}")
    print(f"True orientation: {aa}")
    p = Xhat_O.translation()
    aa = Xhat_O.rotation().ToAngleAxis()
    print(f"Estimated position: {p}")
    print(f"Estimated orientation: {aa}")


def IterativeClosestPoint(p_Om,
                          p_s,
                          X_O=None,
                          meshcat=None,
                          meshcat_scene_path=None):
    Xhat = RigidTransform()
    Nm = p_s.shape[1]
    chat_previous = np.zeros(
        Nm) - 1  # Set chat to a value that FindClosePoints will never return.

    while True:
        chat = FindClosestPoints(p_s, Xhat.multiply(p_Om))
        if np.array_equal(chat, chat_previous):
            # Then I've converged.
            break
        chat_previous = chat
        Xhat = PoseEstimationGivenCorrespondences(p_Om, p_s, chat)
        if meshcat_scene_path:
            meshcat.SetTransform(meshcat_scene_path, Xhat.inverse())

    if X_O:
        PrintResults(X_O, Xhat)

    return Xhat, chat
