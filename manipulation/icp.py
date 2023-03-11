import numpy as np
from pydrake.all import PointCloud, Rgba, RigidTransform, RotationMatrix
from scipy.spatial import KDTree


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


def FindClosestPoints(kdtree_A, point_cloud_B):
    """
    Finds the nearest (Euclidean) neighbor in kdtree_A for each
    point in point_cloud_B.
    @param kdtree_A A scipy KDTree containing point cloud A
    @param point_cloud_B A 3xN numpy array of points.
    @return indices An (N, ) numpy array of the indices in point_cloud_A of each
        point_cloud_B point's nearest neighbor.
    """
    indices = np.empty(point_cloud_B.shape[1], dtype=int)

    for i in range(point_cloud_B.shape[1]):
        distance, indices[i] = kdtree_A.query(point_cloud_B[:, i], k=1)

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


def IterativeClosestPoint(
    p_Om,
    p_Ws,
    X_Ohat=RigidTransform(),
    X_O=None,
    meshcat=None,
    meshcat_scene_path=None,
    max_iterations=None,
):
    """
    Implements the vanilla ICP algorithm corresponding all scene points to a
    model point.
    @param p_Om A 3xN numpy array of "model" points in the model/object frame.
    @param p_Ws A 3xN numpy array of "scene" points in the world frame.
    @param X_Ohat An RigidTransform() containing the initial guess for X_O.
    @param X_O The true, known, X_O; if provided, then the method will print a
               comparison of the results.
    @param meshcat a Meshcat instance for visualizing the point clouds.
    @param meshcat_scene_path A string under which the point clouds will be
                              published.
    @return X_WOhat The estimated pose of the model in the world frame.
    @return chat The indices of correspondences (and index into the model
                 points) for each scene point.
    """
    Nm = p_Ws.shape[1]
    # Set chat to a value that FindClosestPoints will never return.
    chat_previous = np.zeros(Nm) - 1

    kdtree_Om = KDTree(p_Om.T, copy_data=True)

    if meshcat_scene_path:
        cloud = PointCloud(p_Om.shape[1])
        cloud.mutable_xyzs()[:] = p_Om
        meshcat.SetObject(meshcat_scene_path, cloud, rgba=Rgba(0, 0, 1, 1))

    iterations = 0
    while True:
        chat = FindClosestPoints(kdtree_Om, X_Ohat.inverse() @ p_Ws)
        if np.array_equal(chat, chat_previous):
            # Then I've converged.
            break
        chat_previous = chat
        X_Ohat = PoseEstimationGivenCorrespondences(p_Om, p_Ws, chat)
        if meshcat_scene_path:
            meshcat.SetTransform(meshcat_scene_path, X_Ohat)
        iterations += 1
        if max_iterations and iterations >= max_iterations:
            break

    if X_O:
        PrintResults(X_O, X_Ohat)

    return X_Ohat, chat
