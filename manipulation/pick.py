import numpy as np
from pydrake.all import AngleAxis, PiecewisePolynomial, PiecewisePose, RigidTransform


def MakeGripperFrames(X_G, t0=0):
    """
    Takes a partial specification with X_G["initial"], X_G["pick"], and
    X_G["place"], and returns a X_G and times with all of the pick and place
    frames populated.
    """
    # pregrasp is negative y in the gripper frame (see the figure!).
    X_GgraspGpregrasp = RigidTransform([0, -0.2, 0])

    X_G["prepick"] = X_G["pick"] @ X_GgraspGpregrasp
    X_G["preplace"] = X_G["place"] @ X_GgraspGpregrasp

    # I'll interpolate a halfway orientation by converting to axis angle and
    # halving the angle.
    X_GinitialGprepick = X_G["initial"].inverse() @ X_G["prepick"]
    angle_axis = X_GinitialGprepick.rotation().ToAngleAxis()
    X_GinitialGprepare = RigidTransform(
        AngleAxis(angle=angle_axis.angle() / 2.0, axis=angle_axis.axis()),
        X_GinitialGprepick.translation() / 2.0,
    )
    X_G["prepare"] = X_G["initial"] @ X_GinitialGprepare
    p_G = np.array(X_G["prepare"].translation())
    p_G[2] = 0.5
    # To avoid hitting the cameras, make sure the point satisfies x - y < .5
    if p_G[0] - p_G[1] < 0.5:
        scale = 0.5 / (p_G[0] - p_G[1])
        p_G[:1] /= scale
    X_G["prepare"].set_translation(p_G)

    X_GprepickGpreplace = X_G["prepick"].inverse() @ X_G["preplace"]
    angle_axis = X_GprepickGpreplace.rotation().ToAngleAxis()
    X_GprepickGclearance = RigidTransform(
        AngleAxis(angle=angle_axis.angle() / 2.0, axis=angle_axis.axis()),
        X_GprepickGpreplace.translation() / 2.0,
    )
    X_G["clearance"] = X_G["prepick"] @ X_GprepickGclearance
    p_G = np.array(X_G["clearance"].translation())
    p_G[2] = 0.5
    # To avoid hitting the cameras, make sure the point satisfies x - y < .5
    if p_G[0] - p_G[1] < 0.5:
        scale = 0.5 / (p_G[0] - p_G[1])
        p_G[:1] /= scale
    X_G["clearance"].set_translation(p_G)

    # Now let's set the timing
    times = {"initial": t0}
    prepare_time = 10.0 * np.linalg.norm(X_GinitialGprepare.translation())
    times["prepare"] = times["initial"] + prepare_time
    times["prepick"] = times["prepare"] + prepare_time
    # Allow some time for the gripper to close.
    times["pick_start"] = times["prepick"] + 2.0
    times["pick_end"] = times["pick_start"] + 2.0
    X_G["pick_start"] = X_G["pick"]
    X_G["pick_end"] = X_G["pick"]
    times["postpick"] = times["pick_end"] + 2.0
    X_G["postpick"] = X_G["prepick"]
    time_to_from_clearance = 10.0 * np.linalg.norm(X_GprepickGclearance.translation())
    times["clearance"] = times["postpick"] + time_to_from_clearance
    times["preplace"] = times["clearance"] + time_to_from_clearance
    times["place_start"] = times["preplace"] + 2.0
    times["place_end"] = times["place_start"] + 2.0
    X_G["place_start"] = X_G["place"]
    X_G["place_end"] = X_G["place"]
    times["postplace"] = times["place_end"] + 2.0
    X_G["postplace"] = X_G["preplace"]

    return X_G, times


def MakeGripperPoseTrajectory(X_G, times):
    """Constructs a gripper position trajectory from the plan "sketch"."""
    sample_times = []
    poses = []
    for name in [
        "initial",
        "prepare",
        "prepick",
        "pick_start",
        "pick_end",
        "postpick",
        "clearance",
        "preplace",
        "place_start",
        "place_end",
        "postplace",
    ]:
        sample_times.append(times[name])
        poses.append(X_G[name])

    return PiecewisePose.MakeLinear(sample_times, poses)


def MakeGripperCommandTrajectory(times):
    """Constructs a WSG command trajectory from the plan "sketch"."""
    opened = np.array([0.107])
    closed = np.array([0.0])

    traj_wsg_command = PiecewisePolynomial.FirstOrderHold(
        [times["initial"], times["pick_start"]],
        np.hstack([[opened], [opened]]),
    )
    traj_wsg_command.AppendFirstOrderSegment(times["pick_end"], closed)
    traj_wsg_command.AppendFirstOrderSegment(times["place_start"], closed)
    traj_wsg_command.AppendFirstOrderSegment(times["place_end"], opened)
    traj_wsg_command.AppendFirstOrderSegment(times["postplace"], opened)
    return traj_wsg_command
