import matplotlib.pyplot as plt
import numpy as np


def plot_planar_manipulator(q: np.ndarray, p_ACplot: np.ndarray) -> None:
    """
    Plot manipulator with joint positions q and the end-effector location
    p_ACplot.

    The intended use includes the following:

      - plot_planar_manipulator(q, forward_kinematics(q))
      - plot_planar_manipulator(inverse_kinematics(p), p)

    which helps verify the correctness of the implementations.

    """
    base = [0.0, 0.0]
    p_AB = [np.cos(q[0]), np.sin(q[0])]
    p_AC = [
        np.cos(q[0]) + np.cos(q[0] + q[1]),
        np.sin(q[0]) + np.sin(q[0] + q[1]),
    ]

    plt.figure()

    # Plot first link
    plt.plot([base[0], p_AB[0]], [base[1], p_AB[1]], "k-", linewidth=2)
    # Plot second link
    plt.plot([p_AB[0], p_AC[0]], [p_AB[1], p_AC[1]], "k-", linewidth=2)
    # Plot joint locations
    plt.plot(base[0], base[1], "go")
    plt.plot(p_AB[0], p_AB[1], "bo")
    plt.plot(p_AC[0], p_AC[1], "ro")

    # Plot user provided position
    plt.plot(p_ACplot[0], p_ACplot[1], "kx", markersize=15)

    # Set settings so things visualize nicely
    plt.xlim([-2.2, 2.2])
    plt.ylim([-2.2, 2.2])
    plt.gca().set_aspect("equal", adjustable="box")


def plot_two_planar_manipulators(
    q1: np.ndarray, q2: np.ndarray, p_ACplot: np.ndarray
) -> None:
    """
    Plot two planr manipulators with joint positions q1, q2

    The intented use is the following:

    - q1, q2 = inverse_kinematics(x, y)
    - plot_two_planar_manipulators(q1, q2, np.array([x, y]))

    It is meant to verify the correctness of the inverse kinematics solution

    """
    base = [0.0, 0.0]
    p_AB1 = [np.cos(q1[0]), np.sin(q1[0])]
    p_AC1 = [
        np.cos(q1[0]) + np.cos(q1[0] + q1[1]),
        np.sin(q1[0]) + np.sin(q1[0] + q1[1]),
    ]

    p_AB2 = [np.cos(q2[0]), np.sin(q2[0])]
    p_AC2 = [
        np.cos(q2[0]) + np.cos(q2[0] + q2[1]),
        np.sin(q2[0]) + np.sin(q2[0] + q2[1]),
    ]

    plt.figure()

    # Plot first arm
    plt.plot([base[0], p_AB1[0]], [base[1], p_AB1[1]], "k-", linewidth=2)
    plt.plot([p_AB1[0], p_AC1[0]], [p_AB1[1], p_AC1[1]], "k-", linewidth=2)
    # Plot joint locations
    plt.plot(base[0], base[1], "go")
    plt.plot(p_AB1[0], p_AB1[1], "bo")
    plt.plot(p_AC1[0], p_AC1[1], "ro")

    # Plot second arm
    plt.plot([base[0], p_AB2[0]], [base[1], p_AB2[1]], "m-", linewidth=2)
    plt.plot([p_AB2[0], p_AC2[0]], [p_AB2[1], p_AC2[1]], "m-", linewidth=2)
    # Plot joint locations
    plt.plot(base[0], base[1], "go")
    plt.plot(p_AB2[0], p_AB2[1], "bo")
    plt.plot(p_AC2[0], p_AC2[1], "ro")

    # Plot user provided position
    plt.plot(p_ACplot[0], p_ACplot[1], "kx", markersize=15)

    # Set settings so things visualize nicely
    plt.xlim([-2.2, 2.2])
    plt.ylim([-2.2, 2.2])
    plt.gca().set_aspect("equal", adjustable="box")
