import numpy as np
import matplotlib.pyplot as plt


def plot_planar_manipulator(q, p_ACplot):
    base = [0., 0.]
    p_AB = [np.cos(q[0]), np.sin(q[0])]
    p_AC = [
        np.cos(q[0]) + np.cos(q[0] + q[1]),
        np.sin(q[0]) + np.sin(q[0] + q[1])
    ]

    plt.figure()

    # Plot first link
    plt.plot([base[0], p_AB[0]], [base[1], p_AB[1]], 'k-', linewidth=2)
    # Plot second link
    plt.plot([p_AB[0], p_AC[0]], [p_AB[1], p_AC[1]], 'k-', linewidth=2)
    # Plot joint locations
    plt.plot(base[0], base[1], 'go')
    plt.plot(p_AB[0], p_AB[1], 'bo')
    plt.plot(p_AC[0], p_AC[1], 'ro')

    # Plot user provided position
    plt.plot(p_ACplot[0], p_ACplot[1], 'kx', markersize=15)

    # Set settings so things visualize nicely
    plt.xlim([-2.2, 2.2])
    plt.ylim([-2.2, 2.2])
    plt.gca().set_aspect('equal', adjustable='box')
