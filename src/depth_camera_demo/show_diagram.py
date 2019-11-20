import matplotlib.pyplot as plt

from pydrake.systems.drawing import plot_system_graphviz

from demo import DepthCameraDemoSystem

system = DepthCameraDemoSystem()

plot_system_graphviz(system)
plt.show()
