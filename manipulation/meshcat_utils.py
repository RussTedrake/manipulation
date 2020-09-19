import numpy as np

import meshcat.geometry as g
import meshcat.transformations as tf


def plot_surface(meshcat, X, Y, Z, color=0xdd9999, wireframe=False):
    (rows, cols) = Z.shape

    vertices = np.empty((rows * cols, 3), dtype=np.float32)
    vertices[:, 0] = X.reshape((-1))
    vertices[:, 1] = Y.reshape((-1))
    vertices[:, 2] = Z.reshape((-1))

    # Vectorized faces code from https://stackoverflow.com/questions/44934631/making-grid-triangular-mesh-quickly-with-numpy  # noqa
    faces = np.empty((cols - 1, rows - 1, 2, 3), dtype=np.uint32)
    r = np.arange(rows * cols).reshape(cols, rows)
    faces[:, :, 0, 0] = r[:-1, :-1]
    faces[:, :, 1, 0] = r[:-1, 1:]
    faces[:, :, 0, 1] = r[:-1, 1:]
    faces[:, :, 1, 1] = r[1:, 1:]
    faces[:, :, :, 2] = r[1:, :-1, None]
    faces.shape = (-1, 3)
    meshcat.set_object(g.TriangularMeshGeometry(vertices, faces),
                       g.MeshLambertMaterial(color=color, wireframe=wireframe))


def plot_mathematical_program(meshcat, prog, X, Y, result=None):
    assert prog.num_vars() == 2
    assert X.size == Y.size

    N = X.size
    values = np.vstack((X.reshape(-1), Y.reshape(-1)))
    costs = prog.GetAllCosts()

    # Vectorized multiply for the quadratic form.
    # Z = (D*np.matmul(Q,D)).sum(0).reshape(nx, ny)

    if costs:
        Z = prog.EvalBindingVectorized(costs[0], values)
        for b in costs[1:]:
            Z = Z + prog.EvalBindingVectorized(b, values)

    cv = meshcat["constraint"]
    for binding in prog.GetAllConstraints():
        c = binding.evaluator()
        var_indices = [
            int(prog.decision_variable_index()[v.get_id()])
            for v in binding.variables()
        ]
        satisfied = np.array(
            c.CheckSatisfiedVectorized(values[var_indices, :],
                                       0.001)).reshape(1, -1)
        if costs:
            Z[~satisfied] = np.nan

        # Special case linear constraints
        if False:  # isinstance(c, LinearConstraint):
            # TODO: take these as (optional) arguments to avoid computing them
            # inefficiently.
            xmin = np.min(X.reshape(-1))
            xmax = np.max(X.reshape(-1))
            ymin = np.min(Y.reshape(-1))
            ymax = np.max(Y.reshape(-1))
            A = c.A()
            lower = c.lower_bound()
            upper = c.upper_bound()
            # find line / box intersections
            # https://gist.github.com/ChickenProp/3194723
        else:
            v = cv[str(binding)]
            Zc = np.zeros(Z.shape)
            Zc[satisfied] = np.nan
            plot_surface(v,
                         X,
                         Y,
                         Zc.reshape((X.shape[1], X.shape[0])),
                         color=0x9999dd)

    if costs:
        plot_surface(meshcat["objective"],
                     X,
                     Y,
                     Z.reshape(X.shape[1], X.shape[0]),
                     wireframe=True)

    if result:
        v = meshcat["solution"]
        v.set_object(g.Sphere(0.1), g.MeshLambertMaterial(color=0x99ff99))
        x_solution = result.get_x_val()
        v.set_transform(
            tf.translation_matrix(
                [x_solution[0], x_solution[1],
                 result.get_optimal_cost()]))
