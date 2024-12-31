import unittest

import numpy as np
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight

tri_obs = np.array(
    [
        [
            [0.98886109, 0.81957125],
            [0.92702203, 0.71246285],
            [1.05070015, 0.71246285],
        ],
        [
            [0.28044399, 0.86069726],
            [0.21859425, 0.75357036],
            [0.34229373, 0.75357036],
        ],
        [
            [0.10322601, 0.53148317],
            [0.03083525, 0.40609871],
            [0.17561676, 0.40609871],
        ],
        [
            [0.9085955, 0.37510306],
            [0.83802404, 0.25286969],
            [0.97916697, 0.25286969],
        ],
        [
            [0.28777534, 0.21868408],
            [0.21099742, 0.08570082],
            [0.36455326, 0.08570082],
        ],
        [
            [0.01936696, 0.722583],
            [-0.01851946, 0.6569618],
            [0.05725338, 0.6569618],
        ],
        [
            [0.21162812, 0.37628579],
            [0.11572522, 0.21017709],
            [0.30753101, 0.21017709],
        ],
        [
            [0.49157316, 0.09989796],
            [0.4512723, 0.03009484],
            [0.53187401, 0.03009484],
        ],
        [
            [0.57411761, 0.21150516],
            [0.51801944, 0.11434028],
            [0.63021577, 0.11434028],
        ],
        [
            [0.58930554, 0.80235816],
            [0.5004515, 0.64845846],
            [0.67815957, 0.64845846],
        ],
        [[0.0, 0.0], [1.0, 0.0], [0.5, -0.5]],
        [[1.0, 0.0], [1.0, 1.0], [1.5, 0.5]],
        [[1.0, 1.0], [0.0, 1.0], [0.5, 1.5]],
        [[0.0, 1.0], [0.0, 0.0], [-0.5, 0.5]],
    ]
)


class TestTaskspaceIRIS(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(4)
    @timeout_decorator.timeout(5.0)
    def test_closest_point_ellipse(self):
        """Test closest point"""
        f = self.notebook_locals["ClosestPointOnObstacle"]

        test_C = np.array([[0.22810313, -0.10359835], [-0.0444466, 0.27173404]])
        test_C_inv = np.linalg.inv(test_C)
        test_d = np.array([0.36327963, 0.4762024])

        test_idxs = [7, 12, 1, 9]

        x_star_sol = np.array(
            [
                [0.49157316, 0.09989796],
                [0.09874051, 1.0],
                [0.22319751, 0.75357036],
                [0.5004515, 0.64845846],
            ]
        )
        dist_sol = np.array(
            [
                1.3984260160646136,
                1.902331715469969,
                1.0073468778009206925,
                1.24440873680021546,
            ]
        )

        tris = [val for val in tri_obs]

        x_diff = []
        dist_diff = []
        for i, idx in enumerate(test_idxs):
            o = tris[idx]
            x_star_pred, dist_pred = f(test_C, test_C_inv, test_d, o)

            x_diff.append(np.linalg.norm(x_star_sol[i] - x_star_pred))
            dist_diff.append(np.linalg.norm(dist_sol[i] - dist_pred))

        x_diff = np.asarray(x_diff)
        dist_diff = np.asarray(dist_diff)

        self.assertTrue((x_diff < 1e-3).all(), "Closest points are wrong!")
        self.assertTrue((dist_diff < 1e-2).all(), "Closest distances are wrong!")

    @weight(4)
    @timeout_decorator.timeout(5.0)
    def test_separating_hyperplane(self):
        """Test separating hyperplane"""
        f = self.notebook_locals["SeparatingHyperplanes"]

        tris = [val for val in tri_obs]
        test_C = np.array([[0.22810313, -0.10359835], [-0.0444466, 0.27173404]])
        test_d = np.array([0.36327963, 0.4762024])
        A, b, found = f(test_C, test_d, tris)

        if False:  # For debugging
            import matplotlib.pyplot as plt
            from matplotlib.patches import Polygon

            fig, ax = plt.subplots()
            draw_intersection = self.notebook_locals["draw_intersection"]
            draw_intersection(A, b, test_d, ax)
            for region in tri_obs:
                plt.plot(region[:, 0], region[:, 1], color="green", alpha=0.5)
                plt.plot(
                    region[[0, -1], 0], region[[0, -1], 1], color="green", alpha=0.5
                )
                ax.add_patch(Polygon(region, color="green", alpha=0.25))
            plt.show()

        self.assertTrue(found, "No solution found for hyperplanes!")

        self.assertEqual(A.shape, (2, 8), "We expect 8 hyperplanes")
        self.assertEqual(b.shape, (8, 1), "We expect 8 hyperplanes")

        # Check that the point is on one side of all the hyperplanes
        self.assertTrue(
            (A.T @ test_d < b.T).all(),
            "The center point, d, of the ellipse is outside the resulting polytope.",
        )

        # Check each obstacles is outside the at least one hyperplane
        epsilon = 1e-3
        for obs in tri_obs:
            all_vertices_outside = np.all(A.T @ obs.T >= b - epsilon, axis=1)
            at_least_one_separating_hyperplane = np.any(all_vertices_outside)
            self.assertTrue(
                at_least_one_separating_hyperplane,
                f"An obstacle is inside the resulting polytope.",
            )
