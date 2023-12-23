import unittest

import numpy as np
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import weight


def gt_don_loss(f, img_a, img_b, u_a, u_b, match, margin=2.0):
    # in SWE it's not great to copy answer code here but we are just grading
    phi_a, phi_b = f(img_a[None])[0], f(img_b[None])[0]
    match = match[:, 0]
    h, w, d = phi_a.shape
    phi_a = phi_a.reshape(-1, d)[u_a[:, 0] * w + u_a[:, 1]]
    phi_b = phi_b.reshape(-1, d)[u_b[:, 0] * w + u_b[:, 1]]
    norm = np.linalg.norm(phi_a - phi_b, axis=1)
    loss_matches = np.mean(norm[match] ** 2)
    loss_nonmatches = np.mean(np.clip(margin - norm[~match], a_min=0, a_max=None) ** 2)
    return loss_matches, loss_nonmatches


def gt_don_predict(f, img_a, img_b, u_a):
    phi_a, phi_b = f(img_a[None])[0], f(img_b[None])[0]
    h, w, d = phi_a.shape
    phi_query = phi_a[u_a[0], u_a[1]]
    phi_keys = phi_b.reshape(-1, d)
    norm = np.linalg.norm(phi_keys - phi_query[None], axis=1)
    idx = np.argmin(norm)
    u_b = np.array([idx // w, idx % w])
    return u_b


def f_factory(rand_array):
    def dummy_f(x: np.ndarray):
        if x.ndim == 3:
            print(
                "f takes in image of shape (N, H, W, 3), "
                "did you forget to add batch dimension?"
            )
        phi = rand_array.copy()[None]
        phi = np.tile(phi, (len(x), 1, 1, 1))
        phi[..., 1:4] += x
        return phi

    return dummy_f


def get_random_indices(batch_size, h, w):
    return np.stack(
        [
            np.random.randint(h, size=(batch_size,)),
            np.random.randint(w, size=(batch_size,)),
        ],
        -1,
    )


class TestContrastive(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals
        np.random.seed(0)
        self._f = f_factory(np.random.rand(480, 640, 128))
        self._img_a = np.random.rand(480, 640, 3)
        self._img_b = np.random.rand(480, 640, 3)
        self._u_a = get_random_indices(4096, 480, 640)
        self._u_b = get_random_indices(4096, 480, 640)
        self._match = np.random.rand(4096, 1) > 0.4

    @weight(4)
    @timeout_decorator.timeout(10.0)
    def test_don_loss(self):
        """Testing don_loss"""
        expected_sol = gt_don_loss(
            self._f,
            self._img_a,
            self._img_b,
            self._u_a,
            self._u_b,
            self._match,
            margin=6.0,
        )
        sol = self.notebook_locals["don_loss"](
            self._f,
            self._img_a,
            self._img_b,
            self._u_a,
            self._u_b,
            self._match,
            margin=6.0,
        )

        self.assertTrue(
            np.abs(expected_sol[0] - sol[0]) < 1e-8,
            "Computed loss_matches is incorrect",
        )

        self.assertTrue(
            np.abs(expected_sol[1] - sol[1]) < 1e-8,
            "Computed loss_nonmatches is incorrect",
        )

    @weight(4)
    @timeout_decorator.timeout(10.0)
    def test_don_predict(self):
        """Testing don_predict"""
        expected_sol = gt_don_predict(self._f, self._img_a, self._img_b, self._u_a[0])
        sol = self.notebook_locals["don_predict"](
            self._f, self._img_a, self._img_b, self._u_a[0]
        )
        self.assertTrue(
            type(sol) == np.ndarray,
            "don_predict must return a np.ndarray as stated in docstring",
        )
        self.assertTrue(
            (expected_sol == sol).all(), "Predicted pixel position incorrect"
        )
