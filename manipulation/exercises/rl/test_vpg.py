import unittest

import timeout_decorator
import torch
from gradescope_utils.autograder_utils.decorators import weight
from torch.nn.functional import l1_loss


class TestVPG(unittest.TestCase):
    def __init__(self, test_name, notebook_locals):
        super().__init__(test_name)
        self.notebook_locals = notebook_locals

    @weight(2)
    @timeout_decorator.timeout(60.0)
    def test_policy_loss(self):
        """Testing the policy loss"""
        PolicyEstimator = self.notebook_locals["PolicyEstimator"]
        student_policy_loss = self.notebook_locals["util_compute_policy_loss"]
        obs = torch.ones(3, 2) * 2.0
        actions = torch.ones(3, 1) * 3.0
        returns = torch.arange(3, 6).float()
        policy_f = PolicyEstimator(num_hidden=1, hidden_dim=2, obs_dim=2, action_dim=1)
        # initialize to constant values
        for p in policy_f.parameters():
            torch.nn.init.ones_(p)
        loss_student = student_policy_loss(policy_f, obs, actions, returns)
        self.assertTrue(
            torch.abs(loss_student - 7.6758) < 0.1,
            "The policy loss computation is incorrect",
        )

    @weight(2)
    @timeout_decorator.timeout(60.0)
    def test_value_loss(self):
        """Testing the value loss"""
        ValueEstimator = self.notebook_locals["ValueEstimator"]
        value_f = ValueEstimator(num_hidden=1, hidden_dim=2, obs_dim=2, action_dim=1)
        student_value_loss = self.notebook_locals["util_compute_value_loss"]
        # initialize to constant values
        for p in value_f.parameters():
            torch.nn.init.ones_(p)

        # test tensors
        obs = torch.ones(3, 2) * 2.0
        returns = torch.arange(3, 6).float()

        loss_student = student_value_loss(value_f, obs, returns)
        self.assertTrue(
            torch.abs(loss_student - 2.0317) < 0.1,
            "The value loss computation is incorrect",
        )

    @weight(4)
    @timeout_decorator.timeout(60.0)
    def test_advantage_function(self):
        """Testing advantage function"""
        compute_advantages = self.notebook_locals["compute_advantages"]
        discount = 0.99
        gae_lambda = 1.0
        max_episode_length = 3
        baselines = torch.FloatTensor([[1, 2, 3], [2, 1, 1]])
        rewards = torch.FloatTensor([[9, 8, 5], [6, 6, 6]])
        student_sol = compute_advantages(
            discount, gae_lambda, max_episode_length, baselines, rewards
        )
        reference_sol = torch.FloatTensor(
            [[20.8205, 10.9500, 2.9900], [15.8206, 10.9400, 5.9900]]
        )

        l1_error = l1_loss(input=student_sol[:, :-1], target=reference_sol[:, :-1])
        print(l1_error)
        self.assertLess(
            l1_error,
            5.0,
            "computation of the advantage " "function is incorrect",
        )
