import unittest
import torch

from algorithms.autoencoder import autoencoder
from algorithms.neural_net import neural_net


class testCheck_interp_signatures(unittest.TestCase):
    def test_true_signatures(self):
        point_from = torch.tensor([0, 0, 0, 0])
        point_to = torch.tensor([0, 0, 0, 0])
        num_steps = 100
        algorithm = autoencoder(
            topology=[4, 3, 4],
            num_epochs=1,
        )
        neural_net_mod = algorithm.aeModule
        result = algorithm.check_interp_signatures(
            point_from, point_to, num_steps, neural_net_mod
        )
        self.assertEqual(result, True)

    def test_false_signatures(self):
        point_from = torch.tensor([0, 0, 0, 0])
        point_to = torch.tensor([10, 10, 10, 10])
        num_steps = 100
        algorithm = autoencoder(
            topology=[4, 5, 3, 5, 4],
            num_epochs=1,
        )
        neural_net_mod = algorithm.aeModule
        result = algorithm.check_interp_signatures(
            point_from, point_to, num_steps, neural_net_mod
        )
        self.assertEqual(result, False)


if __name__ == "__main__":
    unittest.main()
