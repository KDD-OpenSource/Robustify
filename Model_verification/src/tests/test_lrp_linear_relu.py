import unittest
import torch
import torch.nn as nn


from algorithms.neural_net import neural_net


class test_lrp_linear_relu(unittest.TestCase):
    def test_relevance(self):
        activation = torch.tensor([1, 2, 3])
        layer = nn.Linear(3, 2)
        layer.weight.data.copy_(torch.tensor([[4, -5, 6], [7, -8, -9]]))
        layer.bias.data.copy_(torch.tensor([12, 13]))
        relevance = torch.tensor([10, 11])
        gamma = 2
        # bias =
        result_target = torch.tensor(
            [
                10 * (12 / 92) + 11 * (21 / 17),
                10 * (-10 / 92) + 11 * (-16 / 17),
                10 * (54 / 92) + 11 * (-27 / 17),
                10 * (36 / 92) + 11 * (39 / 17),
            ]
        )
        result_function = neural_net.lrp_linear_relu(
            activation, layer, relevance, gamma
        )
        import pdb

        pdb.set_trace()
        almost_equal = (result_function - result_target).sum() < 0.001
        self.assertTrue(almost_equal)


if __name__ == "__main__":
    unittest.main()
