import unittest
import torch
import torch.nn as nn


from algorithms.neural_net import neural_net


class test_lrp_linear(unittest.TestCase):
    def test_relevance(self):
        activation = torch.tensor([1, 2, 3])
        layer = nn.Linear(3, 2)
        layer.weight.data.copy_(torch.tensor([[4, 5, 6], [7, 8, 9]]))
        layer.bias.data.copy_(torch.tensor([12, 13]))
        relevance = torch.tensor([10, 11])
        # bias =
        result_target = torch.tensor(
            [
                10 * (4 / 44) + 11 * (7 / 63),
                10 * (10 / 44) + 11 * (16 / 63),
                10 * (18 / 44) + 11 * (27 / 63),
                10 * (12 / 44) + 11 * (13 / 63),
            ]
        )
        result_function = neural_net.lrp_linear(activation, layer, relevance)
        almost_equal = (result_function - result_target).sum() < 0.001
        self.assertTrue(almost_equal)


if __name__ == "__main__":
    unittest.main()
