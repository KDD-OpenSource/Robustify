import unittest
import torch


# from algorithms.neural_net import interpolate
from algorithms.neural_net import neural_net


class testInterpolate(unittest.TestCase):
    def test_points_steps(self):
        point_from = torch.tensor([0, 0])
        point_to = torch.tensor([1, 1])
        steps = 3
        result_target = torch.tensor([[0, 0], [0.5, 0.5], [1, 1]])
        result_function = neural_net.interpolate(point_from, point_to, steps)
        equal = torch.equal(result_target, result_function)
        self.assertTrue(equal)


if __name__ == "__main__":
    unittest.main()
