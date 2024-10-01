import unittest
import torch

class TestPyTorchTensors(unittest.TestCase):
    def test_create_empty_tensor(self):
        shape = [2, 3]
        tensor = torch.empty(shape)
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(list(tensor.shape), shape)
        self.assertEqual(tensor.dtype, torch.float32)

    def test_create_ones_tensor(self):
        shape = [2, 2]
        tensor = torch.ones(shape)
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(list(tensor.shape), shape)
        torch.testing.assert_close(tensor, torch.ones(shape))

    def test_matrix_multiplication(self):
        a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
        b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)
        result = torch.matmul(a, b)
        expected = torch.tensor([[19, 22], [43, 50]], dtype=torch.float32)
        torch.testing.assert_close(result, expected)

    def test_relu_activation(self):
        input_tensor = torch.tensor([-2, -1, 0, 1, 2], dtype=torch.float32)
        result = torch.relu(input_tensor)
        expected = torch.tensor([0, 0, 0, 1, 2], dtype=torch.float32)
        torch.testing.assert_close(result, expected)

if __name__ == '__main__':
    unittest.main()
