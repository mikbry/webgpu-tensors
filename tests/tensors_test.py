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

    def test_create_zeros_tensor(self):
        shape = [2, 3]
        tensor = torch.zeros(shape)
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(list(tensor.shape), shape)
        torch.testing.assert_close(tensor, torch.zeros(shape))

    def test_create_rand_tensor(self):
        shape = [2, 3]
        tensor = torch.rand(shape)
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(list(tensor.shape), shape)
        self.assertTrue((tensor >= 0).all() and (tensor < 1).all())

    def test_create_randn_tensor(self):
        shape = [1000]  # Large shape for better statistical properties
        tensor = torch.randn(shape)
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(list(tensor.shape), shape)
        self.assertAlmostEqual(tensor.mean().item(), 0, delta=0.1)
        self.assertAlmostEqual(tensor.std().item(), 1, delta=0.1)

    def test_matrix_multiplication(self):
        a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
        b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)
        result = torch.matmul(a, b)
        expected = torch.tensor([[19, 22], [43, 50]], dtype=torch.float32)
        torch.testing.assert_close(result, expected)

    def test_subtraction(self):
        a = torch.tensor([1, 2, 3], dtype=torch.float32)
        b = torch.tensor([4, 5, 6], dtype=torch.float32)
        result = torch.sub(a, b)
        expected = torch.tensor([-3, -3, -3], dtype=torch.float32)
        torch.testing.assert_close(result, expected)

    def test_power(self):
        a = torch.tensor([1, 2, 3], dtype=torch.float32)
        result = torch.pow(a, 2)
        expected = torch.tensor([1, 4, 9], dtype=torch.float32)
        torch.testing.assert_close(result, expected)

    def test_multiplication(self):
        a = torch.tensor([1, 2, 3], dtype=torch.float32)
        b = torch.tensor([4, 5, 6], dtype=torch.float32)
        result = torch.mul(a, b)
        expected = torch.tensor([4, 10, 18], dtype=torch.float32)
        torch.testing.assert_close(result, expected)

    def test_greater_than(self):
        a = torch.tensor([1, 2, 3], dtype=torch.float32)
        result = a > 2
        expected = torch.tensor([False, False, True])
        torch.testing.assert_close(result, expected)

    def test_transpose(self):
        a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
        result = torch.transpose(a, 0, 1)
        expected = torch.tensor([[1, 4], [2, 5], [3, 6]], dtype=torch.float32)
        torch.testing.assert_close(result, expected)

    def test_maximum(self):
        a = torch.tensor([-1, 0, 1], dtype=torch.float32)
        result = torch.maximum(a, torch.tensor(0))
        expected = torch.tensor([0, 0, 1], dtype=torch.float32)
        torch.testing.assert_close(result, expected)

    def test_relu_activation(self):
        input_tensor = torch.tensor([-2, -1, 0, 1, 2], dtype=torch.float32)
        result = torch.relu(input_tensor)
        expected = torch.tensor([0, 0, 0, 1, 2], dtype=torch.float32)
        torch.testing.assert_close(result, expected)

    def test_mean(self):
        a = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
        result = torch.mean(a)
        expected = torch.tensor(3, dtype=torch.float32)
        torch.testing.assert_close(result, expected)

    def test_softmax(self):
        a = torch.tensor([1, 2, 3], dtype=torch.float32)
        result = torch.softmax(a, dim=0)
        expected = torch.tensor([0.0900, 0.2447, 0.6652], dtype=torch.float32)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_sigmoid(self):
        a = torch.tensor([-2, -1, 0, 1, 2], dtype=torch.float32)
        result = torch.sigmoid(a)
        expected = torch.tensor([0.1192, 0.2689, 0.5000, 0.7311, 0.8808], dtype=torch.float32)
        torch.testing.assert_close(result, expected, atol=1e-4, rtol=1e-4)

    def test_max(self):
        a = torch.tensor([1, 5, 3, 2, 4], dtype=torch.float32)
        result = torch.max(a)
        expected = torch.tensor(5, dtype=torch.float32)
        torch.testing.assert_close(result, expected)

    def test_sum(self):
        a = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)
        result = torch.sum(a)
        expected = torch.tensor(15, dtype=torch.float32)
        torch.testing.assert_close(result, expected)

if __name__ == '__main__':
    unittest.main()
