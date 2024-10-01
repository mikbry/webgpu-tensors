import unittest
import numpy as np
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from webgpu_tensors import WebGPUTensors, Tensor, DType, Device

class TestWebGPUTensors(unittest.TestCase):
    def setUp(self):
        self.tensors = WebGPUTensors()
        self.tensors.init()

    def test_create_empty_tensor(self):
        shape = [2, 3]
        tensor = self.tensors.empty(shape)
        self.assertIsInstance(tensor, Tensor)
        self.assertEqual(tensor.shape.data, shape)
        self.assertEqual(tensor.dtype, DType.Float32)

    def test_create_ones_tensor(self):
        shape = [2, 2]
        tensor = self.tensors.ones(shape)
        self.assertIsInstance(tensor, Tensor)
        self.assertEqual(tensor.shape.data, shape)
        data = tensor.read_float32()
        np.testing.assert_array_equal(data, np.ones(shape))

    def test_matrix_multiplication(self):
        a = self.tensors.tensor([[1, 2], [3, 4]])
        b = self.tensors.tensor([[5, 6], [7, 8]])
        result = self.tensors.matmul(a, b)
        data = result.read_float32()
        np.testing.assert_array_equal(data, np.array([[19, 22], [43, 50]]))

    def test_relu_activation(self):
        input_tensor = self.tensors.tensor([-2, -1, 0, 1, 2])
        result = self.tensors.relu(input_tensor)
        data = result.read_float32()
        np.testing.assert_array_equal(data, np.array([0, 0, 0, 1, 2]))

if __name__ == '__main__':
    unittest.main()
