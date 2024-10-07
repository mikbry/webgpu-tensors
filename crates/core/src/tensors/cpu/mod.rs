use crate::{DType, Device, TensorsOperations, Shape, Size, Tensor, TensorOptions};
use rand::{distributions::Standard, prelude::Distribution, Rng};

use super::tensor::TensorBuffer;

pub struct CPUTensors;

impl CPUTensors {
    pub fn create() -> CPUTensors {
        let tensors = CPUTensors;

        tensors
    }
}

impl TensorsOperations for CPUTensors {
    fn init(&mut self, _device: Option<Device>) -> Result<(), &'static str> {
        Ok(()) // No initialization needed for RS implementation
    }

    fn reset(&mut self) {
        // TODO
    }

    fn compute(&mut self) {
        // TODO
    }

    fn destroy(&mut self) {
        // alert("Hello, wasm!");
    }

    fn empty(&self, shape: Shape, _options: Option<TensorOptions>) -> Tensor {
        let size = Size::new(shape.clone());
        Tensor {
            buffer: TensorBuffer::CPU(vec![0.0; size.size()]),
            shape: size,
            dtype: DType::Float32,
            device: Device::CPU,
            readable: true,
        }
    }

    fn ones(&self, shape: Shape, _options: Option<TensorOptions>) -> Tensor {
        let size = Size::new(shape.clone());
        Tensor {
            buffer: TensorBuffer::CPU(vec![1.0; size.size()]),
            shape: size,
            dtype: DType::Float32,
            device: Device::CPU,
            readable: true,
        }
    }

    fn zeros(&self, shape: Shape, _options: Option<TensorOptions>) -> Tensor {
        let size = Size::new(shape.clone());
        Tensor {
            buffer: TensorBuffer::CPU(vec![0.0; size.size()]),
            shape: size,
            dtype: DType::Float32,
            device: Device::CPU,
            readable: true,
        }
    }

    fn tensor<T: Into<Tensor>>(&self, n_array: T,  _options: Option<TensorOptions>) -> Tensor {
        let t = n_array.into();
        t
    }

    fn rand(&self, shape: Shape, _options: Option<TensorOptions>) -> Tensor {
        let mut rng = rand::thread_rng();
        let size = Size::new(shape.clone());
        Tensor {
            buffer: TensorBuffer::CPU((0..size.size()).map(|_| rng.gen::<f32>()).collect()),
            shape: size,
            dtype: DType::Float32,
            device: Device::CPU,
            readable: true,
        }
    }

    fn randn(&self, shape: Shape, _options: Option<TensorOptions>) -> Tensor {
        let mut rng = rand::thread_rng();
        let size = Size::new(shape.clone());
        Tensor {
            buffer: TensorBuffer::CPU(Standard.sample_iter(&mut rng).take(size.size()).collect()),
            shape: size,
            dtype: DType::Float32,
            device: Device::CPU,
            readable: true,
        }
    }

    fn matmul(&self, tensor_a: &Tensor, tensor_b: &Tensor) -> Tensor {
        // Implement matrix multiplication
        let a = tensor_a.shape();
        let b = tensor_b.shape();

        let m = a.get_dim(0).unwrap();
        let n = a.get_dim(1).unwrap_or(0);
        let p = b.get_dim(1).unwrap_or(0);

        let mut result = vec![0.0; m * p];

        // Implement the actual matrix multiplication
        let buffer_a = tensor_a.buffer_array();
        let buffer_b = tensor_b.buffer_array();
        for i in 0..m {
            for j in 0..p {
                for k in 0..n {
                    result[i * p + j] += buffer_a[i * n + k] * buffer_b[k * p + j];
                }
            }
        }

        Tensor {
            buffer: TensorBuffer::CPU(result),
            shape: Size::new(vec![m, p]),
            dtype: DType::Float32,
            device: Device::CPU,
            readable: true,
        }
    }

    fn copy(&self, src: &Tensor, dst: &mut Tensor) -> Result<(), &'static str> {
        if src.shape() == dst.shape() {
            if let TensorBuffer::CPU(src_buffer) = &src.buffer {
                dst.buffer = TensorBuffer::CPU(src_buffer.to_vec());
                return Ok(());
            }
            return Err("Source has no CPU type.");
        }
        return Err("Source and destination tensors must have the same shape.");
    }

    fn maximum(&self, tensor: &Tensor, value: f32) -> Tensor {
        let array = tensor.buffer_array();
        let data: Vec<f32> = array.iter().map(|&x| x.max(value)).collect();
        Tensor {
            buffer: TensorBuffer::CPU(data),
            shape: tensor.shape().clone(),
            dtype: tensor.dtype(),
            device: tensor.device(),
            readable: tensor.readable(),
        }
    }

    fn relu(&self, tensor: &Tensor) -> Tensor {
        self.maximum(tensor, 0.0)
    }

    fn sub(&self, a: &Tensor, b: &Tensor) -> Tensor {
        assert_eq!(
            a.shape(),
            b.shape(),
            "Tensors must have the same shape for subtraction."
        );
        let data: Vec<f32> = a
            .buffer_array()
            .iter()
            .zip(b.buffer_array().iter())
            .map(|(&x, &y)| x - y)
            .collect();
        Tensor {
            buffer: TensorBuffer::CPU(data),
            shape: a.shape().clone(),
            dtype: a.dtype(),
            device: a.device(),
            readable: a.readable(),
        }
    }

    fn pow(&self, tensor: &Tensor, exponent: f32) -> Tensor {
        let data: Vec<f32> = tensor
            .buffer_array()
            .iter()
            .map(|&x| x.powf(exponent))
            .collect();
        Tensor {
            buffer: TensorBuffer::CPU(data),
            shape: tensor.shape().clone(),
            dtype: tensor.dtype(),
            device: tensor.device(),
            readable: tensor.readable(),
        }
    }

    fn mean(&self, tensor: &Tensor) -> Tensor {
        let sum: f32 = tensor.buffer_array().iter().sum();
        let mean = sum / tensor.numel() as f32;
        Tensor {
            buffer: TensorBuffer::CPU(vec![mean]),
            shape: Size::new(vec![1]),
            dtype: tensor.dtype(),
            device: tensor.device(),
            readable: tensor.readable(),
        }
    }

    fn mul(&self, a: &Tensor, b: &Tensor) -> Tensor {
        assert_eq!(
            a.shape(),
            b.shape(),
            "Tensors must have the same shape for element-wise multiplication"
        );
        let data: Vec<f32> = a
            .buffer_array()
            .iter()
            .zip(b.buffer_array().iter())
            .map(|(&x, &y)| x * y)
            .collect();
        Tensor {
            buffer: TensorBuffer::CPU(data),
            shape: a.shape().clone(),
            dtype: a.dtype(),
            device: a.device(),
            readable: a.readable(),
        }
    }

    fn sigmoid(&self, tensor: &Tensor) -> Tensor {
        let data: Vec<f32> = tensor
            .buffer_array()
            .iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp()))
            .collect();
        Tensor {
            buffer: TensorBuffer::CPU(data),
            shape: tensor.shape().clone(),
            dtype: tensor.dtype(),
            device: tensor.device(),
            readable: tensor.readable(),
        }
    }

    fn mul_scalar(&self, a: &Tensor, b: f32) -> Tensor {
        let data: Vec<f32> = a.buffer_array().iter().map(|&x| x * b).collect();
        Tensor {
            buffer: TensorBuffer::CPU(data),
            shape: a.shape().clone(),
            dtype: a.dtype(),
            device: a.device(),
            readable: a.readable(),
        }
    }

    fn transpose(&self, tensor: &Tensor) -> Tensor {
        let shape = tensor.shape();
        assert_eq!(
            shape.length(),
            2,
            "Transpose is only implemented for 2D tensors"
        );
        let rows = shape.get_dim(0).unwrap();
        let cols = shape.get_dim(1).unwrap();
        let mut data = vec![0.0; rows * cols];
        let buffer = tensor.buffer_array();
        for i in 0..rows {
            for j in 0..cols {
                data[j * rows + i] = buffer[i * cols + j];
            }
        }
        Tensor {
            buffer: TensorBuffer::CPU(data),
            shape: Size::new(vec![cols, rows]),
            dtype: tensor.dtype(),
            device: tensor.device(),
            readable: tensor.readable(),
        }
    }

    fn gt(&self, tensor: &Tensor, value: f32) -> Tensor {
        let data: Vec<f32> = tensor
            .buffer_array()
            .iter()
            .map(|&x| if x > value { 1.0 } else { 0.0 })
            .collect();
        Tensor {
            buffer: TensorBuffer::CPU(data),
            shape: tensor.shape().clone(),
            dtype: tensor.dtype(),
            device: tensor.device(),
            readable: tensor.readable(),
        }
    }

    fn clone_tensor(&self, tensor: &Tensor) -> Tensor {
        Tensor {
            buffer: TensorBuffer::CPU(tensor.buffer_array()),
            shape: tensor.shape().clone(),
            dtype: tensor.dtype(),
            device: tensor.device(),
            readable: tensor.readable(),
        }
    }

    fn clone(&self, tensor: &Tensor) -> Tensor {
        self.clone_tensor(tensor)
    }

    fn item(&self, tensor: &Tensor) -> f32 {
        assert_eq!(tensor.numel(), 1, "Tensor must contain a single element");
        tensor.buffer_array()[0]
    }

    fn max(&self, tensor: &Tensor) -> Tensor {
        let max_value = tensor
            .buffer_array()
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        Tensor {
            buffer: TensorBuffer::CPU(vec![max_value]),
            shape: Size::new(vec![1]),
            dtype: tensor.dtype(),
            device: tensor.device(),
            readable: tensor.readable(),
        }
    }
}
