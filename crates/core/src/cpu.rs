use crate::{Device, DType, RSTensor, Shape, Size, Tensor, TensorOptions, Tensors};
use rand::Rng;

pub struct CPUTensors;

impl Tensors for CPUTensors {
    fn init(&mut self, _device: Option<Device>) -> Result<(), &'static str> {
        Ok(())
    }

    fn reset(&mut self) {}

    fn compute(&mut self) {}

    fn destroy(&mut self) {}

    fn empty(&self, shape: Shape, _options: Option<TensorOptions>) -> RSTensor {
        let size = Size::new(shape.clone());
        RSTensor {
            data: vec![0.0; size.size()],
            shape: size,
            dtype: DType::Float32,
            device: Device::CPU,
            readable: true,
            buffer: None,
        }
    }

    fn ones(&self, shape: Shape, _options: Option<TensorOptions>) -> RSTensor {
        let size = Size::new(shape.clone());
        RSTensor {
            data: vec![1.0; size.size()],
            shape: size,
            dtype: DType::Float32,
            device: Device::CPU,
            readable: true,
            buffer: None,
        }
    }

    fn rand(&self, shape: Shape, _options: Option<TensorOptions>) -> RSTensor {
        let mut rng = rand::thread_rng();
        let size = Size::new(shape.clone());
        RSTensor {
            data: (0..size.size()).map(|_| rng.gen::<f32>()).collect(),
            shape: size,
            dtype: DType::Float32,
            device: Device::CPU,
            readable: true,
            buffer: None,
        }
    }

    fn tensor(&self, data: Vec<f32>, shape: Shape, _options: Option<TensorOptions>) -> RSTensor {
        let size = Size::new(shape.clone());
        assert_eq!(data.len(), size.size(), "Data length must match the shape size");
        RSTensor {
            data,
            shape: size,
            dtype: DType::Float32,
            device: Device::CPU,
            readable: true,
            buffer: None,
        }
    }

    fn clone(&self, tensor: &RSTensor) -> RSTensor {
        RSTensor {
            data: tensor.data.clone(),
            shape: tensor.shape().clone(),
            dtype: tensor.dtype(),
            device: tensor.device(),
            readable: tensor.readable(),
            buffer: None,
        }
    }

    fn matmul(&self, a: &RSTensor, b: &RSTensor) -> RSTensor {
        // Implement matrix multiplication for CPU tensors
        // This is a simplified implementation and may not be optimal for large matrices
        assert_eq!(a.shape().data[1], b.shape().data[0], "Incompatible matrix dimensions");
        let m = a.shape().data[0];
        let n = b.shape().data[1];
        let k = a.shape().data[1];

        let mut result = vec![0.0; m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += a.data[i * k + l] * b.data[l * n + j];
                }
                result[i * n + j] = sum;
            }
        }

        RSTensor {
            data: result,
            shape: Size::new(vec![m, n]),
            dtype: DType::Float32,
            device: Device::CPU,
            readable: true,
            buffer: None,
        }
    }

    fn maximum(&self, tensor: &RSTensor, value: f32) -> RSTensor {
        RSTensor {
            data: tensor.data.iter().map(|&x| x.max(value)).collect(),
            shape: tensor.shape().clone(),
            dtype: tensor.dtype(),
            device: tensor.device(),
            readable: true,
            buffer: None,
        }
    }
}
