use crate::{Device, DType, RSTensor, Shape, Size, Tensor, TensorOptions, Tensors};
use rand::distributions::{Distribution, Standard};
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

    // Implement other methods similarly to the original RSTensors implementation
    // ...

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
}
