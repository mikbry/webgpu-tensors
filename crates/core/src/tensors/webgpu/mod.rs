use std::sync::Arc;

use crate::{DType, Device, RSTensor, Shape, Size, Tensor, TensorBuffer, TensorOptions, Tensors};
use wgpu::util::DeviceExt;

pub struct WGPUTensors {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl WGPUTensors {
    
    pub fn create() -> Arc<WGPUTensors>{
        let wgpu_tensors = pollster::block_on(async { WGPUTensors::new().await });
        Arc::new(wgpu_tensors)
    }

    pub async fn new() -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
        });
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                    label: None,
                },
                None,
            )
            .await
            .unwrap();

        WGPUTensors { device, queue }
    }
}

impl Tensors for WGPUTensors {
    fn init(&mut self, _device: Option<Device>) -> Result<(), &'static str> {
        Ok(())
    }

    fn reset(&mut self) {}

    fn compute(&mut self) {}

    fn destroy(&mut self) {}

    fn empty(&self, shape: Shape, _options: Option<TensorOptions>) -> RSTensor {
        let size = Size::new(shape.clone());
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Empty Tensor"),
            size: (size.size() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        RSTensor {
            shape: size,
            dtype: DType::Float32,
            device: Device::GPU,
            readable: false,
            buffer: TensorBuffer::GPU(buffer),
        }
    }

    fn ones(&self, shape: Shape, _options: Option<TensorOptions>) -> RSTensor {
        let size = Size::new(shape.clone());
        let data = vec![1.0; size.size()];
        let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Ones Tensor"),
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });

        RSTensor {
            shape: size,
            dtype: DType::Float32,
            device: Device::GPU,
            readable: false,
            buffer: TensorBuffer::GPU(buffer),
        }
    }

    fn rand(&self, shape: Shape, _options: Option<TensorOptions>) -> RSTensor {
        let size = Size::new(shape.clone());
        let data: Vec<f32> = (0..size.size()).map(|_| rand::random::<f32>()).collect();
        let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Random Tensor"),
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });

        RSTensor {
            shape: size,
            dtype: DType::Float32,
            device: Device::GPU,
            readable: false,
            buffer: TensorBuffer::GPU(buffer),
        }
    }

    fn tensor<T: Into<RSTensor>>(&self, array: T, shape: Option<Shape>, _options: Option<TensorOptions>) -> RSTensor {
        let mut t: RSTensor = array.into();
        match shape {
            Some(shape) => t.shape.data = shape,
            None => (),
        }
        let  data = t.buffer.into_array();
        // assert_eq!(data.len(), size.size(), "Data length must match the shape size");
        let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tensor"),
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });

        RSTensor {
            shape: t.shape,
            dtype: DType::Float32,
            device: Device::GPU,
            readable: false,
            buffer: TensorBuffer::GPU(buffer),
        }
    }

    fn clone(&self, tensor: &RSTensor) -> RSTensor {
        let new_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cloned Tensor"),
            size: (tensor.numel() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        if let TensorBuffer::GPU(buffer) = &tensor.buffer {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Clone Encoder"),
            });
            encoder.copy_buffer_to_buffer(buffer, 0, &new_buffer, 0, new_buffer.size());
            self.queue.submit(std::iter::once(encoder.finish()));
        }

        RSTensor {
            shape: tensor.shape().clone(),
            dtype: tensor.dtype(),
            device: Device::GPU,
            readable: false,
            buffer: TensorBuffer::GPU(new_buffer),
        }
    }

    fn clone_tensor(&self, tensor: &RSTensor) -> RSTensor {
        self.clone(tensor)
    }

    fn matmul(&self, _a: &RSTensor, _b: &RSTensor) -> RSTensor {
        // Implement matrix multiplication for GPU tensors
        // This requires writing and executing a compute shader
        unimplemented!("GPU matrix multiplication not yet implemented")
    }

    fn maximum(&self, _tensor: &RSTensor, _value: f32) -> RSTensor {
        // Implement element-wise maximum for GPU tensors
        // This requires writing and executing a compute shader
        unimplemented!("GPU element-wise maximum not yet implemented")
    }
    
    fn randn(&self, shape: Shape, options: Option<TensorOptions>) -> RSTensor {
        todo!()
    }
    
    fn zeros(&self, shape: Shape, options: Option<TensorOptions>) -> RSTensor {
        todo!()
    }
    
    fn copy(&self, src: &RSTensor, dst: &mut RSTensor) -> Result<(), &'static str> {
        todo!()
    }
    
    fn sigmoid(&self, tensor: &RSTensor) -> RSTensor {
        todo!()
    }
    
    fn item(&self, tensor: &RSTensor) -> f32 {
        todo!()
    }
    
    fn gt(&self, tensor: &RSTensor, value: f32) -> RSTensor {
        todo!()
    }
    
    fn transpose(&self, tensor: &RSTensor) -> RSTensor {
        todo!()
    }
    
    fn mul(&self, a: &RSTensor, b: &RSTensor) -> RSTensor {
        todo!()
    }
    
    fn mul_scalar(&self, a: &RSTensor, b: f32) -> RSTensor {
        todo!()
    }
    
    fn mean(&self, tensor: &RSTensor) -> RSTensor {
        todo!()
    }
    
    fn pow(&self, tensor: &RSTensor, exponent: f32) -> RSTensor {
        todo!()
    }
    
    fn sub(&self, a: &RSTensor, b: &RSTensor) -> RSTensor {
        todo!()
    }
    
    fn relu(&self, tensor: &RSTensor) -> RSTensor {
        todo!()
    }
    
    fn max(&self, tensor: &RSTensor) -> RSTensor {
        todo!()
    }
}
