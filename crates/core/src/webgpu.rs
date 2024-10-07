use crate::{Device, DType, RSTensor, Shape, Size, Tensor, TensorOptions, Tensors};
use wgpu::util::DeviceExt;

pub struct WGPUTensors {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl WGPUTensors {
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
            data: vec![],
            shape: size,
            dtype: DType::Float32,
            device: Device::GPU,
            readable: false,
            buffer: Some(buffer),
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
            data: vec![],
            shape: size,
            dtype: DType::Float32,
            device: Device::GPU,
            readable: false,
            buffer: Some(buffer),
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
            data: vec![],
            shape: size,
            dtype: DType::Float32,
            device: Device::GPU,
            readable: false,
            buffer: Some(buffer),
        }
    }

    fn tensor(&self, data: Vec<f32>, shape: Shape, _options: Option<TensorOptions>) -> RSTensor {
        let size = Size::new(shape.clone());
        assert_eq!(data.len(), size.size(), "Data length must match the shape size");
        let buffer = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Tensor"),
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });

        RSTensor {
            data: vec![],
            shape: size,
            dtype: DType::Float32,
            device: Device::GPU,
            readable: false,
            buffer: Some(buffer),
        }
    }

    fn clone(&self, tensor: &RSTensor) -> RSTensor {
        let new_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cloned Tensor"),
            size: (tensor.numel() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        if let Some(buffer) = &tensor.buffer {
            let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Clone Encoder"),
            });
            encoder.copy_buffer_to_buffer(buffer, 0, &new_buffer, 0, new_buffer.size());
            self.queue.submit(std::iter::once(encoder.finish()));
        }

        RSTensor {
            data: vec![],
            shape: tensor.shape().clone(),
            dtype: tensor.dtype(),
            device: Device::GPU,
            readable: false,
            buffer: Some(new_buffer),
        }
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
}
