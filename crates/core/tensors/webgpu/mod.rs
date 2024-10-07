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

    // Implement other methods similarly, using WGPU for computations
    // ...

    fn clone(&self, tensor: &RSTensor) -> RSTensor {
        let new_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cloned Tensor"),
            size: (tensor.numel() * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        self.queue.write_buffer(&new_buffer, 0, bytemuck::cast_slice(&tensor.data));

        RSTensor {
            data: vec![],
            shape: tensor.shape().clone(),
            dtype: tensor.dtype(),
            device: Device::GPU,
            readable: false,
            buffer: Some(new_buffer),
        }
    }
}
