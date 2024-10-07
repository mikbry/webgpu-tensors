mod cpu;
mod webgpu;

use std::sync::Arc;

pub use crate::cpu::CPUTensors;
pub use crate::webgpu::WGPUTensors;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Device {
    CPU,
    GPU,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DType {
    Float32,
}

pub type Shape = Vec<usize>;

#[derive(Debug, Clone)]
pub struct Size {
    data: Vec<usize>,
}

impl Size {
    pub fn new(shape: Shape) -> Self {
        Size { data: shape }
    }

    pub fn size(&self) -> usize {
        self.data.iter().product()
    }
}

pub struct TensorOptions {
    dtype: DType,
    device: Device,
}

pub trait Tensor {
    fn shape(&self) -> &Size;
    fn dtype(&self) -> DType;
    fn device(&self) -> Device;
    fn readable(&self) -> bool;
    fn numel(&self) -> usize;
    fn size(&self, dim: Option<usize>) -> Option<usize>;
}

pub struct RSTensor {
    pub(crate) data: Vec<f32>,
    pub(crate) shape: Size,
    pub(crate) dtype: DType,
    pub(crate) device: Device,
    pub(crate) readable: bool,
    pub(crate) buffer: Option<wgpu::Buffer>,
}

impl Tensor for RSTensor {
    fn shape(&self) -> &Size {
        &self.shape
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> Device {
        self.device
    }

    fn readable(&self) -> bool {
        self.readable
    }

    fn numel(&self) -> usize {
        self.shape.size()
    }

    fn size(&self, dim: Option<usize>) -> Option<usize> {
        match dim {
            Some(d) if d < self.shape.data.len() => Some(self.shape.data[d]),
            None => Some(self.shape.size()),
            _ => None,
        }
    }
}

pub trait Tensors {
    fn init(&mut self, device: Option<Device>) -> Result<(), &'static str>;
    fn reset(&mut self);
    fn compute(&mut self);
    fn destroy(&mut self);

    fn empty(&self, shape: Shape, options: Option<TensorOptions>) -> RSTensor;
    fn ones(&self, shape: Shape, options: Option<TensorOptions>) -> RSTensor;
    fn rand(&self, shape: Shape, options: Option<TensorOptions>) -> RSTensor;
    fn tensor(&self, data: Vec<f32>, shape: Shape, options: Option<TensorOptions>) -> RSTensor;
    fn clone(&self, tensor: &RSTensor) -> RSTensor;
    fn matmul(&self, a: &RSTensor, b: &RSTensor) -> RSTensor;
    fn maximum(&self, tensor: &RSTensor, value: f32) -> RSTensor;
}

pub enum RSTensors {
    CPU(CPUTensors),
    GPU(Arc<WGPUTensors>),
}

impl RSTensors {
    pub fn new(device: Device) -> Self {
        match device {
            Device::CPU => RSTensors::CPU(CPUTensors),
            Device::GPU => {
                let runtime = tokio::runtime::Runtime::new().unwrap();
                let wgpu_tensors = runtime.block_on(async { WGPUTensors::new().await });
                RSTensors::GPU(Arc::new(wgpu_tensors))
            }
        }
    }
}

impl Tensors for RSTensors {
    fn init(&mut self, device: Option<Device>) -> Result<(), &'static str> {
        match self {
            RSTensors::CPU(cpu) => cpu.init(device),
            RSTensors::GPU(gpu) => Arc::get_mut(gpu).unwrap().init(device),
        }
    }

    fn reset(&mut self) {
        match self {
            RSTensors::CPU(cpu) => cpu.reset(),
            RSTensors::GPU(gpu) => Arc::get_mut(gpu).unwrap().reset(),
        }
    }

    fn compute(&mut self) {
        match self {
            RSTensors::CPU(cpu) => cpu.compute(),
            RSTensors::GPU(gpu) => Arc::get_mut(gpu).unwrap().compute(),
        }
    }

    fn destroy(&mut self) {
        match self {
            RSTensors::CPU(cpu) => cpu.destroy(),
            RSTensors::GPU(gpu) => Arc::get_mut(gpu).unwrap().destroy(),
        }
    }

    fn empty(&self, shape: Shape, options: Option<TensorOptions>) -> RSTensor {
        match self {
            RSTensors::CPU(cpu) => cpu.empty(shape, options),
            RSTensors::GPU(gpu) => gpu.empty(shape, options),
        }
    }

    fn ones(&self, shape: Shape, options: Option<TensorOptions>) -> RSTensor {
        match self {
            RSTensors::CPU(cpu) => cpu.ones(shape, options),
            RSTensors::GPU(gpu) => gpu.ones(shape, options),
        }
    }

    fn rand(&self, shape: Shape, options: Option<TensorOptions>) -> RSTensor {
        match self {
            RSTensors::CPU(cpu) => cpu.rand(shape, options),
            RSTensors::GPU(gpu) => gpu.rand(shape, options),
        }
    }

    fn tensor(&self, data: Vec<f32>, shape: Shape, options: Option<TensorOptions>) -> RSTensor {
        match self {
            RSTensors::CPU(cpu) => cpu.tensor(data, shape, options),
            RSTensors::GPU(gpu) => gpu.tensor(data, shape, options),
        }
    }

    fn clone(&self, tensor: &RSTensor) -> RSTensor {
        match self {
            RSTensors::CPU(cpu) => cpu.clone(tensor),
            RSTensors::GPU(gpu) => gpu.clone(tensor),
        }
    }

    fn matmul(&self, a: &RSTensor, b: &RSTensor) -> RSTensor {
        match self {
            RSTensors::CPU(cpu) => cpu.matmul(a, b),
            RSTensors::GPU(gpu) => gpu.matmul(a, b),
        }
    }

    fn maximum(&self, tensor: &RSTensor, value: f32) -> RSTensor {
        match self {
            RSTensors::CPU(cpu) => cpu.maximum(tensor, value),
            RSTensors::GPU(gpu) => gpu.maximum(tensor, value),
        }
    }
}

#[macro_export]
macro_rules! tensors_println {
    ($($arg:tt)*) => {{
        use std::io::Write;
        let mut lock = std::io::stdout().lock();
        write!(lock, $($arg)*).unwrap();
        writeln!(lock).unwrap();
    }};
}
