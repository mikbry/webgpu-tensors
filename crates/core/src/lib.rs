mod tensors;

use std::{fmt, sync::Arc};

use serde::{Deserialize, Serialize};
pub use tensors::{cpu::CPUTensors, webgpu::WGPUTensors};

pub use tensors::tensor::Tensor;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Device {
    CPU,
    GPU,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DType {
    Float32,
}

pub type Shape = Vec<usize>;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Size {
    data: Vec<usize>,
}

impl Size {
    pub fn new(data: Vec<usize>) -> Self {
        Size { data }
    }

    pub fn length(&self) -> usize {
        self.data.len()
    }

    pub fn size(&self) -> usize {
        self.data.iter().product()
    }

    pub fn get_dim(&self, dim: usize) -> Option<usize> {
        self.data.get(dim).cloned()
    }
}

impl fmt::Display for Size {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.data)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TensorOptions {
    pub usage: usize,
    pub mapped_at_creation: Option<bool>,
    pub readable: bool,
    pub dtype: DType,
    pub device: Device,
    pub shape: Option<Shape>,
}

impl TensorOptions {
    pub fn default() -> Self {
        Self {
            usage: 0,
            mapped_at_creation: None,
            readable: true,
            dtype: DType::Float32,
            device: Device::CPU,
            shape: None,
        }
    }
}

pub trait TensorsOperations {
    fn init(&mut self, device: Option<Device>) -> Result<(), &'static str>;
    fn reset(&mut self);
    fn compute(&mut self);
    fn destroy(&mut self);

    fn empty(&self, shape: Shape, options: Option<TensorOptions>) -> Tensor;
    fn ones(&self, shape: Shape, options: Option<TensorOptions>) -> Tensor;
    fn rand(&self, shape: Shape, options: Option<TensorOptions>) -> Tensor;
    fn randn(&self, shape: Shape, options: Option<TensorOptions>) -> Tensor;
    fn zeros(&self, shape: Shape, options: Option<TensorOptions>) -> Tensor;
    fn tensor<T: Into<Tensor>>(&self, n_array: T, options: Option<TensorOptions>) -> Tensor;

    fn clone(&self, tensor: &Tensor) -> Tensor;
    fn clone_tensor(&self, tensor: &Tensor) -> Tensor;
    fn copy(&self, src: &Tensor, dst: &mut Tensor) -> Result<(), &'static str>;

    fn sigmoid(&self, tensor: &Tensor) -> Tensor;
    fn item(&self, tensor: &Tensor) -> f32;
    fn gt(&self, tensor: &Tensor, value: f32) -> Tensor;
    fn transpose(&self, tensor: &Tensor) -> Tensor;
    fn mul(&self, a: &Tensor, b: &Tensor) -> Tensor;
    fn mul_scalar(&self, a: &Tensor, b: f32) -> Tensor;
    fn mean(&self, tensor: &Tensor) -> Tensor;
    fn pow(&self, tensor: &Tensor, exponent: f32) -> Tensor;
    fn sub(&self, a: &Tensor, b: &Tensor) -> Tensor;
    fn relu(&self, tensor: &Tensor) -> Tensor;
    fn matmul(&self, tensor_a: &Tensor, tensor_b: &Tensor) -> Tensor;
    fn maximum(&self, tensor: &Tensor, value: f32) -> Tensor;
    fn max(&self, tensor: &Tensor) -> Tensor;
}

pub enum Tensors {
    CPU(CPUTensors),
    GPU(Arc<WGPUTensors>),
}

impl Tensors {
    pub fn new(device: Device) -> Tensors {
        if device == Device::CPU {
            return Tensors::CPU(CPUTensors::create());

        }
        return Tensors::GPU(WGPUTensors::create());
    }
    pub fn default() -> Tensors {
        return Tensors::CPU(CPUTensors::create());
    }
}

impl TensorsOperations for Tensors {
    fn init(&mut self, device: Option<Device>) -> Result<(), &'static str> {
        match self {
            Tensors::CPU(cpu) => cpu.init(device),
            Tensors::GPU(gpu) => Arc::get_mut(gpu).unwrap().init(device),
        }
    }

    fn reset(&mut self) {
        match self {
            Tensors::CPU(cpu) => cpu.reset(),
            Tensors::GPU(gpu) => Arc::get_mut(gpu).unwrap().reset(),
        }
    }

    fn compute(&mut self) {
        match self {
            Tensors::CPU(cpu) => cpu.compute(),
            Tensors::GPU(gpu) => Arc::get_mut(gpu).unwrap().compute(),
        }
    }

    fn destroy(&mut self) {
        match self {
            Tensors::CPU(cpu) => cpu.destroy(),
            Tensors::GPU(gpu) => Arc::get_mut(gpu).unwrap().destroy(),
        }
    }

    fn empty(&self, shape: Shape, options: Option<TensorOptions>) -> Tensor {
        match self {
            Tensors::CPU(cpu) => cpu.empty(shape, options),
            Tensors::GPU(gpu) => gpu.empty(shape, options),
        }
    }

    fn zeros(&self, shape: Shape, options: Option<TensorOptions>) -> Tensor {
        match self {
            Tensors::CPU(cpu) => cpu.zeros(shape, options),
            Tensors::GPU(gpu) => gpu.zeros(shape, options),
        }
    }

    fn ones(&self, shape: Shape, options: Option<TensorOptions>) -> Tensor {
        match self {
            Tensors::CPU(cpu) => cpu.ones(shape, options),
            Tensors::GPU(gpu) => gpu.ones(shape, options),
        }
    }

    fn rand(&self, shape: Shape, options: Option<TensorOptions>) -> Tensor {
        match self {
            Tensors::CPU(cpu) => cpu.rand(shape, options),
            Tensors::GPU(gpu) => gpu.rand(shape, options),
        }
    }

    fn tensor<T: Into<Tensor>>(&self, n_array: T, options: Option<TensorOptions>) -> Tensor {
        match self {
            Tensors::CPU(cpu) => cpu.tensor(n_array, options),
            Tensors::GPU(gpu) => gpu.tensor(n_array, options),
        }
    }

    fn clone_tensor(&self, tensor: &Tensor) -> Tensor {
        match self {
            Tensors::CPU(cpu) => cpu.clone_tensor(tensor),
            Tensors::GPU(gpu) => gpu.clone_tensor(tensor),
        }
    }

    fn clone(&self, tensor: &Tensor) -> Tensor {
        self.clone_tensor(tensor)
    }

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Tensor {
        match self {
            Tensors::CPU(cpu) => cpu.matmul(a, b),
            Tensors::GPU(gpu) => gpu.matmul(a, b),
        }
    }

    fn maximum(&self, tensor: &Tensor, value: f32) -> Tensor {
        match self {
            Tensors::CPU(cpu) => cpu.maximum(tensor, value),
            Tensors::GPU(gpu) => gpu.maximum(tensor, value),
        }
    }

    fn randn(&self, shape: Shape, options: Option<TensorOptions>) -> Tensor {
        match self {
            Tensors::CPU(cpu) => cpu.randn(shape, options),
            Tensors::GPU(gpu) => gpu.randn(shape, options),
        }
    }

    fn copy(&self, src: &Tensor, dst: &mut Tensor) -> Result<(), &'static str> {
        match self {
            Tensors::CPU(cpu) => cpu.copy(src, dst),
            Tensors::GPU(gpu) => gpu.copy(src, dst),
        }
    }

    fn sigmoid(&self, tensor: &Tensor) -> Tensor {
        match self {
            Tensors::CPU(cpu) => cpu.sigmoid(tensor),
            Tensors::GPU(gpu) => gpu.sigmoid(tensor),
        }
    }

    fn item(&self, tensor: &Tensor) -> f32 {
        match self {
            Tensors::CPU(cpu) => cpu.item(tensor),
            Tensors::GPU(gpu) => gpu.item(tensor),
        }
    }

    fn gt(&self, tensor: &Tensor, value: f32) -> Tensor {
        match self {
            Tensors::CPU(cpu) => cpu.gt(tensor, value),
            Tensors::GPU(gpu) => gpu.gt(tensor, value),
        }
    }

    fn transpose(&self, tensor: &Tensor) -> Tensor {
        match self {
            Tensors::CPU(cpu) => cpu.transpose(tensor),
            Tensors::GPU(gpu) => gpu.transpose(tensor),
        }
    }

    fn mul(&self, a: &Tensor, b: &Tensor) -> Tensor {
        match self {
            Tensors::CPU(cpu) => cpu.mul(a, b),
            Tensors::GPU(gpu) => gpu.mul(a, b),
        }
    }

    fn mul_scalar(&self, a: &Tensor, b: f32) -> Tensor {
        match self {
            Tensors::CPU(cpu) => cpu.mul_scalar(a, b),
            Tensors::GPU(gpu) => gpu.mul_scalar(a, b),
        }
    }

    fn mean(&self, tensor: &Tensor) -> Tensor {
        match self {
            Tensors::CPU(cpu) => cpu.mean(tensor),
            Tensors::GPU(gpu) => gpu.mean(tensor),
        }
    }

    fn pow(&self, tensor: &Tensor, exponent: f32) -> Tensor {
        match self {
            Tensors::CPU(cpu) => cpu.pow(tensor, exponent),
            Tensors::GPU(gpu) => gpu.pow(tensor, exponent),
        }
    }

    fn sub(&self, a: &Tensor, b: &Tensor) -> Tensor {
        match self {
            Tensors::CPU(cpu) => cpu.sub(a, b),
            Tensors::GPU(gpu) => gpu.sub(a, b),
        }
    }

    fn relu(&self, tensor: &Tensor) -> Tensor {
        match self {
            Tensors::CPU(cpu) => cpu.relu(tensor),
            Tensors::GPU(gpu) => gpu.relu(tensor),
        }
    }

    fn max(&self, tensor: &Tensor) -> Tensor {
        match self {
            Tensors::CPU(cpu) => cpu.max(tensor),
            Tensors::GPU(gpu) => gpu.max(tensor),
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
