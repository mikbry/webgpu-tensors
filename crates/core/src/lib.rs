mod tensors;

use std::{fmt, sync::Arc};

use serde::{Deserialize, Serialize};
use tensors::{cpu::CPUTensors, webgpu::WGPUTensors};

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


pub trait Tensor {
    fn shape(&self) -> &Size;
    fn dtype(&self) -> DType;
    fn device(&self) -> Device;
    fn readable(&self) -> bool;
    fn numel(&self) -> usize;
    fn size(&self, dim: Option<usize>) -> Option<usize>;
    fn read_array(&self) -> Result<Vec<f32>, &'static str>;
    fn read_float32(&self) -> Result<Vec<f32>, &'static str>;
}

#[derive(Debug, Serialize, Deserialize)]
pub enum TensorBuffer {
    CPU(Vec<f32>),
    #[serde(skip)]
    GPU(wgpu::Buffer),
}

impl TensorBuffer {
    fn into_array(&self) -> Vec<f32> {
        match self {
            TensorBuffer::CPU(vec) => vec.to_vec(),
            TensorBuffer::GPU(_buffer) => Vec::new(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RSTensor {
    pub(crate) buffer: TensorBuffer,
    pub(crate) shape: Size,
    pub(crate) dtype: DType,
    pub(crate) device: Device,
    pub(crate) readable: bool,
    // pub(crate) buffer: Option<wgpu::Buffer>,
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

    fn read_array(&self) -> Result<Vec<f32>, &'static str> {
        if self.readable {
            if let TensorBuffer::CPU(buffer) = &self.buffer {
                return Ok(buffer.clone());
            }
        }
        Err("Tensor is not readable")
    }

    fn read_float32(&self) -> Result<Vec<f32>, &'static str> {
        if self.readable {
            if let TensorBuffer::CPU(buffer) = &self.buffer {
                return Ok(buffer.clone());
            }
        }
        Err("Tensor is not readable")
    }
}

impl From<Vec<f32>> for RSTensor {
    fn from(data: Vec<f32>) -> Self {
        let len = data.len();
        RSTensor {
            shape: Size::new(vec![len]),
            dtype: DType::Float32,
            device: Device::CPU,
            readable: true,
            buffer: TensorBuffer::CPU(data),
        }
    }
}

impl From<Vec<Vec<f32>>> for RSTensor {
    fn from(data: Vec<Vec<f32>>) -> Self {
        let shape = vec![data.len(), data[0].len()];
        let flattened: Vec<f32> = data.into_iter().flatten().collect();
        RSTensor {
            shape: Size::new(shape),
            dtype: DType::Float32,
            device: Device::CPU,
            readable: true,
            buffer: TensorBuffer::CPU(flattened),
        }
    }
}

impl From<Vec<Vec<Vec<f32>>>> for RSTensor {
    fn from(data: Vec<Vec<Vec<f32>>>) -> Self {
        let shape = vec![data.len(), data[0].len(), data[0][0].len()];
        let flattened: Vec<f32> = data.into_iter().flatten().flatten().collect();
        RSTensor {
            shape: Size::new(shape),
            dtype: DType::Float32,
            device: Device::CPU,
            readable: true,
            buffer: TensorBuffer::CPU(flattened),
        }
    }
}

impl fmt::Display for RSTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fn format_nested(data: &[f32], shape: &[usize], depth: usize) -> String {
            if shape.is_empty() {
                return format!("{}", data[0]);
            }
            let mut result = String::new();
            let dim = shape[0];
            let sub_size: usize = shape[1..].iter().product();
            result.push('[');
            for i in 0..dim {
                if i > 0 {
                    result.push_str(",");
                }
                if depth > 0 {
                    result.push_str(&"".repeat(depth));
                }
                let start = i * sub_size;
                let end = start + sub_size;
                if start < data.len() {
                    let end = std::cmp::min(end, data.len());
                    result.push_str(&format_nested(&data[start..end], &shape[1..], depth + 1));
                } else {
                    result.push_str("[...]");
                }
            }
            result.push(']');
            result
        }

        match &self.buffer {
            TensorBuffer::CPU(data) => write!(f, "{}", format_nested(data, &self.shape.data, 0)),
            TensorBuffer::GPU(_) => write!(f, "GPU Tensor: {:?}", self.shape),
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
    fn randn(&self, shape: Shape, options: Option<TensorOptions>) -> RSTensor;
    fn zeros(&self, shape: Shape, options: Option<TensorOptions>) -> RSTensor;
    fn tensor<T: Into<RSTensor>>(&self, n_array: (T, Shape), options: Option<TensorOptions>) -> RSTensor;

    fn clone(&self, tensor: &RSTensor) -> RSTensor;
    fn clone_tensor(&self, tensor: &RSTensor) -> RSTensor;
    fn copy(&self, src: &RSTensor, dst: &mut RSTensor) -> Result<(), &'static str>;

    fn sigmoid(&self, tensor: &RSTensor) -> RSTensor;
    fn item(&self, tensor: &RSTensor) -> f32;
    fn gt(&self, tensor: &RSTensor, value: f32) -> RSTensor;
    fn transpose(&self, tensor: &RSTensor) -> RSTensor;
    fn mul(&self, a: &RSTensor, b: &RSTensor) -> RSTensor;
    fn mul_scalar(&self, a: &RSTensor, b: f32) -> RSTensor;
    fn mean(&self, tensor: &RSTensor) -> RSTensor;
    fn pow(&self, tensor: &RSTensor, exponent: f32) -> RSTensor;
    fn sub(&self, a: &RSTensor, b: &RSTensor) -> RSTensor;
    fn relu(&self, tensor: &RSTensor) -> RSTensor;
    fn matmul(&self, tensor_a: &RSTensor, tensor_b: &RSTensor) -> RSTensor;
    fn maximum(&self, tensor: &RSTensor, value: f32) -> RSTensor;
    fn max(&self, tensor: &RSTensor) -> RSTensor;
}

pub enum RSTensors {
    CPU(CPUTensors),
    GPU(Arc<WGPUTensors>),
}

impl RSTensors {
    pub fn new(device: Device) -> Self {
        match device {
            Device::CPU => RSTensors::CPU(CPUTensors),
            Device::GPU => RSTensors::GPU(WGPUTensors::create()),
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

    fn zeros(&self, shape: Shape, options: Option<TensorOptions>) -> RSTensor {
        match self {
            RSTensors::CPU(cpu) => cpu.zeros(shape, options),
            RSTensors::GPU(gpu) => gpu.zeros(shape, options),
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

    fn tensor<T: Into<RSTensor>>(&self, n_array: (T, Shape), options: Option<TensorOptions>) -> RSTensor {
        match self {
            RSTensors::CPU(cpu) => cpu.tensor(n_array, options),
            RSTensors::GPU(gpu) => gpu.tensor(n_array, options),
        }
    }

    fn clone(&self, tensor: &RSTensor) -> RSTensor {
        match self {
            RSTensors::CPU(cpu) => cpu.clone(tensor),
            RSTensors::GPU(gpu) => gpu.clone_tensor(tensor),
        }
    }

    fn clone_tensor(&self, tensor: &RSTensor) -> RSTensor {
        self.clone(tensor)
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

    fn randn(&self, shape: Shape, options: Option<TensorOptions>) -> RSTensor {
        match self {
            RSTensors::CPU(cpu) => cpu.randn(shape, options),
            RSTensors::GPU(gpu) => gpu.randn(shape, options),
        }
    }

    fn copy(&self, src: &RSTensor, dst: &mut RSTensor) -> Result<(), &'static str> {
        match self {
            RSTensors::CPU(cpu) => cpu.copy(src, dst),
            RSTensors::GPU(gpu) => gpu.copy(src, dst),
        }
    }

    fn sigmoid(&self, tensor: &RSTensor) -> RSTensor {
        match self {
            RSTensors::CPU(cpu) => cpu.sigmoid(tensor),
            RSTensors::GPU(gpu) => gpu.sigmoid(tensor),
        }
    }

    fn item(&self, tensor: &RSTensor) -> f32 {
        match self {
            RSTensors::CPU(cpu) => cpu.item(tensor),
            RSTensors::GPU(gpu) => gpu.item(tensor),
        }
    }

    fn gt(&self, tensor: &RSTensor, value: f32) -> RSTensor {
        match self {
            RSTensors::CPU(cpu) => cpu.gt(tensor, value),
            RSTensors::GPU(gpu) => gpu.gt(tensor, value),
        }
    }

    fn transpose(&self, tensor: &RSTensor) -> RSTensor {
        match self {
            RSTensors::CPU(cpu) => cpu.transpose(tensor),
            RSTensors::GPU(gpu) => gpu.transpose(tensor),
        }
    }

    fn mul(&self, a: &RSTensor, b: &RSTensor) -> RSTensor {
        match self {
            RSTensors::CPU(cpu) => cpu.mul(a, b),
            RSTensors::GPU(gpu) => gpu.mul(a, b),
        }
    }

    fn mul_scalar(&self, a: &RSTensor, b: f32) -> RSTensor {
        match self {
            RSTensors::CPU(cpu) => cpu.mul_scalar(a, b),
            RSTensors::GPU(gpu) => gpu.mul_scalar(a, b),
        }
    }

    fn mean(&self, tensor: &RSTensor) -> RSTensor {
        match self {
            RSTensors::CPU(cpu) => cpu.mean(tensor),
            RSTensors::GPU(gpu) => gpu.mean(tensor),
        }
    }

    fn pow(&self, tensor: &RSTensor, exponent: f32) -> RSTensor {
        match self {
            RSTensors::CPU(cpu) => cpu.pow(tensor, exponent),
            RSTensors::GPU(gpu) => gpu.pow(tensor, exponent),
        }
    }

    fn sub(&self, a: &RSTensor, b: &RSTensor) -> RSTensor {
        match self {
            RSTensors::CPU(cpu) => cpu.sub(a, b),
            RSTensors::GPU(gpu) => gpu.sub(a, b),
        }
    }

    fn relu(&self, tensor: &RSTensor) -> RSTensor {
        match self {
            RSTensors::CPU(cpu) => cpu.relu(tensor),
            RSTensors::GPU(gpu) => gpu.relu(tensor),
        }
    }

    fn max(&self, tensor: &RSTensor) -> RSTensor {
        match self {
            RSTensors::CPU(cpu) => cpu.max(tensor),
            RSTensors::GPU(gpu) => gpu.max(tensor),
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
