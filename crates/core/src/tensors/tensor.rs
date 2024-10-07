use serde::{Deserialize, Serialize};
use std::fmt;

use crate::{DType, Device, Size, TensorOptions};

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
pub struct Tensor {
    pub(crate) buffer: TensorBuffer,
    pub(crate) shape: Size,
    pub(crate) dtype: DType,
    pub(crate) device: Device,
    pub(crate) readable: bool,
}

impl Tensor {
    pub fn shape(&self) -> &Size {
        &self.shape
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn readable(&self) -> bool {
        self.readable
    }

    pub fn numel(&self) -> usize {
        self.shape.size()
    }

    pub fn size(&self, dim: Option<usize>) -> Option<usize> {
        match dim {
            Some(d) if d < self.shape.data.len() => Some(self.shape.data[d]),
            None => Some(self.shape.size()),
            _ => None,
        }
    }

    pub fn buffer_array(&self) -> Vec<f32> {
        return self.buffer.into_array();
    }

    pub fn read_array(&self) -> Result<Vec<f32>, &'static str> {
        if self.readable {
            return Ok(self.buffer.into_array());
        }
        Err("Tensor is not readable")
    }

    pub fn read_float32(&self) -> Result<Vec<f32>, &'static str> {
        if self.readable {
            return Ok(self.buffer.into_array());
        }
        Err("Tensor is not readable")
    }

    fn create(array: Vec<f32>, shape: Vec<usize>, options: TensorOptions) -> Self {
        let mut size = shape;
        if size.len() < 1 {
            size = vec![array.len()];
        }
        Tensor {
            shape: Size::new(size),
            dtype: options.dtype,
            device: options.device,
            readable: options.readable,
            buffer: TensorBuffer::CPU(array),
        }
    }
}

impl From<(Vec<f32>, Vec<usize>)> for Tensor {
    fn from(data: (Vec<f32>, Vec<usize>)) -> Self {
        let len = data.0.len();
        let shape;
        if data.1.len() > 0 {
            shape = data.1;
        } else {
            shape = vec![len];
        }
        Tensor {
            shape: Size::new(shape),
            dtype: DType::Float32,
            device: Device::CPU,
            readable: true,
            buffer: TensorBuffer::CPU(data.0),
        }
    }
}

impl From<(Vec<i32>, Vec<usize>)> for Tensor {
    fn from(data: (Vec<i32>, Vec<usize>)) -> Self {
        let len = data.0.len();
        let shape;
        if data.1.len() > 0 {
            shape = data.1;
        } else {
            shape = vec![len];
        }
        Tensor {
            shape: Size::new(shape),
            dtype: DType::Float32,
            device: Device::CPU,
            readable: true,
            buffer: TensorBuffer::CPU(data.0.iter().map(|x| *x as f32).collect()),
        }
    }
}

impl From<Vec<f32>> for Tensor {
    fn from(data: Vec<f32>) -> Self {
        let len = data.len();
        let shape = vec![len];
        Tensor {
            shape: Size::new(shape),
            dtype: DType::Float32,
            device: Device::CPU,
            readable: true,
            buffer: TensorBuffer::CPU(data),
        }
    }
}

impl From<Vec<i32>> for Tensor {
    fn from(data: Vec<i32>) -> Self {
        let len = data.len();
        let shape = vec![len];
        Tensor {
            shape: Size::new(shape),
            dtype: DType::Float32,
            device: Device::CPU,
            readable: true,
            buffer: TensorBuffer::CPU(data.iter().map(|x| *x as f32).collect()),
        }
    }
}

impl From<Vec<Vec<f32>>> for Tensor {
    fn from(data: Vec<Vec<f32>>) -> Self {
        let shape = vec![data.len(), data[0].len()];
        let flattened: Vec<f32> = data.into_iter().flatten().collect();
        Tensor {
            shape: Size::new(shape),
            dtype: DType::Float32,
            device: Device::CPU,
            readable: true,
            buffer: TensorBuffer::CPU(flattened),
        }
    }
}

impl From<Vec<Vec<i32>>> for Tensor {
    fn from(data: Vec<Vec<i32>>) -> Self {
        let shape = vec![data.len(), data[0].len()];
        let flattened: Vec<f32> = data.into_iter().flatten().map(|x| x as f32).collect();
        Tensor {
            shape: Size::new(shape),
            dtype: DType::Float32,
            device: Device::CPU,
            readable: true,
            buffer: TensorBuffer::CPU(flattened),
        }
    }
}

impl From<Vec<Vec<Vec<f32>>>> for Tensor {
    fn from(data: Vec<Vec<Vec<f32>>>) -> Self {
        let shape = vec![data.len(), data[0].len(), data[0][0].len()];
        let flattened: Vec<f32> = data.into_iter().flatten().flatten().collect();
        Tensor {
            shape: Size::new(shape),
            dtype: DType::Float32,
            device: Device::CPU,
            readable: true,
            buffer: TensorBuffer::CPU(flattened),
        }
    }
}

impl From<Vec<Vec<Vec<i32>>>> for Tensor {
    fn from(data: Vec<Vec<Vec<i32>>>) -> Self {
        let shape = vec![data.len(), data[0].len(), data[0][0].len()];
        let flattened: Vec<f32> = data
            .into_iter()
            .flatten()
            .flatten()
            .map(|x| x as f32)
            .collect();
        Tensor {
            shape: Size::new(shape),
            dtype: DType::Float32,
            device: Device::CPU,
            readable: true,
            buffer: TensorBuffer::CPU(flattened),
        }
    }
}

impl fmt::Display for Tensor {
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
                    result.push_str(&" ".repeat(depth));
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
            TensorBuffer::CPU(data) => {
                write!(f, "tensor({})", format_nested(data, &self.shape.data, 0))
            }
            TensorBuffer::GPU(_) => write!(f, "tensor({:?})", self.shape),
        }
    }
}
