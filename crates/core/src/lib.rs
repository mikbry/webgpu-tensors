use std::fmt;
use serde::{Serialize, Deserialize};
use wgpu::Buffer;

mod tensors;
pub use tensors::cpu::CPUTensors;
pub use tensors::webgpu::WGPUTensors;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum DType {
    Float32,
    // Add other data types as needed
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Device {
    GPU,
    CPU,
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

pub trait Tensor {
    fn shape(&self) -> &Size;
    fn dtype(&self) -> DType;
    fn device(&self) -> Device;
    fn readable(&self) -> bool;
    fn size(&self, dim: Option<usize>) -> Result<usize, &'static str>;
    fn numel(&self) -> usize;
    fn read_array(&self) -> Result<Vec<f32>, &'static str>;
    fn read_float32(&self) -> Result<Vec<f32>, &'static str>;
}

pub trait Tensors {
    fn clone(&self, tensor: &RSTensor) -> RSTensor;
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
    fn init(&mut self, device: Option<Device>) -> Result<(), &'static str>;
    fn reset(&mut self);
    fn compute(&mut self);
    fn destroy(&mut self);

    fn empty(&self, shape: Shape, options: Option<TensorOptions>) -> RSTensor;
    fn ones(&self, shape: Shape, options: Option<TensorOptions>) -> RSTensor;
    fn rand(&self, shape: Shape, options: Option<TensorOptions>) -> RSTensor;
    fn randn(&self, shape: Shape, options: Option<TensorOptions>) -> RSTensor;
    fn zeros(&self, shape: Shape, options: Option<TensorOptions>) -> RSTensor;
    fn tensor<T: Into<RSTensor>>(&self, array: T, options: Option<TensorOptions>) -> RSTensor;
    fn matmul(&self, tensor_a: &RSTensor, tensor_b: &RSTensor) -> RSTensor;
    fn copy(&self, src: &RSTensor, dst: &mut RSTensor) -> Result<(), &'static str>;
    fn maximum(&self, tensor: &RSTensor, value: f32) -> RSTensor;
    fn max(&self, tensor: &RSTensor) -> RSTensor;
}

#[macro_export]
macro_rules! tensors_println {
    ($($arg:tt)*) => {
        print!($($arg)*);
        println!();
    };
}

#[derive(Debug, Clone, Copy)]
pub struct TensorOptions {
    pub usage: u32,
    pub mapped_at_creation: Option<bool>,
    pub readable: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RSTensor {
    pub(crate) data: Vec<f32>,
    pub(crate) shape: Size,
    pub(crate) dtype: DType,
    pub(crate) device: Device,
    pub(crate) readable: bool,
    pub(crate) buffer: Option<Buffer>,
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

    fn size(&self, dim: Option<usize>) -> Result<usize, &'static str> {
        match dim {
            Some(d) => self.shape.get_dim(d).ok_or("Dimension out of range"),
            None => Ok(self.shape.size()),
        }
    }

    fn numel(&self) -> usize {
        self.shape.size()
    }

    fn read_array(&self) -> Result<Vec<f32>, &'static str> {
        if self.readable {
            Ok(self.data.clone())
        } else {
            Err("Tensor is not readable")
        }
    }

    fn read_float32(&self) -> Result<Vec<f32>, &'static str> {
        if self.readable {
            Ok(self.data.clone())
        } else {
            Err("Tensor is not readable")
        }
    }
}

impl From<Vec<f32>> for RSTensor {
    fn from(data: Vec<f32>) -> Self {
        let len = data.len();
        RSTensor {
            data,
            shape: Size::new(vec![len]),
            dtype: DType::Float32,
            device: Device::CPU,
            readable: true,
            buffer: None,
        }
    }
}

impl From<Vec<Vec<f32>>> for RSTensor {
    fn from(data: Vec<Vec<f32>>) -> Self {
        let shape = vec![data.len(), data[0].len()];
        let flattened: Vec<f32> = data.into_iter().flatten().collect();
        RSTensor {
            data: flattened,
            shape: Size::new(shape),
            dtype: DType::Float32,
            device: Device::CPU,
            readable: true,
            buffer: None,
        }
    }
}

impl From<Vec<Vec<Vec<f32>>>> for RSTensor {
    fn from(data: Vec<Vec<Vec<f32>>>) -> Self {
        let shape = vec![data.len(), data[0].len(), data[0][0].len()];
        let flattened: Vec<f32> = data.into_iter().flatten().flatten().collect();
        RSTensor {
            data: flattened,
            shape: Size::new(shape),
            dtype: DType::Float32,
            device: Device::CPU,
            readable: true,
            buffer: None,
        }
    }
}

impl From<Vec<Vec<Vec<Vec<f32>>>>> for RSTensor {
    fn from(data: Vec<Vec<Vec<Vec<f32>>>>) -> Self {
        let shape = vec![data.len(), data[0].len(), data[0][0].len(), data[0][0][0].len()];
        let flattened: Vec<f32> = data.into_iter().flatten().flatten().flatten().collect();
        RSTensor {
            data: flattened,
            shape: Size::new(shape),
            dtype: DType::Float32,
            device: Device::CPU,
            readable: true,
            buffer: None,
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
                result.push_str(&format_nested(&data[start..end], &shape[1..], depth + 1));
            }
            result.push(']');
            result
        }

        write!(f, "{}", format_nested(&self.data, &self.shape.data, 0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_size() {
        let size = Size::new(vec![2, 3, 4]);
        assert_eq!(size.length(), 3);
        assert_eq!(size.size(), 24);
        assert_eq!(size.get_dim(1), Some(3));
        assert_eq!(size.to_string(), "[2, 3, 4]");
    }

    // Add more tests for other structs and traits
}
