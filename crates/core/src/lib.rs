use std::fmt;
use rand::Rng;
use rand::distributions::{Distribution, Standard};
use std::convert::From;
use serde::{Serialize, Deserialize};

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
    fn read_float32(&self) -> Result<Vec<f32>, &'static str>;
    fn shape(&self) -> &Size;
    fn dtype(&self) -> DType;
    fn device(&self) -> Device;
    fn readable(&self) -> bool;
    fn size(&self, dim: Option<usize>) -> Result<usize, &'static str>;
    fn numel(&self) -> usize;
    fn read_array(&self) -> Result<Vec<f32>, &'static str>;
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
    data: Vec<f32>,
    shape: Size,
    dtype: DType,
    device: Device,
    readable: bool,
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
        }
    }
}

impl fmt::Display for RSTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor{:?}", self.data)
    }
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
            Ok(create_nested_array(&self.data, &self.shape.data))
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

fn create_nested_array(data: &[f32], shape: &[usize]) -> Vec<f32> {
    if shape.len() == 1 {
        return data.to_vec();
    }

    let mut result = Vec::new();
    let sub_size: usize = shape[1..].iter().product();
    
    for i in 0..shape[0] {
        let start = i * sub_size;
        let end = start + sub_size;
        let sub_array = create_nested_array(&data[start..end], &shape[1..]);
        result.push(sub_array);
    }

    result
}

pub struct RSTensors;


impl Tensors for RSTensors {
    fn init(&mut self, _device: Option<Device>) -> Result<(), &'static str> {
        Ok(()) // No initialization needed for RS implementation
    }

    fn reset(&mut self) {
        // TODO
    }

    fn compute(&mut self) {
        // TODO
    }

    fn destroy(&mut self) {
        // alert("Hello, wasm!");
    }

    fn empty(&self, shape: Shape, _options: Option<TensorOptions>) -> RSTensor {
        let size = Size::new(shape.clone());
        RSTensor {
            data: vec![0.0; size.size()],
            shape: size,
            dtype: DType::Float32,
            device: Device::CPU,
            readable: true,
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
        }
    }

    fn randn(&self, shape: Shape, _options: Option<TensorOptions>) -> RSTensor {
        let mut rng = rand::thread_rng();
        let size = Size::new(shape.clone());
        RSTensor {
            data: Standard.sample_iter(&mut rng).take(size.size()).collect(),
            shape: size,
            dtype: DType::Float32,
            device: Device::CPU,
            readable: true,
        }
    }

    fn zeros(&self, shape: Shape, _options: Option<TensorOptions>) -> RSTensor {
        let size = Size::new(shape.clone());
        RSTensor {
            data: vec![0.0; size.size()],
            shape: size,
            dtype: DType::Float32,
            device: Device::CPU,
            readable: true,
        }
    }

    fn create_nested_array(&self, data: &[f32], shape: &[usize]) -> Vec<f32> {
        if shape.len() == 1 {
            return data.to_vec();
        }

        let mut result = Vec::new();
        let sub_size: usize = shape[1..].iter().product();
        
        for i in 0..shape[0] {
            let start = i * sub_size;
            let end = start + sub_size;
            let sub_array = self.create_nested_array(&data[start..end], &shape[1..]);
            result.push(sub_array);
        }

        result
    }

    fn tensor<T: Into<RSTensor>>(&self, array: T, _options: Option<TensorOptions>) -> RSTensor {
        array.into()
    }

    fn matmul(&self, tensor_a: &RSTensor, tensor_b: &RSTensor) -> RSTensor {
        // Implement matrix multiplication
        let a = tensor_a.shape();
        let b = tensor_b.shape();

        let m = a.get_dim(0).unwrap();
        let n = a.get_dim(1).unwrap_or(0);
        let p = b.get_dim(1).unwrap_or(0);

        let mut result = vec![0.0; m * p];

        // Implement the actual matrix multiplication
        for i in 0..m {
            for j in 0..p {
                for k in 0..n {
                    result[i * p + j] += tensor_a.data[i * n + k] * tensor_b.data[k * p + j];
                }
            }
        }

        RSTensor {
            data: result,
            shape: Size::new(vec![m, p]),
            dtype: DType::Float32,
            device: Device::CPU,
            readable: true,
        }
    }

    fn copy(&self, src: &RSTensor, dst: &mut RSTensor) -> Result<(), &'static str> {
        if src.shape() != dst.shape() {
            return Err("Source and destination tensors must have the same shape");
        }
        dst.data.copy_from_slice(&src.data);
        Ok(())
    }

    fn maximum(&self, tensor: &RSTensor, value: f32) -> RSTensor {
        let data = tensor.data.iter().map(|&x| x.max(value)).collect();
        RSTensor {
            data,
            shape: tensor.shape().clone(),
            dtype: tensor.dtype(),
            device: tensor.device(),
            readable: tensor.readable(),
        }
    }

    fn relu(&self, tensor: &RSTensor) -> RSTensor {
        self.maximum(tensor, 0.0)
    }

    fn sub(&self, a: &RSTensor, b: &RSTensor) -> RSTensor {
        assert_eq!(a.shape(), b.shape(), "Tensors must have the same shape for subtraction");
        let data: Vec<f32> = a.data.iter().zip(b.data.iter()).map(|(&x, &y)| x - y).collect();
        RSTensor {
            data,
            shape: a.shape().clone(),
            dtype: a.dtype(),
            device: a.device(),
            readable: a.readable(),
        }
    }

    fn pow(&self, tensor: &RSTensor, exponent: f32) -> RSTensor {
        let data: Vec<f32> = tensor.data.iter().map(|&x| x.powf(exponent)).collect();
        RSTensor {
            data,
            shape: tensor.shape().clone(),
            dtype: tensor.dtype(),
            device: tensor.device(),
            readable: tensor.readable(),
        }
    }

    fn mean(&self, tensor: &RSTensor) -> RSTensor {
        let sum: f32 = tensor.data.iter().sum();
        let mean = sum / tensor.numel() as f32;
        RSTensor {
            data: vec![mean],
            shape: Size::new(vec![1]),
            dtype: tensor.dtype(),
            device: tensor.device(),
            readable: tensor.readable(),
        }
    }

    fn mul(&self, a: &RSTensor, b: &RSTensor) -> RSTensor {
        assert_eq!(a.shape(), b.shape(), "Tensors must have the same shape for element-wise multiplication");
        let data: Vec<f32> = a.data.iter().zip(b.data.iter()).map(|(&x, &y)| x * y).collect();
        RSTensor {
            data,
            shape: a.shape().clone(),
            dtype: a.dtype(),
            device: a.device(),
            readable: a.readable(),
        }
    }

    fn sigmoid(&self, tensor: &RSTensor) -> RSTensor {
        let data: Vec<f32> = tensor.data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
        RSTensor {
            data,
            shape: tensor.shape().clone(),
            dtype: tensor.dtype(),
            device: tensor.device(),
            readable: tensor.readable(),
        }
    }

    fn mul_scalar(&self, a: &RSTensor, b: f32) -> RSTensor {
        let data: Vec<f32> = a.data.iter().map(|&x| x * b).collect();
        RSTensor {
            data,
            shape: a.shape().clone(),
            dtype: a.dtype(),
            device: a.device(),
            readable: a.readable(),
        }
    }

    fn transpose(&self, tensor: &RSTensor) -> RSTensor {
        let shape = tensor.shape();
        assert_eq!(shape.length(), 2, "Transpose is only implemented for 2D tensors");
        let rows = shape.get_dim(0).unwrap();
        let cols = shape.get_dim(1).unwrap();
        let mut data = vec![0.0; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                data[j * rows + i] = tensor.data[i * cols + j];
            }
        }
        RSTensor {
            data,
            shape: Size::new(vec![cols, rows]),
            dtype: tensor.dtype(),
            device: tensor.device(),
            readable: tensor.readable(),
        }
    }

    fn gt(&self, tensor: &RSTensor, value: f32) -> RSTensor {
        let data: Vec<f32> = tensor.data.iter().map(|&x| if x > value { 1.0 } else { 0.0 }).collect();
        RSTensor {
            data,
            shape: tensor.shape().clone(),
            dtype: tensor.dtype(),
            device: tensor.device(),
            readable: tensor.readable(),
        }
    }

    fn clone(&self, tensor: &RSTensor) -> RSTensor {
        RSTensor {
            data: tensor.data.clone(),
            shape: tensor.shape().clone(),
            dtype: tensor.dtype(),
            device: tensor.device(),
            readable: tensor.readable(),
        }
    }

    fn item(&self, tensor: &RSTensor) -> f32 {
        assert_eq!(tensor.numel(), 1, "Tensor must contain a single element");
        tensor.data[0]
    }

    fn max(&self, tensor: &RSTensor) -> RSTensor {
        let max_value = tensor.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        RSTensor {
            data: vec![max_value],
            shape: Size::new(vec![1]),
            dtype: tensor.dtype(),
            device: tensor.device(),
            readable: tensor.readable(),
        }
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
