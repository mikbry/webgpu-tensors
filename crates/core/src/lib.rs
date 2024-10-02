use std::fmt;

#[derive(Debug, Clone, Copy)]
pub enum DType {
    Float32,
    // Add other data types as needed
}

#[derive(Debug, Clone, Copy)]
pub enum Device {
    GPU,
    CPU,
}

pub type Shape = Vec<usize>;

#[derive(Debug, Clone)]
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
}

pub trait Tensors {
    fn init(&mut self, device: Option<Device>) -> Result<(), &'static str>;
    fn empty(&self, shape: Shape, options: Option<TensorOptions>) -> Box<dyn Tensor>;
    fn ones(&self, shape: Shape, options: Option<TensorOptions>) -> Box<dyn Tensor>;
    fn rand(&self, shape: Shape, options: Option<TensorOptions>) -> Box<dyn Tensor>;
    fn randn(&self, shape: Shape, options: Option<TensorOptions>) -> Box<dyn Tensor>;
    fn zeros(&self, shape: Shape, options: Option<TensorOptions>) -> Box<dyn Tensor>;
    fn tensor(&self, array: Vec<f32>, options: Option<TensorOptions>) -> Box<dyn Tensor>;
    fn matmul(&self, tensor_a: &dyn Tensor, tensor_b: &dyn Tensor) -> Box<dyn Tensor>;
    fn print(&self, data: &[Box<dyn std::fmt::Debug>]) -> Result<(), &'static str>;
}

pub struct TensorOptions {
    pub usage: u32,
    pub mapped_at_creation: Option<bool>,
    pub readable: bool,
}

// Implement RSTensor and RSTensors as structs
pub struct RSTensor {
    data: Vec<f32>,
    shape: Size,
    dtype: DType,
    device: Device,
    readable: bool,
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
}

pub struct RSTensors;

impl Tensors for RSTensors {
    fn init(&mut self, _device: Option<Device>) -> Result<(), &'static str> {
        Ok(()) // No initialization needed for JS implementation
    }

    fn empty(&self, shape: Shape, _options: Option<TensorOptions>) -> Box<dyn Tensor> {
        let size = Size::new(shape.clone());
        Box::new(RSTensor {
            data: vec![0.0; size.size()],
            shape: size,
            dtype: DType::Float32,
            device: Device::CPU,
            readable: true,
        })
    }

    fn ones(&self, shape: Shape, _options: Option<TensorOptions>) -> Box<dyn Tensor> {
        let size = Size::new(shape.clone());
        Box::new(RSTensor {
            data: vec![1.0; size.size()],
            shape: size,
            dtype: DType::Float32,
            device: Device::CPU,
            readable: true,
        })
    }

    fn rand(&self, shape: Shape, _options: Option<TensorOptions>) -> Box<dyn Tensor> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let size = Size::new(shape.clone());
        Box::new(RSTensor {
            data: (0..size.size()).map(|_| rng.gen::<f32>()).collect(),
            shape: size,
            dtype: DType::Float32,
            device: Device::CPU,
            readable: true,
        })
    }

    fn randn(&self, shape: Shape, _options: Option<TensorOptions>) -> Box<dyn Tensor> {
        use rand::distributions::{Distribution, Standard};
        use rand::thread_rng;
        let mut rng = thread_rng();
        let size = Size::new(shape.clone());
        Box::new(RSTensor {
            data: Standard.sample_iter(&mut rng).take(size.size()).collect(),
            shape: size,
            dtype: DType::Float32,
            device: Device::CPU,
            readable: true,
        })
    }

    fn zeros(&self, shape: Shape, _options: Option<TensorOptions>) -> Box<dyn Tensor> {
        let size = Size::new(shape.clone());
        Box::new(RSTensor {
            data: vec![0.0; size.size()],
            shape: size,
            dtype: DType::Float32,
            device: Device::CPU,
            readable: true,
        })
    }

    fn tensor(&self, array: Vec<f32>, _options: Option<TensorOptions>) -> Box<dyn Tensor> {
        let shape = vec![array.len()];
        Box::new(RSTensor {
            data: array,
            shape: Size::new(shape),
            dtype: DType::Float32,
            device: Device::CPU,
            readable: true,
        })
    }

    fn matmul(&self, tensor_a: &dyn Tensor, tensor_b: &dyn Tensor) -> Box<dyn Tensor> {
        // Implement matrix multiplication
        // This is a simplified version and doesn't handle all cases
        let a = tensor_a.shape();
        let b = tensor_b.shape();
        let m = a.get_dim(0).unwrap();
        let n = a.get_dim(1).unwrap();
        let p = b.get_dim(1).unwrap();

        let mut result = vec![0.0; m * p];

        // Implement the actual matrix multiplication here
        // For simplicity, we're not implementing the actual algorithm

        Box::new(RSTensor {
            data: result,
            shape: Size::new(vec![m, p]),
            dtype: DType::Float32,
            device: Device::CPU,
            readable: true,
        })
    }

    fn print(&self, data: &[Box<dyn std::fmt::Debug>]) -> Result<(), &'static str> {
        for item in data {
            println!("{:?}", item);
        }
        Ok(())
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
