use webgpu_tensors::{RSTensors, Tensors, Tensor, tensors_println, Device};

fn main() {
    let mut t = RSTensors::new(Device::CPU);
    t.init(None).unwrap();

    // Rest of the code remains the same
    let x = t.rand(vec![5, 3], None);
    tensors_println!("{}", x);

    let x = t.ones(vec![5, 3], None);
    tensors_println!("{}", x);

    let x = t.rand(vec![5, 3], None);
    tensors_println!("{}", x);

    let x = t.tensor(vec![0.0, 1.0, 2.0, 3.0], None);
    tensors_println!("{}", x);

    let x = t.tensor(vec![vec![0.0, 1.0, 2.0, 3.0], vec![3.0, 4.0, 5.0, 6.0]], None);
    println!("{}", x);

    let x = t.tensor(vec![vec![vec![0,1],vec![2,3]],vec![vec![3,4],vec![5,6]]], None);
    println!("{}", x);

    let x = t.tensor((vec![0.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0], vec![2, 2, 2, 1]), None);
    println!("4D tensor:");
    println!("{}", x);

    let tensor = t.rand(vec![3, 4], None);
    println!("Shape of tensor: {:?}", tensor.shape());
    println!("Datatype of tensor: {:?}", tensor.dtype());
    println!("Device tensor is stored on: {:?}", tensor.device());
    println!("tensor: {}", tensor);

    let x = t.empty(vec![3, 4, 5], None);
    println!("Size of empty tensor: {}", x.numel());
    println!("Size of dimension 1: {}", x.size(Some(1)).unwrap());

    t.reset();
    let x = t.tensor((vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0], vec![4, 4]), None);
    let y = t.tensor((vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0], vec![4, 4]), None);
    println!("Matrix multiplication:");
    println!("x: {}", x);
    println!("y: {}", y);
    let result = t.matmul(&x, &y);
    println!("Result: {}", result);

    let x = t.tensor((vec![-3.0, -2.0, -1.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0], vec![4, 4]), None);
    let max = t.maximum(&x, 0.0);
    println!("Maximum (element-wise) with 0:");
    println!("{}", max);

    // Test for maximum function
    let test_tensor = t.tensor((vec![-3.0, -2.0, -1.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0], vec![4, 4]), None);
    let mul = t.matmul(&test_tensor, &x);
    let max_result = t.maximum(&mul, 0.0);
    println!("Test maximum:");
    println!("{}", max_result);

    t.destroy();

    // Now test with GPU
    let mut t = RSTensors::new(Device::GPU);
    t.init(None).unwrap();

    let x = t.rand(vec![5, 3], None);
    tensors_println!("GPU tensor: {} {:?}", x, x.device());

    t.destroy();
}
