use webgpu_tensors::{tensors_println, Device, Tensors, Shape, TensorOptions, TensorsOperations};

fn main() {
    let mut tensors = Tensors::new(Device::CPU);
    tensors.init(None).unwrap();

    let shape: Shape = vec![2, 3];
    let options = Some(TensorOptions::default());

    let a = tensors.ones(shape.clone(), options.clone());
    let b = tensors.rand(shape.clone(), options);

    let c = tensors.matmul(&a, &b);

    tensors_println!("{:} {:} {:}", a, b, c);

    println!("Tensor operations completed successfully!");
}
