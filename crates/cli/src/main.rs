use webgpu_tensors::{tensors_println, Device, RSTensors, Shape, TensorOptions, Tensors};

fn main() {
    let mut tensors = RSTensors::new(Device::CPU);
    tensors.init(None).unwrap();

    let shape: Shape = vec![2, 3];
    let options = Some(TensorOptions::default());

    let a = tensors.ones(shape.clone(), options.clone());
    let b = tensors.rand(shape.clone(), options);

    let c = tensors.matmul(&a, &b);

    tensors_println!("{:} {:} {:}", a, b, c);

    println!("Tensor operations completed successfully!");
}
