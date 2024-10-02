use webgpu_tensors::{RSTensors, Shape, Tensors, TensorOptions};

fn main() {
    let tensors = RSTensors;
    tensors.init(None).unwrap();

    let shape: Shape = vec![2, 3];
    let options = Some(TensorOptions {
        usage: 0,
        mapped_at_creation: None,
        readable: true,
    });

    let a = tensors.ones(shape.clone(), options.clone());
    let b = tensors.rand(shape.clone(), options);

    let c = tensors.matmul(&a, &b);

    tensors.print(&[a, b, c]).unwrap();
}
