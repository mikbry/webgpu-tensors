use webgpu_tensors::{RSTensors, Tensors, Tensor};

fn main() {
    let mut tensors = RSTensors;
    tensors.init(None).unwrap();

    let a = tensors.tensor(vec![1.0, 2.0, 3.0, 4.0], None);
    let b = tensors.tensor(vec![2.0, 3.0, 4.0, 5.0], None);

    println!("a: {}", a);
    println!("b: {}", b);

    let c = tensors.matmul(&a, &b);
    println!("c: {}", c);

    let d = tensors.ones(vec![2, 2], None);
    println!("d: {}", d);

    let e = tensors.zeros(vec![2, 2], None);
    println!("e: {}", e);

    let f = tensors.randn(vec![2, 2], None);
    println!("f: {}", f);

    tensors.destroy();
}
