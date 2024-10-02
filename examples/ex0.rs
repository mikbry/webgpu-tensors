use webgpu_tensors::{RSTensors, Shape, Tensors};

fn main() {
    let mut t = RSTensors;
    t.init(None).unwrap();

    let shape: Shape = vec![2, 3];

    let writeTensor = t.tensor([0.0, 1.0, 2.0, 3.0].to_vec(), None);
    let readTensor = t.empty([4].to_vec(), None);

    t.copy(&writeTensor, &readTensor);

    t.print(&[readTensor]).unwrap();
}
