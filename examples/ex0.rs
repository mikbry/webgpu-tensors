use webgpu_tensors::{RSTensors, Tensors};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut t = RSTensors;
    t.init(None)?;

    let write_tensor = t.tensor(vec![0.0, 1.0, 2.0, 3.0], None);
    let mut read_tensor = t.empty(vec![4], None);

    t.copy(&write_tensor, &mut read_tensor)?;
    t.print(&[read_tensor])?;

    Ok(())
}
