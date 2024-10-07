use webgpu_tensors::{tensors_println, Device, RSTensor, RSTensors, Tensor, Tensors};
use std::time::Instant;

struct SimpleNN {
    w1: RSTensor,
    w2: RSTensor,
}

impl SimpleNN {
    fn new(w1: RSTensor, w2: RSTensor) -> Self {
        SimpleNN { w1, w2 }
    }

    fn init(t: &RSTensors, input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        let w1 = t.randn(vec![input_size, hidden_size], None);
        let w2 = t.randn(vec![hidden_size, output_size], None);
        SimpleNN::new(w1, w2)
    }

    fn forward(&self, t: &RSTensors, x: &RSTensor) -> RSTensor {
        let xw1 = t.matmul(x, &self.w1);
        let h = t.relu(&xw1);
        t.matmul(&h, &self.w2)
    }
}

fn main() {
    let mut t = RSTensors::new(Device::CPU);
    t.init(None).unwrap();

    tensors_println!("Train a simple 2-layer neural network");

    let start = Instant::now();

    // Generate random data
    let x = t.rand(vec![5, 10], None);
    let y = t.rand(vec![5, 1], None);

    // Initialize the model
    let mut model = SimpleNN::init(&t, 10, 5, 1);

    // Training loop
    let learning_rate = 0.01;
    let epochs = 100;

    for epoch in 0..=epochs {
        // Forward pass
        let mut y_pred = model.forward(&t, &x);

        // Compute loss (Mean Squared Error MSE)
        y_pred = t.sub(&y_pred, &y);
        let loss = t.mean(&t.pow(&y_pred, 2.0));

        // Backward pass (manual gradient computation)
        let grad_y_pred = t.mul_scalar(&t.sub(&y_pred, &y), 2.0 / y.numel() as f32);
        let h_relu = t.relu(&t.matmul(&x, &model.w1));
        let grad_w2 = t.matmul(&t.transpose(&h_relu), &grad_y_pred);
        let grad_h = t.matmul(&grad_y_pred, &t.transpose(&model.w2));
        let mask = t.gt(&t.matmul(&x, &model.w1), 0.0);
        let grad_w1 = t.matmul(&t.transpose(&x), &t.mul(&grad_h, &mask));

        // Update weights
        model.w1 = t.sub(&model.w1, &t.mul_scalar(&grad_w1, learning_rate));
        model.w2 = t.sub(&model.w2, &t.mul_scalar(&grad_w2, learning_rate));

        // Print progress
        if epoch % 10 == 0 {
            let loss_value = t.item(&loss);
            tensors_println!("Epoch {}, Loss: {}", epoch, loss_value);
        }
    }

    let duration = start.elapsed();
    tensors_println!("Training complete! in {} ms", duration.as_millis());

    t.destroy();
}
