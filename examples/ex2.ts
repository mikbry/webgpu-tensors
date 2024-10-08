import t, { Tensor } from '@/webgpu-tensors';

await t.init();
await t.print('Train a simple 2-layer neural network');
// Create a simple 2-layer neural network
class SimpleNN {
  w1: Tensor;
  w2: Tensor;

  constructor(w1: Tensor, w2: Tensor) {
    this.w1 = w1;
    this.w2 = w2;
  }

  static async init(inputSize: number, hiddenSize: number, outputSize: number) {
    const w1 = t.randn([inputSize, hiddenSize]);
    const w2 = t.randn([hiddenSize, outputSize]);
    return new SimpleNN(w1, w2);
  }

  async forward(x: Tensor): Promise<Tensor> {
    const xW1 = t.matmul(x, this.w1);
    const h = t.relu(xW1);
    const result = t.matmul(h, this.w2);
    return result;
  }
}
const start = performance.now();
// Generate random data
const X = t.rand([5, 10]);
const y = t.rand([5, 1]);
// Initialize the model
const model = await SimpleNN.init(10, 5, 1);

// Training loop
const learningRate = 0.01;
const epochs = 100;

for (let epoch = 0; epoch <= epochs; epoch++) {
  // Forward pass
  let yPred = await model.forward(X);

  // Compute loss (Mean Squared Error MSE)
  yPred = t.sub(yPred, y);
  const loss = t.mean(t.pow(yPred, 2));

  // Backward pass (manual gradient computation)
  const gradYPred = t.mul(t.sub(yPred, y), 2.0 / y.numel());
  const hRelu = t.relu(t.matmul(X, model.w1));
  const gradW2 = t.matmul(t.transpose(hRelu), gradYPred);
  const gradH = t.matmul(gradYPred, t.transpose(model.w2));
  const mask = t.gt(t.matmul(X, model.w1), 0);
  const gradW1 = t.matmul(t.transpose(X), t.mul(gradH, mask));

  // Update weights
  model.w1 = t.sub(model.w1, t.mul(gradW1, learningRate));
  model.w2 = t.sub(model.w2, t.mul(gradW2, learningRate));

  // Print progress
  if (epoch % 10 === 0) {
    await t.print(`Epoch ${epoch}, Loss: ${await t.item(loss)}`);
  }
}
const end = performance.now();
await t.print('Training complete! in', end - start, 'ms');
