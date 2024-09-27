import t, { Tensor, NN } from '../src/webgpu-tensors'

// Create a simple 2-layer neural network
class SimpleNN {
    w1: Tensor;
    w2: Tensor;
  
    constructor(w1: Tensor, w2: Tensor) {
      this.w1 = w1;
      this.w2 = w2;
    }

    static async init(inputSize: number, hiddenSize: number, outputSize: number) {
      const w1 = await t.randn([inputSize, hiddenSize]);
      const w2 = await t.randn([hiddenSize, outputSize]);
      return new SimpleNN(w1, w2);
    }
  
    async forward(x: Tensor): Promise<Tensor> {
        const h = await t.matmul(x, this.w1);
        const h_relu = await NN.relu(t, h);
        return await t.matmul(h_relu, this.w2);
    }
  }

const X = await t.rand([5, 3]);
const y = await t.rand([5, 3]);

const model = await SimpleNN.init(10, 5, 1);

// Training loop
const learningRate = 0.01;
const epochs = 100;

for (let epoch = 0; epoch < epochs; epoch++) {
  // Forward pass
  const yPred = await model.forward(X);

  // Compute loss (Mean Squared Error)
  const loss = (await yPred.sub(y)).pow(2).mean();

  // Backward pass (assuming we have autograd functionality)
  loss.backward();

  // Update weights (assuming we have an optimizer)
  model.w1 = model.w1.sub(model.w1.grad.mul(learningRate));
  model.w2 = model.w2.sub(model.w2.grad.mul(learningRate));

  // Print progress
  if (epoch % 10 === 0) {
    console.log(`Epoch ${epoch}, Loss: ${loss.item()}`);
  }
}

console.log('Training complete!');