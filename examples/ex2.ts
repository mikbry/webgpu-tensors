import t, { Tensor } from '../src/webgpu-tensors'

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
        const w1 = await t.randn([inputSize, hiddenSize]);
        const w2 = await t.randn([hiddenSize, outputSize]);
        return new SimpleNN(w1, w2);
    }

    async forward(x: Tensor): Promise<Tensor> {
        const xW1 = await t.matmul(x, this.w1);
        const h = await t.relu(xW1);
        const result = await t.matmul(h, this.w2);
        return result;
    }
}
const start = performance.now();
// Generate random data
const X = await t.rand([5, 10]);
const y = await t.rand([5, 1]);
// Initialize the model
const model = await SimpleNN.init(10, 5, 1);

// Training loop
const learningRate = 0.01;
const epochs = 100;

for (let epoch = 0; epoch <= epochs; epoch++) {
    // Forward pass
    let yPred = await model.forward(X);

    // Compute loss (Mean Squared Error MSE)
    yPred = await t.sub(yPred, y); 
    const loss = await t.mean(await t.pow(yPred, 2));

    // Backward pass (manual gradient computation)
    const gradYPred = await t.mul(await t.sub(yPred, y), 2.0 / y.numel());
    const hRelu = await t.relu(await t.matmul(X, model.w1));
    const gradW2 = await t.matmul(await t.transpose(hRelu), gradYPred);
    const gradH = await t.matmul(gradYPred, await t.transpose(model.w2));
    const mask = await t.gt(await t.matmul(X, model.w1), 0);
    const gradW1 = await t.matmul(await t.transpose(X), await t.mul(gradH, mask));

    // Update weights
    model.w1 = await t.sub(model.w1, await t.mul(gradW1, learningRate));
    model.w2 = await t.sub(model.w2, await t.mul(gradW2, learningRate));

    // Print progress
    if (epoch % 10 === 0) {
        await t.print(`Epoch ${epoch}, Loss: ${await t.item(loss)}`);
    }
}
const end = performance.now();
await t.print('Training complete! in', (end - start), 'ms');
