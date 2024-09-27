import t, { Tensor } from '../src/webgpu-tensors'

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
        t.print('h', h)
        return await t.matmul(h, this.w2);
    }
}

// Generate random data
const X = await t.rand([5, 10]);
const y = await t.rand([5, 1]);

// Initialize the model
const model = await SimpleNN.init(10, 5, 1);

// Training loop
const learningRate = 0.01;
const epochs = 100;

for (let epoch = 0; epoch < epochs; epoch++) {
    // Forward pass
    let yPred = await model.forward(X);
    t.print('yPred', yPred);

    // Compute loss (Mean Squared Error)
    yPred = await t.sub(yPred, y);

    let loss = await t.pow(yPred, 2)
    loss = await t.mean(loss);

    // Backward pass (manual gradient computation)
    const grad_y_pred = await t.sub(yPred, y);
    const grad_y_pred_scaled = await t.mul(grad_y_pred, 2.0 / y.numel());
    const h_relu = await t.relu(await t.matmul(X, model.w1));
    const grad_w2 = await t.matmul(await t.transpose(h_relu), grad_y_pred_scaled);
    const grad_h = await t.matmul(grad_y_pred_scaled, await t.transpose(model.w2));
    const h = await t.matmul(X, model.w1);
    const mask = await t.gt(h, 0);
    const grad_w1 = await t.matmul(await t.transpose(X), await t.mul(grad_h, mask));

    // Update weights
    model.w1 = await t.sub(model.w1, await t.mul(grad_w1, learningRate));
    model.w2 = await t.sub(model.w2, await t.mul(grad_w2, learningRate));

    // Print progress
    if (epoch % 10 === 0) {
        t.print(`Epoch ${epoch}, Loss: ${await t.item(loss)}`);
    }
}

t.print('Training complete!');
