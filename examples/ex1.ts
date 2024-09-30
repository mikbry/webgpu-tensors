import init from '../src/webgpu-tensors'

const t = await init();
let x = t.rand([5, 3]);
await t.print(x);

x = t.ones([5, 3]);
await t.print(x);

x = t.rand([5, 3]);
await t.print(x);

x = t.tensor([0, 1, 2, 3]);
await t.print(x);

x = t.tensor([[0, 1, 2, 3],[3, 4, 5, 6]]);
await t.print(x);

x = t.tensor([[[0, 1], [2, 3]],[[3, 4], [5, 6]]]);
await t.print(x);

x = t.tensor([[[[0], [1]], [[2], [3]]],[[[3], [4]], [[5], [6]]]]);
await t.print(x);

const tensor = t.rand([3,4]);
t.print(`Shape of tensor: ${tensor.shape}`);
t.print(`Datatype of tensor: ${tensor.dtype}`);
t.print(`Device tensor is stored on: ${tensor.device}`);
await t.print("tensor", tensor, x);

x = t.empty([3, 4, 5]);
t.print(x.size());
t.print(x.size(1));

t.reset();
x = t.tensor([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]);
const y = t.tensor([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]);
await t.print('matmul', x, y);
const result = t.matmul(x, y);
t.print('matmul', result);

x = t.tensor([[-3, -2, -1, 0], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]);
const max = t.maximum(x, 0);
await t.print("maximum=", max);

// Test for maximum function
const testTensor = t.tensor([[-3, -2, -1, 0], [0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 2, 3]]);
const mul = t.matmul(testTensor, x);
const maxResult = t.maximum(mul, 0);
await t.print("Test maximum=", maxResult);
