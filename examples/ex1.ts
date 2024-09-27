import t from '../src/webgpu-tensors'

let x = await t.rand([5, 3]);
await t.print(x);

x = await t.ones([5, 3]);
await t.print(x);

x = await t.rand([5, 3]);
await t.print(x);

x = await t.tensor([0, 1, 2, 3]);
await t.print(x);

x = await t.tensor([[0, 1, 2, 3],[3, 4, 5, 6]]);
await t.print(x);

x = await t.tensor([[[0, 1], [2, 3]],[[3, 4], [5, 6]]]);
await t.print(x);

x = await t.tensor([[[[0], [1]], [[2], [3]]],[[[3], [4]], [[5], [6]]]]);
await t.print(x);

const tensor = await t.rand([3,4]);
t.print(`Shape of tensor: ${tensor.shape}`);
t.print(`Datatype of tensor: ${tensor.dtype}`);
t.print(`Device tensor is stored on: ${tensor.device}`);
await t.print("tensor", tensor, x);

x = await t.empty([3, 4, 5]);
t.print(x.size());
t.print(x.size(1));

t.reset();
x = await t.tensor([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]);
const y = await t.tensor([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]);
await t.print('matmul', x, y);
const result = await t.matmul(x, y);
t.print('matmul', x, y, result);

x = await t.tensor([-2, -1, 0, 1, 2, 3]);
const max = await t.maximum(x, 0);
await t.print("maximum=", max);

// Test for maximum function
const testTensor = await t.tensor([-3, -2, -1, 0, 1, 2, 3]);
const maxResult = await t.maximum(testTensor, 0);
await t.print("Test maximum([-3, -2, -1, 0, 1, 2, 3], 0) =", maxResult);
