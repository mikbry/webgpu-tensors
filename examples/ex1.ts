import t from '../src/webgpu-tensors'

let x = await t.rand([5, 3]);
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

x = await t.empty([3, 4, 5]);
t.print(x.size());
t.print(x.size(1));

t.reset();
x = await t.rand([4, 4]);
let y = await t.rand([4, 4]);
y = await t.matmul(x, y);
t.print('matmul', y);
