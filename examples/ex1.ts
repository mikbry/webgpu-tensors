import t from '../src/webgpu-tensors'

let x = await t.rand([5, 3]);
t.print(x);

x = await t.tensor([0, 1, 2, 3]);
t.print(x);

x = await t.tensor([[0, 1, 2, 3],[3, 4, 5, 6]]);
t.print(x);

x = await t.tensor([[[0, 1], [2, 3]],[[3, 4], [5, 6]]]);
t.print(x);

x = await t.tensor([[[[0], [1]], [[2], [3]]],[[[3], [4]], [[5], [6]]]]);
t.print(x);