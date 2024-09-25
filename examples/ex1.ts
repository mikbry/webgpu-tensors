import t from '../src/webgpu-tensors'

const x = await t.rand([5, 3]);
t.print(x);
