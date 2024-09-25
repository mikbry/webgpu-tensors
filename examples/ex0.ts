import t from '../src/webgpu-tensors'

const writeTensor = await t.tensor([[0, 1, 2, 3]]);
const readTensor = await t.empty([1, 4]);
await t.copy(writeTensor, readTensor);
t.print(readTensor);