import t from '../src/webgpu-tensors';

await t.init();
const writeTensor = t.tensor([0, 1, 2, 3]);
const readTensor = t.empty([4]);
t.copy(writeTensor, readTensor);
t.print(readTensor);
