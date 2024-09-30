import init from '../src/webgpu-tensors'

const t = await init();
const writeTensor = t.tensor([0, 1, 2, 3]);
const readTensor = t.empty([4]);
t.copy(writeTensor, readTensor);
t.print(readTensor);