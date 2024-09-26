import './style.css'
import t from './webgpu-tensors'

/* const writeTensor = await t.tensor([[0, 1, 2, 3]]);
console.log('writeTensor', writeTensor);
const readTensor = await t.empty([1, 4]);
console.log('writeTensor', readTensor);

// t.compute();
await t.copy(writeTensor, readTensor);

t.print(readTensor); */

const x = await t.rand([5, 3]);
t.print(x);