import './style.css';
import { Device, device } from '@/webgpu-tensors';

/* const writeTensor = await t.tensor([[0, 1, 2, 3]]);
console.log('writeTensor', writeTensor);
const readTensor = await t.empty([1, 4]);
console.log('writeTensor', readTensor);

// t.compute();
await t.copy(writeTensor, readTensor);

t.print(readTensor); */

const t = await device(Device.WASM);
const x = t.rand([5, 3]);
t.print(x);
const xx = t.clone(x);
console.log(xx);
let array = await xx.readArray();
console.log('array=', array);
array = await x.readArray();
console.log('array=', array);

const a = t.tensor([
  [1, 2],
  [3, 4],
]);
const b = t.tensor([
  [5, 6],
  [7, 8],
]);
console.log('a', a);
const result = t.matmul(a, b);
console.log('result', result);
