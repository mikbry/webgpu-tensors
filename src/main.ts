import './style.css'
import WebGPUTensors from './webgpu-tensors'

const ts = WebGPUTensors.create();

const writeTensor = await ts.tensor([[0, 1, 2, 3]], { usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC });
const readTensor = await ts.empty([4, 1], { usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });

await ts.copy(writeTensor, readTensor);
ts.compute();

const resultArray = await readTensor.readF16();

console.log(readTensor.shape, readTensor.dtype, readTensor.device, resultArray);