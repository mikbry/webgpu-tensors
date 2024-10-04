import { Device, DType, NestedArray, Shape, Tensor, TensorOptions, Tensors } from './types';
import { JSTensors } from './tensors/js';
import WebGPUTensors from './tensors/webgpu';
import { WASMTensors } from './tensors/wasm';

class Framework implements Tensors {
  device: Device;

  implementation: Tensors;

  constructor(device: Device, implementation: Tensors) {
    this.device = device;
    this.implementation = implementation;
  }

  init(device?: Device) { return this.implementation.init(device);}
  empty(shape: Shape, options?: Partial<TensorOptions> | undefined) { return this.implementation.empty(shape, options);}
  ones(shape: Shape, options?: Partial<TensorOptions> | undefined) { return this.implementation.ones(shape, options);}
  rand(shape: Shape, options?: Partial<TensorOptions> | undefined) { return this.implementation.rand(shape, options);}
  randn(shape: Shape, options?: Partial<TensorOptions> | undefined) { return this.implementation.randn(shape, options);}
  zeros(shape: Shape, options?: Partial<TensorOptions> | undefined) { return this.implementation.zeros(shape, options);}
  tensor(array: NestedArray<number>, options?: Partial<TensorOptions> | undefined) { return this.implementation.tensor(array, options);}
  matmul(tensorA: Tensor, tensorB: Tensor) { return this.implementation.matmul(tensorA, tensorB);}
  sub(tensorA: Tensor, tensorB: Tensor) { return this.implementation.sub(tensorA, tensorB);}
  pow(tensor: Tensor, exponent: number) { return this.implementation.pow(tensor, exponent);}
  mul(tensorA: Tensor, tensorB: Tensor | number) { return this.implementation.mul(tensorA, tensorB);}
  gt(tensor: Tensor, value: number) { return this.implementation.gt(tensor, value);}
  transpose(tensor: Tensor) { return this.implementation.transpose(tensor);}
  maximum(tensor: Tensor, value: number) { return this.implementation.maximum(tensor, value);}
  relu(x: Tensor) { return this.implementation.relu(x);}
  max(tensor: Tensor) { return this.implementation.max(tensor);}
  mean(tensor: Tensor) { return this.implementation.mean(tensor);}
  sigmoid(tensor: Tensor) { return this.implementation.sigmoid(tensor);}
  clone(tensor: Tensor) { return this.implementation.clone(tensor);}
  copy(tensorSource: Tensor, tensorDestination: Tensor) { return this.implementation.copy(tensorSource, tensorDestination);}
  item(tensor: Tensor) { return this.implementation.item(tensor);}
  reset() { return this.implementation.reset();}
  compute() { return this.implementation.compute();}
  destroy() { return this.implementation.destroy();}
  print(...data: unknown[]) { return this.implementation.print(...data);}
}



export function isAvailable(device: Device): boolean {
  if (device === Device.GPU) {
    return WebGPUTensors.isAvailable();
  } else if (device === Device.CPU) {
    return true;
  } else {
    return false;
  }
}

export function device(
  device: Device | undefined = WebGPUTensors.isAvailable() ? Device.GPU : Device.CPU,
): Tensors {

  let implementation: Tensors;
  if (device === Device.GPU) {
    implementation = WebGPUTensors.create();
  } else if (device === Device.CPU) {
    implementation = JSTensors.create();

  } else if (device === Device.WASM) {
    implementation = WASMTensors.create();
  } else {
    throw new Error('Unknown device ' + device);
  }
  const framework = new Framework(device, implementation);
  framework.device = device;
  framework.init();
  return framework;
}

export default device();

export { DType };
export type { Tensors as Tensors, Tensor };
