import { Device, DType, NestedArray, Shape, Tensor, TensorOptions, Tensors } from './types';
import { JSTensors } from './tensors/js';
import WebGPUTensors from './tensors/webgpu';

class Framework implements Tensors {
  static implementation: Tensors;

  init(device?: Device) { return Framework.implementation.init(device);}
  empty(shape: Shape, options?: Partial<TensorOptions> | undefined) { return Framework.implementation.empty(shape, options);}
  ones(shape: Shape, options?: Partial<TensorOptions> | undefined) { return Framework.implementation.ones(shape, options);}
  rand(shape: Shape, options?: Partial<TensorOptions> | undefined) { return Framework.implementation.rand(shape, options);}
  randn(shape: Shape, options?: Partial<TensorOptions> | undefined) { return Framework.implementation.randn(shape, options);}
  zeros(shape: Shape, options?: Partial<TensorOptions> | undefined) { return Framework.implementation.zeros(shape, options);}
  tensor(array: NestedArray<number>, options?: Partial<TensorOptions> | undefined) { return Framework.implementation.tensor(array, options);}
  matmul(tensorA: Tensor, tensorB: Tensor) { return Framework.implementation.matmul(tensorA, tensorB);}
  sub(tensorA: Tensor, tensorB: Tensor) { return Framework.implementation.sub(tensorA, tensorB);}
  pow(tensor: Tensor, exponent: number) { return Framework.implementation.pow(tensor, exponent);}
  mul(tensorA: Tensor, tensorB: Tensor | number) { return Framework.implementation.mul(tensorA, tensorB);}
  gt(tensor: Tensor, value: number) { return Framework.implementation.gt(tensor, value);}
  transpose(tensor: Tensor) { return Framework.implementation.transpose(tensor);}
  maximum(tensor: Tensor, value: number) { return Framework.implementation.maximum(tensor, value);}
  relu(x: Tensor) { return Framework.implementation.relu(x);}
  max(tensor: Tensor) { return Framework.implementation.max(tensor);}
  mean(tensor: Tensor) { return Framework.implementation.mean(tensor);}
  sigmoid(tensor: Tensor) { return Framework.implementation.sigmoid(tensor);}
  clone(tensor: Tensor) { return Framework.implementation.clone(tensor);}
  copy(tensorSource: Tensor, tensorDestination: Tensor) { return Framework.implementation.copy(tensorSource, tensorDestination);}
  item(tensor: Tensor) { return Framework.implementation.item(tensor);}
  reset() { return Framework.implementation.reset();}
  compute() { return Framework.implementation.compute();}
  destroy() { return Framework.implementation.destroy();}
  print(...data: unknown[]) { return Framework.implementation.print(...data);}
}

const framework = new Framework();

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
  if (Framework.implementation) return framework;

  if (device === Device.GPU) {
    Framework.implementation = WebGPUTensors.create();
    framework.init();
  } else if (device === Device.CPU) {
    Framework.implementation = JSTensors.create();
    framework.init();
  } else {
    throw new Error('Unknown device ' + device);
  }
  return framework;
}

export default device();

export { DType };
export type { Tensors as Tensors, Tensor };
