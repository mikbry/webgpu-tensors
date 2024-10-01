import { Device, DType, Tensor, Tensors } from './types';
import { JSTensors } from './tensors/js';
import WebGPUTensors from './tensors/webgpu';

let tensors: Tensors | undefined;

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
  if (tensors) return tensors;

  if (device === Device.GPU) {
    tensors = WebGPUTensors.create();
    (tensors as WebGPUTensors).init();
  } else if (device === Device.CPU) {
    tensors = JSTensors.create();
    tensors.init();
  } else {
    throw new Error('Unknown device ' + device);
  }
  return tensors;
}

export default device();

export { DType };
export type { Tensors, Tensor };
