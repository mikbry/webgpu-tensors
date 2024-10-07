import {
  Tensor,
  Tensors,
  Shape,
  Size,
  DType,
  Device,
  TensorOptions,
  NestedArray,
  Float32NestedArray,
} from '../../types';
import ShapeSize from '../../size';
import * as wasmModule from '../../../../wasm/webgpu-tensors';
import { stridedToNestedArray, stridedToNestedFloat32Array } from '../../utils';

export class WASMTensor implements Tensor {
  shape: ShapeSize;
  dtype: DType;
  device: Device;
  readable: boolean;
  data: unknown;

  constructor(wasmTensor: WASMTensor) {
    this.shape = wasmTensor.shape;
    this.dtype = wasmTensor.dtype;
    this.device = wasmTensor.device;
    this.readable = wasmTensor.readable;
    this.data = wasmTensor.data;
  }

  async readArray<T>(_options?: { mode?: 1 | undefined }): Promise<NestedArray<T>> {
    const buffer = await WASMTensors.getInstance().read_array(this);
    return stridedToNestedArray<T>(buffer as T[], this.shape.data);
  }

  async readFloat32(_options?: { mode?: 1 | undefined }): Promise<Float32NestedArray> {
    const data = await WASMTensors.getInstance().read_float32(this);
    return stridedToNestedFloat32Array(data, this.shape.data);
  }

  size(dim?: number): Size | number {
    if (dim === undefined) {
      return this.shape;
    }
    return this.shape.getDim(dim);
  }

  numel(): number {
    return this.shape.size;
  }
}

export class WASMTensors implements Tensors {
  device: Device;
  private static _instance: wasmModule.WASMTensorsImpl;

  private constructor() {
    this.device = Device.WASM;
  }

  public static getInstance(): wasmModule.WASMTensorsImpl {
    if (!WASMTensors._instance) {
      WASMTensors._instance = new wasmModule.WASMTensorsImpl();
    }
    return WASMTensors._instance;
  }

  static create(): Tensors {
    if (!WASMTensors._instance) {
      WASMTensors._instance = new wasmModule.WASMTensorsImpl();
    }
    return new WASMTensors();
  }

  async init(device: Device = Device.WASM): Promise<void> {
    if (device !== Device.WASM) {
      throw new Error('WASMTensors only supports WASM device');
    }
    WASMTensors._instance.init();
  }

  reset(): void {
    WASMTensors._instance.reset();
  }

  compute(): void {
    WASMTensors._instance.compute();
  }

  destroy(): void {
    WASMTensors._instance.destroy();
  }

  empty(shape: Shape, options?: Partial<TensorOptions>): Tensor {
    const wasmTensor = WASMTensors._instance.empty(new Uint32Array(shape), options);
    return new WASMTensor(wasmTensor);
  }

  ones(shape: Shape, options?: Partial<TensorOptions>): Tensor {
    const wasmTensor = WASMTensors._instance.ones(new Uint32Array(shape), options);
    return new WASMTensor(wasmTensor);
  }

  rand(shape: Shape, options?: Partial<TensorOptions>): Tensor {
    const wasmTensor = WASMTensors._instance.rand(new Uint32Array(shape), options);
    return new WASMTensor(wasmTensor);
  }

  zeros(shape: Shape, options?: Partial<TensorOptions>): Tensor {
    const wasmTensor = WASMTensors._instance.zeros(new Uint32Array(shape), options);
    return new WASMTensor(wasmTensor);
  }

  tensor(array: NestedArray<number>, options?: Partial<TensorOptions>): Tensor {
    const shape = ShapeSize.createFromNestedArray(array);
    const oned = shape.length === 1;
    const stridedArray = (oned ? array : array.flat(shape.length as 10)) as number[];
    const wasmTensor = WASMTensors._instance.tensor(stridedArray, {
      ...options,
      shape: oned ? undefined : shape.data,
    });
    if (!oned) {
      wasmTensor.shape = shape;
    }
    return new WASMTensor(wasmTensor);
  }

  sub(tensorA: Tensor, tensorB: Tensor): Tensor {
    const result = WASMTensors._instance.sub(tensorA, tensorB);
    return new WASMTensor(result);
  }

  pow(tensor: Tensor, exponent: number): Tensor {
    const result = WASMTensors._instance.pow(tensor, exponent);
    return new WASMTensor(result);
  }

  gt(tensor: Tensor, value: number): Tensor {
    const result = WASMTensors._instance.gt(tensor, value);
    return new WASMTensor(result);
  }

  transpose(tensor: Tensor): Tensor {
    const result = WASMTensors._instance.transpose(tensor);
    return new WASMTensor(result);
  }

  maximum(tensor: Tensor, value: number): Tensor {
    const result = WASMTensors._instance.maximum(tensor, value);
    return new WASMTensor(result);
  }

  relu(x: Tensor): Tensor {
    const result = WASMTensors._instance.relu(x);
    return new WASMTensor(result);
  }

  max(tensor: Tensor): Tensor {
    const result = WASMTensors._instance.max(tensor);
    return new WASMTensor(result);
  }

  mean(tensor: Tensor): Tensor {
    const result = WASMTensors._instance.mean(tensor);
    return new WASMTensor(result);
  }

  sigmoid(tensor: Tensor): Tensor {
    const result = WASMTensors._instance.sigmoid(tensor);
    return new WASMTensor(result);
  }

  clone(tensor: Tensor): Tensor {
    const result = WASMTensors._instance.clone(tensor);
    return new WASMTensor(result);
  }

  copy(tensorSource: Tensor, tensorDestination: Tensor): void {
    WASMTensors._instance.copy(tensorSource, tensorDestination);
  }

  async item(tensor: Tensor): Promise<number> {
    return WASMTensors._instance.item(tensor);
  }

  randn(shape: Shape, options?: Partial<TensorOptions>): Tensor {
    const result = WASMTensors._instance.randn(new Uint32Array(shape), options);
    return new WASMTensor(result);
  }

  matmul(tensorA: Tensor, tensorB: Tensor): Tensor {
    const result = WASMTensors._instance.matmul(tensorA, tensorB);
    return new WASMTensor(result);
  }

  mul(tensorA: Tensor, tensorB: Tensor | number): Tensor {
    if (typeof tensorB === 'number') {
      const result = WASMTensors._instance.mul_scalar(tensorA, tensorB);
      return new WASMTensor(result);
    } else {
      const result = WASMTensors._instance.mul(tensorA, tensorB);
      return new WASMTensor(result);
    }
  }

  async print(...data: unknown[]): Promise<void> {
    console.log(...data);
  }
}
