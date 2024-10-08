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
import { stridedToNestedArray, stridedToNestedFloat32Array } from '../../utils';
import importWASMModule, { WASMTensorsImpl } from './module';

export class WASMTensor implements Tensor {
  shape: ShapeSize;
  dtype: DType;
  device: Device;
  readable: boolean;
  data: unknown;

  constructor(tensor: Tensor) {
    this.shape = tensor.shape;
    this.dtype = tensor.dtype;
    this.device = tensor.device;
    this.readable = tensor.readable;
    this.data = (tensor as WASMTensor).data;
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
  private static wasm: WASMTensorsImpl;

  private constructor() {
    this.device = Device.WASM;
  }

  public static getInstance(): WASMTensorsImpl {
    if (!WASMTensors.wasm) {
      throw new Error("WASM not instancied");
    }
    return WASMTensors.wasm;
  }

  static create(): Tensors {
    return new WASMTensors();
  }

  async init(device: Device = Device.WASM): Promise<void> {
    if (device !== Device.WASM) {
      throw new Error('WASMTensors only supports WASM device');
    }
    if (!WASMTensors.wasm) {
      const wasmModule = await importWASMModule();
      const impl = new wasmModule.WASMTensorsImpl();
      WASMTensors.wasm = impl;
    }
    WASMTensors.wasm.init();
  }

  reset(): void {
    WASMTensors.wasm.reset();
  }

  compute(): void {
    WASMTensors.wasm.compute();
  }

  destroy(): void {
    WASMTensors.wasm.destroy();
  }

  empty(shape: Shape, options?: Partial<TensorOptions>): Tensor {
    const wasmTensor = WASMTensors.wasm.empty(new Uint32Array(shape), options);
    return new WASMTensor(wasmTensor);
  }

  ones(shape: Shape, options?: Partial<TensorOptions>): Tensor {
    const wasmTensor = WASMTensors.wasm.ones(new Uint32Array(shape), options);
    return new WASMTensor(wasmTensor);
  }

  rand(shape: Shape, options?: Partial<TensorOptions>): Tensor {
    const wasmTensor = WASMTensors.wasm.rand(new Uint32Array(shape), options);
    return new WASMTensor(wasmTensor);
  }

  zeros(shape: Shape, options?: Partial<TensorOptions>): Tensor {
    const wasmTensor = WASMTensors.wasm.zeros(new Uint32Array(shape), options);
    return new WASMTensor(wasmTensor);
  }

  tensor(array: NestedArray<number>, options?: Partial<TensorOptions>): Tensor {
    const shape = ShapeSize.createFromNestedArray(array);
    const oned = shape.length === 1;
    const stridedArray = (oned ? array : array.flat(shape.length as 10)) as number[];
    const wasmTensor = WASMTensors.wasm.tensor(stridedArray, {
      ...options,
      shape: oned ? undefined : shape.data,
    });
    if (!oned) {
      wasmTensor.shape = shape;
    }
    return new WASMTensor(wasmTensor);
  }

  sub(tensorA: Tensor, tensorB: Tensor): Tensor {
    const result = WASMTensors.wasm.sub(tensorA, tensorB);
    return new WASMTensor(result);
  }

  pow(tensor: Tensor, exponent: number): Tensor {
    const result = WASMTensors.wasm.pow(tensor, exponent);
    return new WASMTensor(result);
  }

  gt(tensor: Tensor, value: number): Tensor {
    const result = WASMTensors.wasm.gt(tensor, value);
    return new WASMTensor(result);
  }

  transpose(tensor: Tensor): Tensor {
    const result = WASMTensors.wasm.transpose(tensor);
    return new WASMTensor(result);
  }

  maximum(tensor: Tensor, value: number): Tensor {
    const result = WASMTensors.wasm.maximum(tensor, value);
    return new WASMTensor(result);
  }

  relu(x: Tensor): Tensor {
    const result = WASMTensors.wasm.relu(x);
    return new WASMTensor(result);
  }

  max(tensor: Tensor): Tensor {
    const result = WASMTensors.wasm.max(tensor);
    return new WASMTensor(result);
  }

  mean(tensor: Tensor): Tensor {
    const result = WASMTensors.wasm.mean(tensor);
    return new WASMTensor(result);
  }

  sigmoid(tensor: Tensor): Tensor {
    const result = WASMTensors.wasm.sigmoid(tensor);
    return new WASMTensor(result);
  }

  clone(tensor: Tensor): Tensor {
    const result = WASMTensors.wasm.clone(tensor);
    return new WASMTensor(result);
  }

  copy(tensorSource: Tensor, tensorDestination: Tensor): void {
    WASMTensors.wasm.copy(tensorSource, tensorDestination);
  }

  async item(tensor: Tensor): Promise<number> {
    return WASMTensors.wasm.item(tensor);
  }

  randn(shape: Shape, options?: Partial<TensorOptions>): Tensor {
    const result = WASMTensors.wasm.randn(new Uint32Array(shape), options);
    return new WASMTensor(result);
  }

  matmul(tensorA: Tensor, tensorB: Tensor): Tensor {
    const result = WASMTensors.wasm.matmul(tensorA, tensorB);
    return new WASMTensor(result);
  }

  mul(tensorA: Tensor, tensorB: Tensor | number): Tensor {
    if (typeof tensorB === 'number') {
      const result = WASMTensors.wasm.mul_scalar(tensorA, tensorB);
      return new WASMTensor(result);
    } else {
      const result = WASMTensors.wasm.mul(tensorA, tensorB);
      return new WASMTensor(result);
    }
  }

  async print(...data: unknown[]): Promise<void> {
    console.log(...data);
  }
}
