import { Tensor, Tensors, Shape, Size, DType, Device, TensorOptions, NestedArray, 
  Float32NestedArray } from '../../types';
  import ShapeSize from '../../size';
  import * as wasmModule from '../../../../wasm/webgpu-tensors';
  
  export class WASMTensor implements Tensor {
    _wasmTensor: wasmModule.WASMTensorsImpl;
    shape: Size;
    dtype: DType;
    device: Device;
    readable: boolean;
  
    constructor(wasmTensor: wasmModule.WASMTensorsImpl, shape: Shape, dtype: DType = DType.Float32) {
      this._wasmTensor = wasmTensor;
      this.shape = new ShapeSize(shape);
      this.dtype = dtype;
      this.device = Device.WASM;
      this.readable = true;
    }
  
    async readArray<T>(_options?: { mode?: 1 | undefined }): Promise<NestedArray<T>> {
      const result = await WASMTensors.getInstance().read_array(this._wasmTensor);
      return result as NestedArray<T>;
    }
  
    async readFloat32(_options?: { mode?: 1 | undefined }): Promise<Float32NestedArray> {
      const result = await WASMTensors.getInstance().read_float32(this._wasmTensor);
      return result as Float32NestedArray;
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
      return new WASMTensor(wasmTensor, shape, options?.dtype);
    }

    ones(shape: Shape, options?: Partial<TensorOptions>): Tensor {
      const wasmTensor = WASMTensors._instance.ones(new Uint32Array(shape), options);
      return new WASMTensor(wasmTensor, shape, options?.dtype);
    }

    rand(shape: Shape, options?: Partial<TensorOptions>): Tensor {
      const wasmTensor = WASMTensors._instance.rand(new Uint32Array(shape), options);
      return new WASMTensor(wasmTensor, shape, options?.dtype);
    }

    zeros(shape: Shape, options?: Partial<TensorOptions>): Tensor {
      const wasmTensor = WASMTensors._instance.zeros(new Uint32Array(shape), options);
      return new WASMTensor(wasmTensor, shape, options?.dtype);
    }

    tensor(array: NestedArray<number>, options?: Partial<TensorOptions>): Tensor {
      const wasmTensor = WASMTensors._instance.tensor(array, options);
      return new WASMTensor(wasmTensor, wasmTensor.shape, options?.dtype);
    }

    sub(tensorA: Tensor, tensorB: Tensor): Tensor {
      const result = WASMTensors._instance.sub(
        (tensorA as WASMTensor)._wasmTensor,
        (tensorB as WASMTensor)._wasmTensor
      );
      return new WASMTensor(result, result.shape, result.dtype);
    }

    pow(tensor: Tensor, exponent: number): Tensor {
      const result = WASMTensors._instance.pow(
        (tensor as WASMTensor)._wasmTensor,
        exponent
      );
      return new WASMTensor(result, result.shape, result.dtype);
    }

    gt(tensor: Tensor, value: number): Tensor {
      const result = WASMTensors._instance.gt(
        (tensor as WASMTensor)._wasmTensor,
        value
      );
      return new WASMTensor(result, result.shape, result.dtype);
    }

    transpose(tensor: Tensor): Tensor {
      const result = WASMTensors._instance.transpose(
        (tensor as WASMTensor)._wasmTensor
      );
      return new WASMTensor(result, result.shape, result.dtype);
    }

    maximum(tensor: Tensor, value: number): Tensor {
      const result = WASMTensors._instance.maximum(
        (tensor as WASMTensor)._wasmTensor,
        value
      );
      return new WASMTensor(result, result.shape, result.dtype);
    }

    relu(x: Tensor): Tensor {
      const result = WASMTensors._instance.relu(
        (x as WASMTensor)._wasmTensor
      );
      return new WASMTensor(result, result.shape, result.dtype);
    }

    max(tensor: Tensor): Tensor {
      const result = WASMTensors._instance.max(
        (tensor as WASMTensor)._wasmTensor
      );
      return new WASMTensor(result, result.shape, result.dtype);
    }

    mean(tensor: Tensor): Tensor {
      const result = WASMTensors._instance.mean(
        (tensor as WASMTensor)._wasmTensor
      );
      return new WASMTensor(result, result.shape, result.dtype);
    }

    sigmoid(tensor: Tensor): Tensor {
      const result = WASMTensors._instance.sigmoid(
        (tensor as WASMTensor)._wasmTensor
      );
      return new WASMTensor(result, result.shape, result.dtype);
    }

    clone(tensorSource: Tensor): Tensor {
      const result = WASMTensors._instance.clone(
        (tensorSource as WASMTensor)._wasmTensor
      );
      return new WASMTensor(result, result.shape, result.dtype);
    }

    copy(tensorSource: Tensor, tensorDestination: Tensor): void {
      WASMTensors._instance.copy(
        (tensorSource as WASMTensor)._wasmTensor,
        (tensorDestination as WASMTensor)._wasmTensor
      );
    }
    
    async item(tensor: Tensor): Promise<number> {
      return WASMTensors._instance.item((tensor as WASMTensor)._wasmTensor);
    }

    randn(shape: Shape, options?: Partial<TensorOptions>): Tensor {
      const wasmTensor = WASMTensors._instance.randn(new Uint32Array(shape), options);
      return new WASMTensor(wasmTensor, shape, options?.dtype);
    }

    matmul(tensorA: Tensor, tensorB: Tensor): Tensor {
      const result = WASMTensors._instance.matmul(
        (tensorA as WASMTensor)._wasmTensor,
        (tensorB as WASMTensor)._wasmTensor
      );
      return new WASMTensor(result, result.shape, result.dtype);
    }

    mul(tensorA: Tensor, tensorB: Tensor | number): Tensor {
      if (typeof tensorB === 'number') {
        const result = WASMTensors._instance.mul_scalar(
          (tensorA as WASMTensor)._wasmTensor,
          tensorB
        );
        return new WASMTensor(result, result.shape, result.dtype);
      } else {
        const result = WASMTensors._instance.mul(
          (tensorA as WASMTensor)._wasmTensor,
          (tensorB as WASMTensor)._wasmTensor
        );
        return new WASMTensor(result, result.shape, result.dtype);
      }
    }

    async print(...data: unknown[]): Promise<void> {
      console.log(...data);
    }
  }
