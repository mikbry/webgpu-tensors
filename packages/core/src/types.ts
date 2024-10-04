export enum DType {
  Float32 = 'Float32',
}

export enum Device {
  GPU = 'GPU',
  CPU = 'CPU',
  WASM = 'WASM',
}

export type NestedArray<T> = Array<NestedArray<T> | T>;

export type Float32NestedArray = Array<Float32NestedArray> | Float32Array;

export type Shape = number[];

export interface Tensors {
  device: Device;
  init(device?: Device): Promise<void>;
  empty(shape: Shape, options?: Partial<TensorOptions> | undefined): Tensor;
  ones(shape: Shape, options?: Partial<TensorOptions> | undefined): Tensor;
  rand(shape: Shape, options?: Partial<TensorOptions> | undefined): Tensor;
  randn(shape: Shape, options?: Partial<TensorOptions> | undefined): Tensor;
  zeros(shape: Shape, options?: Partial<TensorOptions> | undefined): Tensor;
  tensor(array: NestedArray<number>, options?: Partial<TensorOptions> | undefined): Tensor;

  matmul(tensorA: Tensor, tensorB: Tensor): Tensor;
  sub(tensorA: Tensor, tensorB: Tensor): Tensor;
  pow(tensor: Tensor, exponent: number): Tensor;
  mul(tensorA: Tensor, tensorB: Tensor | number): Tensor;
  gt(tensor: Tensor, value: number): Tensor;
  transpose(tensor: Tensor): Tensor;

  maximum(tensor: Tensor, value: number): Tensor;
  relu(x: Tensor): Tensor;

  max(tensor: Tensor): Tensor;

  mean(tensor: Tensor): Tensor;

  sigmoid(tensor: Tensor): Tensor;

  clone(tensorSource: Tensor): Tensor;
  copy(tensorSource: Tensor, tensorDestination: Tensor): void;

  item(tensor: Tensor): Promise<number>;

  reset(): void;
  compute(): void;
  destroy(): void;

  print(...data: unknown[]): Promise<void>;
}

export type TensorOptions = {
  usage: number;
  mappedAtCreation: boolean | undefined;
  readable: boolean;
  dtype?: DType;
  shape?: number[];
};

export interface Size {
  data: number[];
  length: number;
  size: number;

  getDim(dim: number): number;

  toString(): string;
}

export interface Tensor {
  shape: Size;
  dtype: DType;
  device: Device;
  readable: boolean;

  readArray<T>(options?: { mode?: 1 | undefined } | undefined): Promise<NestedArray<T>>;
  readFloat32(options?: { mode?: 1 | undefined } | undefined): Promise<Float32NestedArray>;
  size(dim?: number): Size | number;
  numel(): number;
}
