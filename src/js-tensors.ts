import { Tensors, Tensor, DType, Device, Shape, TensorOptions, NestedArray } from './webgpu-tensors';

class Size {
  data: number[];
  constructor(data: number[]) {
    this.data = data;
  }

  get length() {
    return this.data.length;
  }

  get size() {
    return this.data.reduce((a, b) => a * b, 1);
  }

  getDim(dim: number) {
    return this.data[dim];
  }

  toString() {
    return `Size[${this.data.join(',')}]`;
  }
}

class JSTensor implements Tensor {
  data: number[];
  shape: Size;
  dtype: DType;
  device: Device;
  readable: boolean;

  constructor(data: number[], shape: Shape, dtype: DType = DType.Float32) {
    this.data = data;
    this.shape = new Size(shape);
    this.dtype = dtype;
    this.device = Device.CPU;
    this.readable = true;
  }

  async readArray<T>(options?: { mode?: 1 | undefined } | undefined): Promise<NestedArray<T>> {
    return this.data as unknown as NestedArray<T>;
  }

  async readFloat32(options?: { mode?: 1 | undefined } | undefined): Promise<Float32Array> {
    return new Float32Array(this.data);
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

export class JSTensors implements Tensors {
  async init(device: Device = Device.CPU): Promise<void> {
    // No initialization needed for JS implementation
  }

  empty(shape: Shape, options?: Partial<TensorOptions> | undefined): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    return new JSTensor(new Array(size), shape);
  }

  ones(shape: Shape, options?: Partial<TensorOptions> | undefined): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    return new JSTensor(new Array(size).fill(1), shape);
  }

  rand(shape: Shape, options?: Partial<TensorOptions> | undefined): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    return new JSTensor(Array.from({ length: size }, () => Math.random()), shape);
  }

  randn(shape: Shape, options?: Partial<TensorOptions> | undefined): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    return new JSTensor(Array.from({ length: size }, () => {
      let u = 0, v = 0;
      while (u === 0) u = Math.random();
      while (v === 0) v = Math.random();
      return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }), shape);
  }

  zeros(shape: Shape, options?: Partial<TensorOptions> | undefined): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    return new JSTensor(new Array(size).fill(0), shape);
  }

  tensor(array: NestedArray<number>, options?: Partial<TensorOptions> | undefined): Tensor {
    const flatArray = this.flattenArray(array);
    const shape = this.getShape(array);
    return new JSTensor(flatArray, shape);
  }

  private flattenArray(array: NestedArray<number>): number[] {
    return array.flat(Infinity) as number[];
  }

  private getShape(array: NestedArray<number>): Shape {
    const shape: Shape = [];
    let current: any = array;
    while (Array.isArray(current)) {
      shape.push(current.length);
      current = current[0];
    }
    return shape;
  }

  matmul(tensorA: Tensor, tensorB: Tensor): Tensor {
    const a = tensorA as JSTensor;
    const b = tensorB as JSTensor;
    const [m, n] = a.shape.data;
    const [_, p] = b.shape.data;
    const result = new Array(m * p).fill(0);

    for (let i = 0; i < m; i++) {
      for (let j = 0; j < p; j++) {
        for (let k = 0; k < n; k++) {
          result[i * p + j] += a.data[i * n + k] * b.data[k * p + j];
        }
      }
    }

    return new JSTensor(result, [m, p]);
  }

  sub(tensorA: Tensor, tensorB: Tensor): Tensor {
    const a = tensorA as JSTensor;
    const b = tensorB as JSTensor;
    const result = a.data.map((val, i) => val - b.data[i]);
    return new JSTensor(result, a.shape.data);
  }

  pow(tensor: Tensor, exponent: number): Tensor {
    const t = tensor as JSTensor;
    const result = t.data.map(val => Math.pow(val, exponent));
    return new JSTensor(result, t.shape.data);
  }

  mul(tensorA: Tensor, tensorB: Tensor | number): Tensor {
    const a = tensorA as JSTensor;
    let result: number[];
    if (typeof tensorB === 'number') {
      result = a.data.map(val => val * tensorB);
    } else {
      const b = tensorB as JSTensor;
      result = a.data.map((val, i) => val * b.data[i]);
    }
    return new JSTensor(result, a.shape.data);
  }

  gt(tensor: Tensor, value: number): Tensor {
    const t = tensor as JSTensor;
    const result = t.data.map(val => val > value ? 1 : 0);
    return new JSTensor(result, t.shape.data);
  }

  transpose(tensor: Tensor): Tensor {
    const t = tensor as JSTensor;
    const [rows, cols] = t.shape.data;
    const result = new Array(rows * cols);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        result[j * rows + i] = t.data[i * cols + j];
      }
    }
    return new JSTensor(result, [cols, rows]);
  }

  maximum(tensor: Tensor, value: number): Tensor {
    const t = tensor as JSTensor;
    const result = t.data.map(val => Math.max(val, value));
    return new JSTensor(result, t.shape.data);
  }

  relu(x: Tensor): Tensor {
    return this.maximum(x, 0);
  }

  max(tensor: Tensor): Tensor {
    const t = tensor as JSTensor;
    const maxValue = Math.max(...t.data);
    return new JSTensor([maxValue], [1]);
  }

  mean(tensor: Tensor): Tensor {
    const t = tensor as JSTensor;
    const sum = t.data.reduce((a, b) => a + b, 0);
    const mean = sum / t.data.length;
    return new JSTensor([mean], [1]);
  }

  sigmoid(tensor: Tensor): Tensor {
    const t = tensor as JSTensor;
    const result = t.data.map(val => 1 / (1 + Math.exp(-val)));
    return new JSTensor(result, t.shape.data);
  }

  clone(tensorSource: Tensor): Tensor {
    const source = tensorSource as JSTensor;
    return new JSTensor([...source.data], source.shape.data);
  }

  copy(tensorSource: Tensor, tensorDestination: Tensor): void {
    const source = tensorSource as JSTensor;
    const destination = tensorDestination as JSTensor;
    destination.data = [...source.data];
  }

  async item(tensor: Tensor): Promise<number> {
    const t = tensor as JSTensor;
    if (t.numel() !== 1) {
      throw new Error('item() can only be called on tensors with a single element');
    }
    return t.data[0];
  }

  reset(): void {
    // No-op for JS implementation
  }

  compute(): void {
    // No-op for JS implementation
  }

  destroy(): void {
    // No-op for JS implementation
  }

  async print(...data: unknown[]): Promise<void> {
    console.log(...data);
  }
}

export default function createJSTensors(): Tensors {
  return new JSTensors();
}
