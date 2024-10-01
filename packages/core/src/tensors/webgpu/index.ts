import ShapeSize, { getShapeSize } from '../../size';
import {
  Device,
  DType,
  Float32NestedArray,
  NestedArray,
  Shape,
  Size,
  Tensor,
  TensorOptions,
  Tensors,
} from '../../types';

const F32SIZE = 4;

function stridedToNestedFloat32Array(
  buffer: ArrayBuffer,
  dim: number[],
  offset = 0,
  depth = 0,
): Float32NestedArray {
  if (dim.length === 1) {
    const length = getShapeSize(dim) * F32SIZE;
    return new Float32Array(buffer.slice(offset, offset + length));
  }
  const array: Float32NestedArray = [];
  for (let n = 0; n < dim[0]; n++) {
    const nestedShape = [...dim];
    nestedShape.shift();
    const length = getShapeSize(nestedShape) * F32SIZE;
    const i = length * n;
    const nestedArray = stridedToNestedFloat32Array(buffer, nestedShape, offset + i, depth + 1);
    array.push(nestedArray);
  }
  return array;
}

function stridedToNestedArray<T>(
  buffer: ArrayBuffer,
  dim: number[],
  offset = 0,
  depth = 0,
): NestedArray<T> {
  if (dim.length === 1) {
    const length = getShapeSize(dim) * F32SIZE;
    return Array.from(new Float32Array(buffer.slice(offset, offset + length))) as NestedArray<T>;
  }
  const array: NestedArray<T> = [];
  for (let n = 0; n < dim[0]; n++) {
    const nestedShape = [...dim];
    nestedShape.shift();
    const length = getShapeSize(nestedShape) * F32SIZE;
    const i = length * n;
    const nestedArray = stridedToNestedArray(buffer, nestedShape, offset + i, depth + 1);
    array.push(Array.from(nestedArray) as NestedArray<T>);
  }
  return array;
}

function float32ArrayToString(array: Float32NestedArray): string {
  let output = '';
  array.forEach((value) => {
    if (output.length > 0) {
      output += ',';
    }
    if (value instanceof Float32Array || Array.isArray(value)) {
      output += float32ArrayToString(value);
    } else {
      output += value.toString();
    }
  });
  return `[${output}]`;
}

function getGPUBuffer(tensor: Tensor): GPUBuffer {
  return (tensor as GPUTensor).buffer;
}

export interface WebGPUInstance {
  device: GPUDevice;
}

export default class WebGPUTensors implements Tensors {
  static _instance?: WebGPUInstance;

  commands: Array<GPUCommandBuffer>;
  hasComputeOnce: boolean;

  constructor() {
    this.commands = [];
    this.hasComputeOnce = false;
  }

  static isAvailable(): boolean {
    return 'gpu' in navigator;
  }

  static create() {
    if (!WebGPUTensors.isAvailable()) {
      throw new Error('WebGPU not supported on this browser.');
    }
    return new WebGPUTensors() as Tensors;
  }

  reset() {
    this.commands = [];
  }

  async init(device: Device = Device.GPU) {
    if (device !== Device.GPU) {
      throw new Error('Unknown device ' + device);
    }
    if (!WebGPUTensors.isAvailable()) {
      throw new Error('WebGPU not supported on this browser.');
    }

    if (!WebGPUTensors._instance) {
      const adapter = await navigator.gpu.requestAdapter();
      if (!adapter) {
        throw new Error('No appropriate GPUAdapter found.');
      }

      const device = await adapter.requestDevice();

      WebGPUTensors._instance = { device };
    }
    // return WebGPUTensors._instance;
  }

  get instance() {
    /* return (async () => {
                return this.init();
            })(); */
    if (!WebGPUTensors._instance) {
      throw new Error('Tensors no instance initialized.');
    }
    return WebGPUTensors._instance;
  }

  destroy() {
    WebGPUTensors._instance?.device.destroy();
  }

  createTensor(shape: Size, options: TensorOptions) {
    const { usage, mappedAtCreation, readable } = options;
    const { device } = this.instance;
    const size = shape.size * F32SIZE;
    return new GPUTensor(
      device.createBuffer({
        mappedAtCreation,
        size,
        usage,
      }),
      shape,
      readable,
    );
  }

  resultTensor(shape: Shape, options?: Partial<TensorOptions> | undefined) {
    const { usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, mappedAtCreation = false } =
      options || {};
    return this.createTensor(new ShapeSize(shape), {
      usage,
      mappedAtCreation,
      readable: false,
    });
  }

  buildTensor(
    buildArray: (i: number) => number[],
    shape: Shape,
    options?: Partial<TensorOptions> | undefined,
  ) {
    const { usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, mappedAtCreation = true } =
      options || {};
    const tensor = this.createTensor(new ShapeSize(shape), {
      usage,
      mappedAtCreation,
      readable: false,
    });
    return tensor.set(buildArray(tensor.shape.size));
  }

  empty(shape: Shape, options?: Partial<TensorOptions> | undefined) {
    const { usage = GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, mappedAtCreation = false } =
      options || {};
    return this.createTensor(new ShapeSize(shape), {
      usage,
      mappedAtCreation,
      readable: true,
    });
  }

  tensor(array: NestedArray<number>, options?: Partial<TensorOptions> | undefined) {
    const shape = ShapeSize.createFromNestedArray(array);
    const stridedArray = (shape.length === 1 ? array : array.flat(shape.length as 10)) as number[];
    return this.buildTensor(() => stridedArray, shape.data, options);
  }

  ones(shape: Shape, options?: Partial<TensorOptions> | undefined) {
    return this.buildTensor((size) => new Array(size).fill(1), shape, options);
  }

  rand(shape: Shape, options?: Partial<TensorOptions> | undefined) {
    return this.buildTensor(
      (size) => Array.from(new Array(size), () => Math.random()),
      shape,
      options,
    );
  }

  randn(shape: Shape, options?: Partial<TensorOptions> | undefined) {
    // Box-Muller transform
    const boxMullerTransform = () => {
      const u1 = Math.random();
      const u2 = Math.random();
      const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
      return z0;
    };
    return this.buildTensor(
      (size) => Array.from(new Array(size), () => boxMullerTransform()),
      shape,
      options,
    );
  }

  zeros(shape: Shape, options?: Partial<TensorOptions> | undefined) {
    return this.buildTensor((size) => new Array(size).fill(0), shape, options);
  }

  compute() {
    if (this.hasComputeOnce && this.commands.length === 0) {
      return;
    }
    const instance = this.instance;
    instance?.device.queue.submit(this.commands);
    this.hasComputeOnce = true;
    this.reset();
  }

  createKernel(resultShape: Shape, entries: GPUBuffer[], code: string, workGroupCounts: number[]) {
    const result = this.resultTensor(resultShape);
    const { device } = this.instance;
    const computePipeline = device.createComputePipeline({
      layout: 'auto',
      compute: {
        module: device.createShaderModule({
          code,
        }),
        entryPoint: 'main',
      },
    });

    const bindGroupLayout = computePipeline.getBindGroupLayout(0);
    const bindGroup = device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        ...entries.map((buffer, index) => ({
          binding: index,
          resource: { buffer },
        })),
        { binding: entries.length, resource: { buffer: result.buffer } },
      ],
    });

    const commandEncoder = device.createCommandEncoder();
    const passEncoder = commandEncoder.beginComputePass();
    passEncoder.setPipeline(computePipeline);
    passEncoder.setBindGroup(0, bindGroup);
    passEncoder.dispatchWorkgroups(workGroupCounts[0], workGroupCounts[1], workGroupCounts[3]);
    passEncoder.end();
    this.commands.push(commandEncoder.finish());

    return result;
  }

  matmul(input: Tensor, other: Tensor) {
    const inputShape = input.shape.data;
    const otherShape = other.shape.data;

    if (inputShape[1] !== otherShape[0]) {
      const error = `Incompatible matrix dimensions for multiplication ${inputShape[1]} ${otherShape[0]}`;
      throw new Error(error);
    }

    return this.createKernel(
      [inputShape[0], otherShape[1]],
      [getGPUBuffer(input), getGPUBuffer(other)],
      `
                @group(0) @binding(0) var<storage, read> a: array<f32>;
                @group(0) @binding(1) var<storage, read> b: array<f32>;
                @group(0) @binding(2) var<storage, read_write> c: array<f32>;
      
                @compute @workgroup_size(8, 8)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                  let row = global_id.x;
                  let col = global_id.y;
                  let m = ${inputShape[0]}u;
                  let n = ${inputShape[1]}u;
                  let p = ${otherShape[1]}u;
      
                  if (row < m && col < p) {
                    var sum = 0.0;
                    for (var i = 0u; i < n; i = i + 1u) {
                      sum = sum + a[row * n + i] * b[i * p + col];
                    }
                    c[row * p + col] = sum;
                  }
                }
              `,
      [Math.ceil(inputShape[0] / 8), Math.ceil(otherShape[1] / 8)],
    );
  }

  maximum(tensor: Tensor, value: number) {
    return this.createKernel(
      tensor.shape.data,
      [getGPUBuffer(tensor)],
      `
                    @group(0) @binding(0) var<storage, read> input: array<f32>;
                    @group(0) @binding(1) var<storage, read_write> output: array<f32>;

                    @compute @workgroup_size(64)
                    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                        let index = global_id.x;
                        if (index < arrayLength(&input)) {
                            output[index] = max(input[index], ${value}f);
                        }
                    }
            `,
      [Math.ceil(tensor.shape.size / 64)],
    );
  }

  relu(x: Tensor) {
    return this.maximum(x, 0);
  }

  sub(tensorA: Tensor, tensorB: Tensor) {
    if (!tensorA.shape.data.every((dim, i) => dim === tensorB.shape.data[i])) {
      throw new Error('Tensor shapes must match for subtraction');
    }

    return this.createKernel(
      tensorA.shape.data,
      [getGPUBuffer(tensorA), getGPUBuffer(tensorB)],
      `
                @group(0) @binding(0) var<storage, read> a: array<f32>;
                @group(0) @binding(1) var<storage, read> b: array<f32>;
                @group(0) @binding(2) var<storage, read_write> output: array<f32>;

                @compute @workgroup_size(256)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let index = global_id.x;
                    if (index < arrayLength(&a)) {
                        output[index] = a[index] - b[index];
                    }
                }
            `,
      [Math.ceil(tensorA.shape.size / 256)],
    );
  }

  pow(tensor: Tensor, exponent: number) {
    return this.createKernel(
      tensor.shape.data,
      [getGPUBuffer(tensor)],
      `
                @group(0) @binding(0) var<storage, read> input: array<f32>;
                @group(0) @binding(1) var<storage, read_write> output: array<f32>;

                @compute @workgroup_size(256)
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                    let index = global_id.x;
                    if (index < arrayLength(&input)) {
                        output[index] = pow(input[index], ${exponent}f);
                    }
                }
            `,
      [Math.ceil(tensor.shape.size / 256)],
    );
  }

  mean(tensor: Tensor) {
    return this.createKernel(
      [1],
      [getGPUBuffer(tensor)],
      `
                @group(0) @binding(0) var<storage, read> input: array<f32>;                          
                @group(0) @binding(1) var<storage, read_write> output: array<f32>;                   
                                                                                                         
                @compute @workgroup_size(256)                                                        
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {                       
                    let index = global_id.x;                                                         
                    if (index == 0) {                                                                
                        var sum = 0.0;                                                               
                        for (var i = 0u; i < arrayLength(&input); i = i + 1u) {                      
                            sum += input[i];                                                         
                        }                                                                            
                        output[0] = sum / f32(arrayLength(&input));                                  
                    }                                                                                
                }
            `,
      [1],
    );
  }

  softmax(tensor: Tensor) {
    return this.createKernel(
      tensor.shape.data,
      [getGPUBuffer(tensor)],
      `
            @group(0) @binding(0) var<storage, read> input: array<f32>;
            @group(0) @binding(1) var<storage, read_write> output: array<f32>;

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let index = global_id.x;
                if (index < arrayLength(&input)) {
                    var max_val = input[0];
                    for (var i = 1u; i < arrayLength(&input); i = i + 1u) {
                        max_val = max(max_val, input[i]);
                    }

                    var sum = 0.0;
                    for (var i = 0u; i < arrayLength(&input); i = i + 1u) {
                        sum += exp(input[i] - max_val);
                    }

                    output[index] = exp(input[index] - max_val) / sum;
                }
            }
            `,
      [Math.ceil(tensor.shape.size / 256)],
    );
  }

  mul(tensorA: Tensor, tensorB: Tensor | number) {
    if (typeof tensorB === 'number') {
      return this.createKernel(
        tensorA.shape.data,
        [getGPUBuffer(tensorA)],
        `
                @group(0) @binding(0) var<storage, read> a: array<f32>;                              
                @group(0) @binding(1) var<storage, read_write> output: array<f32>;                   
                                                                                                        
                @compute @workgroup_size(256)                                                        
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {                       
                    let index = global_id.x;                                                         
                    if (index < arrayLength(&a)) {                                                   
                        output[index] = a[index] * ${tensorB}f;                                      
                    }                                                                                
                } 
                `,
        [Math.ceil(tensorA.shape.size / 256)],
      );
    } else {
      if (!tensorA.shape.data.every((dim, i) => dim === tensorB.shape.data[i])) {
        throw new Error('Tensor shapes must match for element-wise multiplication');
      }
      return this.createKernel(
        tensorA.shape.data,
        [getGPUBuffer(tensorA), getGPUBuffer(tensorB)],
        `
                @group(0) @binding(0) var<storage, read> a: array<f32>;                              
                @group(0) @binding(1) var<storage, read> b: array<f32>;                              
                @group(0) @binding(2) var<storage, read_write> output: array<f32>;                   
                                                                                                        
                @compute @workgroup_size(256)                                                        
                fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {                       
                    let index = global_id.x;                                                         
                    if (index < arrayLength(&a)) {                                                   
                        output[index] = a[index] * b[index];                                         
                    }                                                                                
                }
                `,
        [Math.ceil(tensorA.shape.size / 256)],
      );
    }
  }

  gt(tensor: Tensor, value: number) {
    return this.createKernel(
      tensor.shape.data,
      [getGPUBuffer(tensor)],
      `
            @group(0) @binding(0) var<storage, read> input: array<f32>;                              
            @group(0) @binding(1) var<storage, read_write> output: array<f32>;                       
                                                                                                        
            @compute @workgroup_size(256)                                                            
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {                           
                let index = global_id.x;                                                             
                if (index < arrayLength(&input)) {                                                   
                    output[index] = f32(input[index] > ${value}f);                                   
                }                                                                                    
            }
            `,
      [Math.ceil(tensor.shape.size / 256)],
    );
  }

  transpose(tensor: Tensor) {
    if (tensor.shape.data.length !== 2) {
      throw new Error('Transpose operation is only supported for 2D tensors');
    }
    const [rows, cols] = tensor.shape.data;
    return this.createKernel(
      [cols, rows],
      [getGPUBuffer(tensor)],
      `
            @group(0) @binding(0) var<storage, read> input: array<f32>;                              
            @group(0) @binding(1) var<storage, read_write> output: array<f32>;                       
                                                                                                        
            @compute @workgroup_size(16, 16)                                                         
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {                           
                let row = global_id.x;                                                               
                let col = global_id.y;                                                               
                let rows = ${rows}u;                                                                 
                let cols = ${cols}u;                                                                 
                                                                                                        
                if (row < rows && col < cols) {                                                      
                    output[col * rows + row] = input[row * cols + col];                              
                }                                                                                    
            } 
            `,
      [Math.ceil(rows / 16), Math.ceil(cols / 16)],
    );
  }

  sigmoid(tensor: Tensor) {
    return this.createKernel(
      tensor.shape.data,
      [getGPUBuffer(tensor)],
      `
            @group(0) @binding(0) var<storage, read> input: array<f32>;
            @group(0) @binding(1) var<storage, read_write> output: array<f32>;

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let index = global_id.x;
                if (index < arrayLength(&input)) {
                    output[index] = 1.0 / (1.0 + exp(-input[index]));
                }
            }
            `,
      [Math.ceil(tensor.shape.size / 256)],
    );
  }

  max(tensor: Tensor) {
    return this.createKernel(
      [1],
      [getGPUBuffer(tensor)],
      `
            @group(0) @binding(0) var<storage, read> input: array<f32>;
            @group(0) @binding(1) var<storage, read_write> output: array<f32>;

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let index = global_id.x;
                if (index == 0) {
                    var max_value = input[0];
                    for (var i = 1u; i < arrayLength(&input); i = i + 1u) {
                        max_value = max(max_value, input[i]);
                    }
                    output[0] = max_value;
                }
            }
            `,
      [1],
    );
  }

  sum(tensor: Tensor) {
    return this.createKernel(
      [1],
      [getGPUBuffer(tensor)],
      `
            @group(0) @binding(0) var<storage, read> input: array<f32>;
            @group(0) @binding(1) var<storage, read_write> output: array<f32>;

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                let index = global_id.x;
                if (index == 0) {
                    var sum: f32 = 0.0;
                    for (var i: u32 = 0u; i < arrayLength(&input); i++) {
                        sum += input[i];
                    }
                    output[0] = sum;
                }
            }
            `,
      [1],
    );
  }

  copy(tensorSource: Tensor, tensorDestination: Tensor) {
    const { device } = this.instance;
    const copyEncoder = device.createCommandEncoder();
    if ('buffer' in tensorSource)
      copyEncoder.copyBufferToBuffer(
        (tensorSource as GPUTensor).buffer,
        0,
        (tensorDestination as GPUTensor).buffer,
        0,
        tensorSource.shape.size * F32SIZE,
      );

    this.commands.push(copyEncoder.finish());
  }

  getStaging(source: GPUTensor) {
    let staging = source;
    if (!source.readable) {
      staging = this.empty(source.shape.data);
      this.copy(source, staging);
    }
    this.compute();
    this.commands = [];
    return staging;
  }

  clone(tensorSource: Tensor): Tensor {
    return this.getStaging(tensorSource as GPUTensor);
  }

  async item(tensor: Tensor): Promise<number> {
    if (tensor.numel() !== 1) {
      throw new Error('item() can only be called on tensors with a single element');
    }
    const staging = this.getStaging(tensor as GPUTensor);
    const data = await staging.readFloat32();
    return data[0] as number;
  }

  async print(...data: { toString?: () => string; asyncToString?: () => Promise<void> }[]) {
    const promises = data.map((d) => {
      return (async () => {
        if (d instanceof GPUTensor) {
          const staging = this.getStaging(d);

          return staging.asyncToString();
        }

        return d.toString?.() || d;
      })();
    });
    const values = await Promise.all(promises);
    console.log(values.join(' '));
  }
}

class GPUTensor implements Tensor {
  buffer: GPUBuffer;
  shape: Size;
  dtype: DType;
  readable: boolean;

  constructor(buffer: GPUBuffer, shape: Size, readable = false, dtype = DType.Float32) {
    this.buffer = buffer;
    this.shape = shape;
    this.dtype = dtype;
    this.readable = readable;
  }

  size(dim?: number) {
    if (dim === undefined) {
      return this.shape;
    }
    return this.shape.getDim(dim);
  }

  get device() {
    return Device.GPU;
  }

  set(array: Array<number>) {
    if (array) {
      const arrayBufferTensor = this.buffer.getMappedRange();
      new Float32Array(arrayBufferTensor).set(array);
      this.buffer.unmap();
      // console.log(this, arrayBufferTensor.byteLength, array);
    }
    return this;
  }

  numel(): number {
    return this.shape.size;
  }

  async read({ mode = GPUMapMode.READ } = {}) {
    await this.buffer.mapAsync(mode);
    return this.buffer.getMappedRange();
  }

  async readFloat32(options?: { mode?: 1 | undefined } | undefined) {
    const buffer = await this.read(options);
    const array = stridedToNestedFloat32Array(buffer, this.shape.data);
    this.buffer.unmap();

    return array;
  }

  async readArray<T>(options?: { mode?: 1 | undefined } | undefined): Promise<NestedArray<T>> {
    const buffer = await this.read(options);
    const array = stridedToNestedArray<T>(buffer, this.shape.data);
    this.buffer.unmap();

    return array;
  }

  async asyncToString() {
    const array = await this.readFloat32();
    return `tensor(${float32ArrayToString(array)})`;
  }
}
