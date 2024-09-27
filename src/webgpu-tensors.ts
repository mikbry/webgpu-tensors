
function buildShapeFromRecursiveArray(array: NDArray): Size {
    let a: number | NDArray = array;
    let shape = new Size([]);
    while (Array.isArray(a)) {
        shape.data = [...shape.data, a.length];
        a = a[0];
    }
    return shape;
}

function getShapeSize(dim: number[]): number {
    return dim.reduce((acc, s) => s * acc, 1);
}

function stridedToNDFloat32Array(buffer: ArrayBuffer, dim: number[], offset = 0, depth = 0): Float32NDArray {

    if (dim.length === 1) {
        const length = getShapeSize(dim) * F32SIZE;
        return new Float32Array(buffer.slice(offset, offset + length));
    }
    let array: Float32NDArray = [];
    for (let n = 0; n < dim[0]; n++) {
        let nestedShape = [...dim];
        nestedShape.shift();
        const length = getShapeSize(nestedShape) * F32SIZE;
        let i = length * n;
        let nestedArray = stridedToNDFloat32Array(buffer, nestedShape, offset + i, depth + 1);
        array.push(nestedArray);
    }
    return array;
}

function float32ArrayToString(array: Float32NDArray): string {
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

export interface WebGPUInstance {
    device: GPUDevice;
}

const F32SIZE = 4;

type TensorOptions = { usage: GPUFlagsConstant, mappedAtCreation: boolean | undefined };

export interface Tensor {
    buffer: GPUBuffer;
    shape: Size;
    dtype: DType;
    device: Device;
    readable: boolean;
    readFloat32(options?: { mode: GPUFlagsConstant }): Promise<Float32NDArray>;
    size(dim?: number): Size | number;
};

class Size {
    data: number[];
    constructor(data: number[]) {
        this.data = data;
    }

    get length() {
        return this.data.length;
    }

    get size() {
        return getShapeSize(this.data);
    }

    getDim(dim: number) {
        return this.data[dim];
    }

    toString() {
        return `Size[${this.data.join(',')}]`;
    }
}

// export type Shape = Size;

export enum DType {
    Float32 = 'Float32',
}

export enum Device {
    GPU = 'GPU',
}


type NDArray = Array<NDArray | number>;

type Float32NDArray = Array<Float32NDArray> | Float32Array;

type Shape = number[];

export interface Tensors {
    empty(shape: Shape, options?: Partial<TensorOptions> | undefined): Promise<Tensor>;
    ones(shape: Shape, options?: Partial<TensorOptions> | undefined): Promise<Tensor>;
    rand(shape: Shape, options?: Partial<TensorOptions> | undefined): Promise<Tensor>;
    randn(shape: Shape, options?: Partial<TensorOptions> | undefined): Promise<Tensor>;
    zeros(shape: Shape, options?: Partial<TensorOptions> | undefined): Promise<Tensor>;
    tensor(array: NDArray, options?: Partial<TensorOptions> | undefined): Promise<Tensor>;

    matmul(tensorA: Tensor, tensorB: Tensor): Promise<Tensor>;
    sub(tensorA: Tensor, tensorB: Tensor): Promise<Tensor>;
    pow(tensor: Tensor, exponent: number): Promise<Tensor>;
    mul(tensorA: Tensor, tensorB: Tensor | number): Promise<Tensor>;
    gt(tensor: Tensor, value: number): Promise<Tensor>;
    transpose(tensor: Tensor): Promise<Tensor>;

    maximum(tensor: Tensor, value: number): Promise<Tensor>;
    max(tensor: Tensor): Promise<Tensor>;

    mean(tensor: Tensor): Promise<Tensor>;

    compute(): Promise<void>;
    copy(tensorSource: Tensor, tensorDestination: Tensor): Promise<void>;

    destroy(): void;

    print(...data: unknown[]): Promise<void>;
};

class WebGPUTensors implements Tensors {
    static _instance?: WebGPUInstance;

    commands: Array<any>;
    hasComputeOnce: boolean;

    constructor() {
        this.commands = [];
        this.hasComputeOnce = false;
    }

    static create() {
        return new WebGPUTensors() as Tensors;
    }

    async max(tensor: Tensor): Promise<Tensor> {
        const { device } = await this.instance;
        const result = await this.empty([1], { usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

        const computePipeline = device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: device.createShaderModule({
                    code: `
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
                    `
                }),
                entryPoint: 'main'
            }
        });

        const bindGroup = device.createBindGroup({
            layout: computePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: tensor.buffer } },
                { binding: 1, resource: { buffer: result.buffer } },
            ],
        });

        const commandEncoder = device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(computePipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(1);
        passEncoder.end();

        this.commands.push(commandEncoder.finish());

        return result;
    }

    async init(): Promise<WebGPUInstance> {
        if (!("gpu" in navigator)) {
            throw new Error("WebGPU not supported on this browser.");
        }

        if (!WebGPUTensors._instance) {
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
                throw new Error("No appropriate GPUAdapter found.");
            }

            const device = await adapter.requestDevice();

            WebGPUTensors._instance = { device };
        }
        return WebGPUTensors._instance;
    }

    get instance() {
        return (async () => {
            return this.init();
        })();
    }

    destroy() {
        WebGPUTensors._instance?.device.destroy();
    }

    async create(shape: Size, options: TensorOptions) {
        const { usage, mappedAtCreation } = options;
        const { device } = await this.instance || {};
        let size = shape.size * F32SIZE;
        return new GPUTensor(device.createBuffer({
            mappedAtCreation,
            size,
            usage
        }), shape, !mappedAtCreation);
    }

    async empty(shape: Shape, options?: Partial<TensorOptions> | undefined) {
        const { usage = GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, mappedAtCreation = false } = options || {};
        return this.create(new Size(shape), { usage, mappedAtCreation });
    }

    async tensor(array: NDArray, options?: Partial<TensorOptions> | undefined): Promise<Tensor> {
        let shape = buildShapeFromRecursiveArray(array);
        let flat = (shape.length === 1 ? array : array.flat(shape.length as 10)) as number[];
        const { usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, mappedAtCreation = true } = options || {};
        let tensor = await this.create(shape, { usage, mappedAtCreation });
        return tensor.set(flat);
    }

    async ones(shape: Shape, options?: Partial<TensorOptions> | undefined): Promise<Tensor> {
        const { usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, mappedAtCreation = true } = options || {};
        let tensor = await this.create(new Size(shape), { usage, mappedAtCreation });
        const array = new Array(tensor.shape.size).fill(1);
        return tensor.set(array);
    }

    async rand(shape: Shape, options?: Partial<TensorOptions> | undefined): Promise<Tensor> {
        const { usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, mappedAtCreation = true } = options || {};
        let tensor = await this.create(new Size(shape), { usage, mappedAtCreation });
        const array = Array.from(new Array(tensor.size), () => Math.random());
        return tensor.set(array);
    }

    async randn(shape: Shape, options?: Partial<TensorOptions> | undefined): Promise<Tensor> {
        const { usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, mappedAtCreation = true } = options || {};
        let tensor = await this.create(new Size(shape), { usage, mappedAtCreation });

        // Box-Muller transform
        const boxMullerTransform = () => {
            const u1 = Math.random();
            const u2 = Math.random();
            const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
            return z0;
        };

        const array = Array.from(new Array(tensor.size), () => boxMullerTransform());
        return tensor.set(array);
    }

    async zeros(shape: Shape, options?: Partial<TensorOptions> | undefined): Promise<Tensor> {
        const { usage = GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC, mappedAtCreation = true } = options || {};
        let tensor = await this.create(new Size(shape), { usage, mappedAtCreation });
        const array = new Array(tensor.shape.size).fill(0);
        return tensor.set(array);
    }

    async compute() {
        if (this.hasComputeOnce && this.commands.length === 0) {
            return;
        }
        const instance = await this.instance;
        instance?.device.queue.submit(this.commands);
        this.hasComputeOnce = true;
    }


    async matmul(input: Tensor, other: Tensor) {
        const inputShape = input.shape.data;
        const otherShape = other.shape.data;

        if (inputShape[1] !== otherShape[0]) {
            throw new Error("Incompatible matrix dimensions for multiplication");
        }

        const destination = await this.empty(inputShape);
        const { device } = await this.instance;
        const computePipeline = device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: device.createShaderModule({
                    code: `
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
              `
                }),
                entryPoint: 'main'
            }
        });

        const bindGroupLayout = computePipeline.getBindGroupLayout(0);
        const bindGroup = device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                { binding: 0, resource: { buffer: input.buffer } },
                { binding: 1, resource: { buffer: other.buffer } },
                { binding: 2, resource: { buffer: destination.buffer } },
            ],
        });

        const commandEncoder = device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(computePipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(
            Math.ceil(inputShape[0] / 8),
            Math.ceil(otherShape[1] / 8)
        );
        passEncoder.end();

        this.commands.push(commandEncoder.finish());

        return destination;
    }


    async maximum(tensor: Tensor, value: number): Promise<Tensor> {
        const { device } = await this.instance;
        const result = await this.empty(tensor.shape.data, { usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

        const computePipeline = device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: device.createShaderModule({
                    code: `
                    @group(0) @binding(0) var<storage, read> input: array<f32>;
                    @group(0) @binding(1) var<storage, read_write> output: array<f32>;

                    @compute @workgroup_size(256)
                    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                        let index = global_id.x;
                        if (index < arrayLength(&input)) {
                            output[index] = max(input[index], ${value}f);
                        }
                    }
                    `
                }),
                entryPoint: 'main'
            }
        });

        const bindGroup = device.createBindGroup({
            layout: computePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: tensor.buffer } },
                { binding: 1, resource: { buffer: result.buffer } },
            ],
        });

        const commandEncoder = device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(computePipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(Math.ceil(tensor.shape.size / 256));
        passEncoder.end();

        this.commands.push(commandEncoder.finish());

        return result;
    }

    async sub(tensorA: Tensor, tensorB: Tensor): Promise<Tensor> {
        const { device } = await this.instance;
        if (!tensorA.shape.data.every((dim, i) => dim === tensorB.shape.data[i])) {
            throw new Error("Tensor shapes must match for subtraction");
        }
        const result = await this.empty(tensorA.shape.data, { usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

        const computePipeline = device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: device.createShaderModule({
                    code: `
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
                    `
                }),
                entryPoint: 'main'
            }
        });

        const bindGroup = device.createBindGroup({
            layout: computePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: tensorA.buffer } },
                { binding: 1, resource: { buffer: tensorB.buffer } },
                { binding: 2, resource: { buffer: result.buffer } },
            ],
        });

        const commandEncoder = device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(computePipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(Math.ceil(tensorA.shape.size / 256));
        passEncoder.end();

        this.commands.push(commandEncoder.finish());

        return result;
    }

    async pow(tensor: Tensor, exponent: number): Promise<Tensor> {
        const { device } = await this.instance;
        const result = await this.empty(tensor.shape.data, { usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

        const computePipeline = device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: device.createShaderModule({
                    code: `
                    @group(0) @binding(0) var<storage, read> input: array<f32>;
                    @group(0) @binding(1) var<storage, read_write> output: array<f32>;

                    @compute @workgroup_size(256)
                    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
                        let index = global_id.x;
                        if (index < arrayLength(&input)) {
                            output[index] = pow(input[index], ${exponent}f);
                        }
                    }
                    `
                }),
                entryPoint: 'main'
            }
        });

        const bindGroup = device.createBindGroup({
            layout: computePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: tensor.buffer } },
                { binding: 1, resource: { buffer: result.buffer } },
            ],
        });

        const commandEncoder = device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(computePipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(Math.ceil(tensor.shape.size / 256));
        passEncoder.end();

        this.commands.push(commandEncoder.finish());

        return result;
    }

    async copy(tensorSource: Tensor, tensorDestination: Tensor) {
        const { device } = await this.instance;
        const copyEncoder = device.createCommandEncoder();
        if ("buffer" in tensorSource)
            copyEncoder.copyBufferToBuffer(
                (tensorSource as GPUTensor).buffer,
                0,
                (tensorDestination as GPUTensor).buffer,
                0,
                tensorSource.shape.size * F32SIZE
            );

        this.commands.push(copyEncoder.finish());
    }

    async mean(tensor: Tensor): Promise<Tensor> {                                                        
        const { device } = await this.instance;                                                          
        const result = await this.empty([1], { usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
                                                                                                         
        const computePipeline = device.createComputePipeline({                                           
            layout: 'auto',                                                                              
            compute: {                                                                                   
                module: device.createShaderModule({                                                      
                    code: `                                                                              
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
                    `                                                                                    
                }),                                                                                      
                entryPoint: 'main'                                                                       
            }                                                                                            
        });                                                                                              
                                                                                                         
        const bindGroup = device.createBindGroup({                                                       
            layout: computePipeline.getBindGroupLayout(0),                                               
            entries: [                                                                                   
                { binding: 0, resource: { buffer: tensor.buffer } },                                     
                { binding: 1, resource: { buffer: result.buffer } },                                     
            ],                                                                                           
        });                                                                                              
                                                                                                         
        const commandEncoder = device.createCommandEncoder();                                            
        const passEncoder = commandEncoder.beginComputePass();                                           
        passEncoder.setPipeline(computePipeline);                                                        
        passEncoder.setBindGroup(0, bindGroup);                                                          
        passEncoder.dispatchWorkgroups(1);                                                               
        passEncoder.end();                                                                               
                                                                                                         
        this.commands.push(commandEncoder.finish());                                                     
                                                                                                         
        return result;                                                                                   
    }

    async softmax(tensor: Tensor): Promise<Tensor> {
        const { device } = await this.instance;
        const result = await this.empty(tensor.shape.data, { usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

        const computePipeline = device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: device.createShaderModule({
                    code: `
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
                    `
                }),
                entryPoint: 'main'
            }
        });

        const bindGroup = device.createBindGroup({
            layout: computePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: tensor.buffer } },
                { binding: 1, resource: { buffer: result.buffer } },
            ],
        });

        const commandEncoder = device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(computePipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(Math.ceil(tensor.shape.size / 256));
        passEncoder.end();

        this.commands.push(commandEncoder.finish());

        return result;
    }

        
    mul(tensorA: Tensor, tensorB: Tensor | number): Promise<Tensor> {
        throw new Error("Method not implemented.");
    }
    
    gt(tensor: Tensor, value: number): Promise<Tensor> {
        throw new Error("Method not implemented.");
    }
    
    transpose(tensor: Tensor): Promise<Tensor> {
        throw new Error("Method not implemented.");
    }

    async getStaging(source: GPUTensor) {
        let staging = source;
        if (!source.readable) {
            staging = await this.empty(source.shape.data);
            await this.copy(source, staging);
        }
        await this.compute();
        this.commands = [];
        return staging;
    }

    async print(...data: ({ toString?: () => {}; asyncToString?: () => Promise<void> })[]) {
        const promises = data.map((d) => {
            return (async () => {
                if (d instanceof GPUTensor) {
                    const staging = await this.getStaging(d);
                    return staging.asyncToString();
                }

                return d.toString?.() || d;
            })()
        })
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

    async set(array: Array<number>) {
        if (array) {
            const arrayBufferTensor = this.buffer.getMappedRange();
            new Float32Array(arrayBufferTensor).set(array);
            this.buffer.unmap();
            // console.log(this, arrayBufferTensor.byteLength, array);
        }
        return this;
    }

    async read({ mode = GPUMapMode.READ } = {}) {
        await this.buffer.mapAsync(mode);
        return this.buffer.getMappedRange();
    }

    async readFloat32(options?: { mode: GPUFlagsConstant }) {
        const buffer = await this.read(options);
        const array = stridedToNDFloat32Array(buffer, this.shape.data);
        this.buffer.unmap();

        return array;
    }

    async asyncToString() {
        const array = await this.readFloat32();
        return `tensor(${float32ArrayToString(array)})`;
    }
}

let tensors: Tensors | undefined;

export default (function (): Tensors {
    if (!tensors) {
        tensors = WebGPUTensors.create();
    }
    return tensors;
})();

export class NN {
    static async relu(t: Tensors, x: Tensor): Promise<Tensor> {
        return t.maximum(x, 0);
    }
}
