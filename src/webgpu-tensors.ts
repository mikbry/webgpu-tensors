
function buildShapeFromNestedArray(array: NestedArray<number>): Size {
    let a: number | NestedArray<number> = array;
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

function stridedToNestedFloat32Array(buffer: ArrayBuffer, dim: number[], offset = 0, depth = 0): Float32NestedArray {

    if (dim.length === 1) {
        const length = getShapeSize(dim) * F32SIZE;
        return new Float32Array(buffer.slice(offset, offset + length));
    }
    let array: Float32NestedArray = [];
    for (let n = 0; n < dim[0]; n++) {
        let nestedShape = [...dim];
        nestedShape.shift();
        const length = getShapeSize(nestedShape) * F32SIZE;
        let i = length * n;
        let nestedArray = stridedToNestedFloat32Array(buffer, nestedShape, offset + i, depth + 1);
        array.push(nestedArray);
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

export interface WebGPUInstance {
    device: GPUDevice;
}

const F32SIZE = 4;

type TensorOptions = { usage: GPUFlagsConstant; mappedAtCreation: boolean | undefined; readable: boolean };

export interface Tensor {
    buffer: GPUBuffer;
    shape: Size;
    dtype: DType;
    device: Device;
    readable: boolean;
    readFloat32(options?: { mode: GPUFlagsConstant }): Promise<Float32NestedArray>;
    size(dim?: number): Size | number;
    numel(): number;
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


type NestedArray<T> = Array<NestedArray<T> | T>;

type Float32NestedArray = Array<Float32NestedArray> | Float32Array;

type Shape = number[];

export interface Tensors {
    empty(shape: Shape, options?: Partial<TensorOptions> | undefined): Promise<Tensor>;
    ones(shape: Shape, options?: Partial<TensorOptions> | undefined): Promise<Tensor>;
    rand(shape: Shape, options?: Partial<TensorOptions> | undefined): Promise<Tensor>;
    randn(shape: Shape, options?: Partial<TensorOptions> | undefined): Promise<Tensor>;
    zeros(shape: Shape, options?: Partial<TensorOptions> | undefined): Promise<Tensor>;
    tensor(array: NestedArray<number>, options?: Partial<TensorOptions> | undefined): Promise<Tensor>;

    matmul(tensorA: Tensor, tensorB: Tensor): Promise<Tensor>;
    sub(tensorA: Tensor, tensorB: Tensor): Promise<Tensor>;
    pow(tensor: Tensor, exponent: number): Promise<Tensor>;
    mul(tensorA: Tensor, tensorB: Tensor | number): Promise<Tensor>;
    gt(tensor: Tensor, value: number): Promise<Tensor>;
    transpose(tensor: Tensor): Promise<Tensor>;

    maximum(tensor: Tensor, value: number): Promise<Tensor>;
    relu(x: Tensor): Promise<Tensor>;

    max(tensor: Tensor): Promise<Tensor>;

    mean(tensor: Tensor): Promise<Tensor>;

    sigmoid(tensor: Tensor): Promise<Tensor>;

    reset(): void;
    compute(): Promise<void>;
    copy(tensorSource: Tensor, tensorDestination: Tensor): Promise<void>;

    item(tensor: Tensor): Promise<number>;

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

    reset() {
        this.commands = [];
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

    async createTensor(shape: Size, options: TensorOptions) {
        const { usage, mappedAtCreation, readable } = options;
        const { device } = await this.instance || {};
        let size = shape.size * F32SIZE;
        return new GPUTensor(device.createBuffer({
            mappedAtCreation,
            size,
            usage
        }), shape, readable);
    }

    async resultTensor(shape: Shape, options?: Partial<TensorOptions> | undefined) {
        const { usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, mappedAtCreation = false } = options || {};
        return this.createTensor(new Size(shape), { usage, mappedAtCreation, readable: false });
    }

    async buildTensor(buildArray: (i: number) => number[], shape: Shape, options?: Partial<TensorOptions> | undefined) {
        const { usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, mappedAtCreation = true } = options || {};
        let tensor = await this.createTensor(new Size(shape), { usage, mappedAtCreation, readable: false });
        return tensor.set(buildArray(tensor.shape.size));
    }

    async empty(shape: Shape, options?: Partial<TensorOptions> | undefined) {
        const { usage = GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, mappedAtCreation = false } = options || {};
        return this.createTensor(new Size(shape), { usage, mappedAtCreation, readable: true });
    }

    async tensor(array: NestedArray<number>, options?: Partial<TensorOptions> | undefined): Promise<Tensor> {
        let shape = buildShapeFromNestedArray(array);
        let stridedArray = (shape.length === 1 ? array : array.flat(shape.length as 10)) as number[];
        return this.buildTensor(() => stridedArray, shape.data, options);
    }

    async ones(shape: Shape, options?: Partial<TensorOptions> | undefined): Promise<Tensor> {
        return this.buildTensor((size) => new Array(size).fill(1), shape, options);
    }

    async rand(shape: Shape, options?: Partial<TensorOptions> | undefined): Promise<Tensor> {
        return this.buildTensor((size) => Array.from(new Array(size), () => Math.random()), shape, options);
    }

    async randn(shape: Shape, options?: Partial<TensorOptions> | undefined): Promise<Tensor> {
        // Box-Muller transform
        const boxMullerTransform = () => {
            const u1 = Math.random();
            const u2 = Math.random();
            const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
            return z0;
        };
        return this.buildTensor((size) => Array.from(new Array(size), () => boxMullerTransform()), shape, options);
    }

    async zeros(shape: Shape, options?: Partial<TensorOptions> | undefined): Promise<Tensor> {
        return this.buildTensor((size) => new Array(size).fill(0), shape, options);
    }

    async compute() {
        if (this.hasComputeOnce && this.commands.length === 0) {
            return;
        }
        const instance = await this.instance;
        instance?.device.queue.submit(this.commands);
        this.hasComputeOnce = true;
        this.reset();
    }

    async createKernel(resultShape: Shape, entries: any[], code: string, workGroupCounts: number[]) {
        const result = await this.resultTensor(resultShape);
        const { device } = await this.instance;
        const computePipeline = device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: device.createShaderModule({
                    code
                }),
                entryPoint: 'main'
            }
        });

        const bindGroupLayout = computePipeline.getBindGroupLayout(0);
        const bindGroup = device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
                ...entries,
                { binding: entries.length, resource: { buffer: result.buffer } },
            ],
        });

        const commandEncoder = device.createCommandEncoder();
        const passEncoder = commandEncoder.beginComputePass();
        passEncoder.setPipeline(computePipeline);
        passEncoder.setBindGroup(0, bindGroup);
        passEncoder.dispatchWorkgroups(
            workGroupCounts[0],
            workGroupCounts[1],
            workGroupCounts[3]
        );
        passEncoder.end();
        this.commands.push(commandEncoder.finish());

        return result;
    }

    async matmul(input: Tensor, other: Tensor) {
        const inputShape = input.shape.data;
        const otherShape = other.shape.data;

        if (inputShape[1] !== otherShape[0]) {
            let error = `Incompatible matrix dimensions for multiplication ${inputShape[1]} ${otherShape[0]}`;
            throw new Error(error);
        }

        return this.createKernel(
            [inputShape[0], otherShape[1]],
            [
                { binding: 0, resource: { buffer: input.buffer } },
                { binding: 1, resource: { buffer: other.buffer } },
            ],
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
            [Math.ceil(inputShape[0] / 8),
            Math.ceil(otherShape[1] / 8)]
        );
    }


    async maximum(tensor: Tensor, value: number): Promise<Tensor> {
        return this.createKernel(
            tensor.shape.data,
            [{ binding: 0, resource: { buffer: tensor.buffer } }],
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
            [Math.ceil(tensor.shape.size / 64)]
        );
    }

    async relu(x: Tensor): Promise<Tensor> {
        return this.maximum(x, 0);
    }

    async sub(tensorA: Tensor, tensorB: Tensor): Promise<Tensor> {
        if (!tensorA.shape.data.every((dim, i) => dim === tensorB.shape.data[i])) {
            throw new Error("Tensor shapes must match for subtraction");
        }

        return this.createKernel(
            tensorA.shape.data,
            [{ binding: 0, resource: { buffer: tensorA.buffer } },
            { binding: 1, resource: { buffer: tensorB.buffer } },],
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
            [Math.ceil(tensorA.shape.size / 256)]
        );
    }

    async pow(tensor: Tensor, exponent: number): Promise<Tensor> {
        return this.createKernel(
            tensor.shape.data,
            [{ binding: 0, resource: { buffer: tensor.buffer } }],
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
            [Math.ceil(tensor.shape.size / 256)]
        );
    }

    async mean(tensor: Tensor): Promise<Tensor> {
        return this.createKernel(
            [1],
            [{ binding: 0, resource: { buffer: tensor.buffer } }],
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
            [1]
        );
    }

    async softmax(tensor: Tensor): Promise<Tensor> {
        return this.createKernel(
            tensor.shape.data,
            [{ binding: 0, resource: { buffer: tensor.buffer } }],
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
            [Math.ceil(tensor.shape.size / 256)]
        );
    }

    async mul(tensorA: Tensor, tensorB: Tensor | number): Promise<Tensor> {
        if (typeof tensorB === 'number') {
            return this.createKernel(
                tensorA.shape.data,
                [{ binding: 0, resource: { buffer: tensorA.buffer } }],
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
                [Math.ceil(tensorA.shape.size / 256)]
            );
        } else {
            if (!tensorA.shape.data.every((dim, i) => dim === tensorB.shape.data[i])) {
                throw new Error("Tensor shapes must match for element-wise multiplication");
            }
            return this.createKernel(
                tensorA.shape.data,
                [{ binding: 0, resource: { buffer: tensorA.buffer } },
                { binding: 1, resource: { buffer: tensorB.buffer } },],
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
                [Math.ceil(tensorA.shape.size / 256)]
            );
        }
    }

    async gt(tensor: Tensor, value: number): Promise<Tensor> {
        return this.createKernel(
            tensor.shape.data,
            [{ binding: 0, resource: { buffer: tensor.buffer } }],
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
            [Math.ceil(tensor.shape.size / 256)]
        );
    }

    async transpose(tensor: Tensor): Promise<Tensor> {
        if (tensor.shape.data.length !== 2) {
            throw new Error("Transpose operation is only supported for 2D tensors");
        }
        const [rows, cols] = tensor.shape.data;
        return this.createKernel(
            [cols, rows],
            [{ binding: 0, resource: { buffer: tensor.buffer } }],
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
            [Math.ceil(rows / 16), Math.ceil(cols / 16)]
        );
    }


    async sigmoid(tensor: Tensor): Promise<Tensor> {
        return this.createKernel(
            tensor.shape.data,
            [{ binding: 0, resource: { buffer: tensor.buffer } }],
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
            [Math.ceil(tensor.shape.size / 256)]
        );
    }

    async max(tensor: Tensor): Promise<Tensor> {
        return this.createKernel(
            [1],
            [{ binding: 0, resource: { buffer: tensor.buffer } }],
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
            [1]
        );
    }

    async sum(tensor: Tensor): Promise<Tensor> {
        return this.createKernel(
            [1],
            [{ binding: 0, resource: { buffer: tensor.buffer } }],
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
            [1]
        );
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


    async item(tensor: Tensor): Promise<number> {
        if (tensor.numel() !== 1) {
            throw new Error("item() can only be called on tensors with a single element");
        }
        const staging = await this.getStaging(tensor as GPUTensor);
        const data = await staging.readFloat32();
        return data[0] as number;
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

    numel(): number {
        return this.shape.size;
    }

    async read({ mode = GPUMapMode.READ } = {}) {
        await this.buffer.mapAsync(mode);
        return this.buffer.getMappedRange();
    }

    async readFloat32(options?: { mode: GPUFlagsConstant }) {
        const buffer = await this.read(options);
        const array = stridedToNestedFloat32Array(buffer, this.shape.data);
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
