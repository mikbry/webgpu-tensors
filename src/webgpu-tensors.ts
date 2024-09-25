export interface WebGPUInstance {
    device: GPUDevice;
}

const F32SIZE = 4;

type TensorOptions = { usage: GPUFlagsConstant, mappedAtCreation: boolean | undefined };

export interface Tensor {
    buffer: GPUBuffer;
    shape: Shape;
    dtype: DType;
    device: Device;
    readable: boolean;
    readF16(options?: { mode: GPUFlagsConstant }): Promise<Float32Array>;
};

export type Shape = number[];

export enum DType {
    Float16 = 'Float16',
}

export enum Device {
    GPU = 'GPU',
}


type RecursiveArray = Array<RecursiveArray | number>;

export interface Tensors {
    empty(shape: Shape, options?: Partial<TensorOptions> | undefined): Promise<Tensor>;
    ones(shape: Shape, options?: Partial<TensorOptions> | undefined): Promise<Tensor>;
    rand(shape: Shape, options?: Partial<TensorOptions> | undefined): Promise<Tensor>;
    zeros(shape: Shape, options?: Partial<TensorOptions> | undefined): Promise<Tensor>;
    tensor(array: RecursiveArray, options?: Partial<TensorOptions> | undefined): Promise<Tensor>;

    matmul(tensorA: Tensor, tensorB: Tensor): Promise<void>;

    compute(): Promise<void>;
    copy(tensorSource: Tensor, tensorDestination: Tensor): Promise<void>;

    destroy(): void;

    print(...data: unknown[]): Promise<void>;
};

function buildShapeFromRecursiveArray(array: RecursiveArray): Array<number> {
    let a: number | RecursiveArray = array;
    let shape: Shape = [];
    while (Array.isArray(a)) {
        shape = [...shape, a.length];
        a = a[0];
    }
    return shape;
}

function getShapeSize(shape: Shape): number {
    return shape.reduce((s, acc) => s * acc, 1);
}

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

    async create(shape: Shape, options: TensorOptions) {
        const { usage, mappedAtCreation } = options;
        const { device } = await this.instance || {};
        let size = getShapeSize(shape) * F32SIZE;
        console.log('create Tensor', usage, shape, mappedAtCreation)
        return new GPUTensor(device.createBuffer({
            mappedAtCreation,
            size,
            usage
        }), shape, !mappedAtCreation);
    }

    async empty(shape: Shape, options?: Partial<TensorOptions> | undefined) {
        const { usage =  GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST, mappedAtCreation = false } = options || {};
        return this.create(shape, { usage, mappedAtCreation });
    }

    async tensor(array: RecursiveArray, options?: Partial<TensorOptions> | undefined): Promise<Tensor> {
        let shape = buildShapeFromRecursiveArray(array);
        let flat = (shape.length === 1 ? array : array.flat()) as number[];
        const { usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, mappedAtCreation = true } = options || {};
        let tensor = await this.create(shape, {usage, mappedAtCreation });
        return tensor.set(flat);
    }

    async ones(shape: Shape, options?: Partial<TensorOptions> | undefined): Promise<Tensor> {
        const { usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, mappedAtCreation = true } = options || {};
        let tensor = await this.create(shape, {usage, mappedAtCreation });
        // TODO build array with 1 values
        const array = new Array(tensor.size).fill(1);
        return tensor.set(array);
    }

    async rand(shape: Shape, options?: Partial<TensorOptions> | undefined): Promise<Tensor> {
        const { usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC, mappedAtCreation = true } = options || {};
        let tensor = await this.create(shape, {usage, mappedAtCreation });
        const array = Array.from(new Array(tensor.size), () => Math.random());
        return tensor.set(array);
    }

    async zeros(shape: Shape, options?: Partial<TensorOptions> | undefined): Promise<Tensor> {
        const { usage = GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC, mappedAtCreation = true } = options || {};
        let tensor = await this.create(shape, {usage, mappedAtCreation });
        const array = new Array(tensor.size).fill(0);
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


    async matmul(tensorSource: Tensor, tensorDestination: Tensor) {
        // TODO matmul
        const { device } = await this.instance;
        const copyEncoder = device.createCommandEncoder();
        if ("buffer" in tensorSource)
            copyEncoder.copyBufferToBuffer(
                (tensorSource as GPUTensor).buffer, /* source buffer */
                0 /* source offset */,
                (tensorDestination as GPUTensor).buffer, /* destination buffer */
                0 /* destination offset */,
                4 * F32SIZE /* size */
            );

        // Submit copy commands.
        this.commands.push(copyEncoder.finish());
    }

    async copy(tensorSource: Tensor, tensorDestination: Tensor) {
        // await this.compute();
        const { device } = await this.instance;
        const copyEncoder = device.createCommandEncoder();
        if ("buffer" in tensorSource)
            copyEncoder.copyBufferToBuffer(
                (tensorSource as GPUTensor).buffer, /* source buffer */
                0 /* source offset */,
                (tensorDestination as GPUTensor).buffer, /* destination buffer */
                0 /* destination offset */,
                4 * F32SIZE /* size */
            );

        // Submit copy commands.
        this.commands.push(copyEncoder.finish());
    }

    async getStaging(source: GPUTensor) {
        let staging = source;
        if (!source.readable) {
            staging = await this.empty(source.shape);
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
                    console.log('staging', staging);
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
    shape: Shape;
    dtype: DType;
    readable: boolean;

    constructor(buffer: GPUBuffer, shape: Shape, readable = false, dtype = DType.Float16) {
        this.buffer = buffer;
        this.shape = shape;
        this.dtype = dtype;
        this.readable = readable;
    }

    get size() {
        return getShapeSize(this.shape);
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

    async readF16(options?: { mode: GPUFlagsConstant }) {
        const array = await this.read(options);
        const data = array.slice(0);
        //console.log('array', fArray);
        this.buffer.unmap();

        return new Float32Array(data);
    }

    async asyncToString() {
        const array = await this.readF16();
        console.log('array', array)
        return `tensor([${array}])`;
    }
}

let tensors: Tensors | undefined;

export default (function (): Tensors {
    if (!tensors) {
        tensors = WebGPUTensors.create();
    }
    return tensors;
})();
