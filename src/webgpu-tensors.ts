export interface WebGPUInstance {
    device: GPUDevice;
}

const F32SIZE = 4;

type TensorOptions = { usage?: GPUFlagsConstant, mappedAtCreation?: boolean | undefined };

export interface Tensor {
    buffer: GPUBuffer;
    shape: Shape;
    dtype: DType;
    device: Device;
    readF16(options?: { mode: GPUFlagsConstant }): Promise<Float32Array>;
};

export type Shape = number[];

export enum DType {
    Float16 = 'Float16',
}

export enum Device {
    GPU = 'GPU',
}

export interface Tensors {
    empty(shape: Shape, options: TensorOptions | undefined): Promise<Tensor>;
    ones(shape: Shape, options: TensorOptions | undefined): Promise<Tensor>;
    rand(shape: Shape, options: TensorOptions | undefined): Promise<Tensor>;
    zeros(shape: Shape, options: TensorOptions | undefined): Promise<Tensor>;
    tensor(array: Array<Array<number>>, options?: TensorOptions | undefined): Promise<Tensor>;

    matmul(tensorA: Tensor, tensorB: Tensor): Promise<void>;

    compute(): Promise<void>;
    copy(tensorSource: Tensor, tensorDestination: Tensor): Promise<void>;
};

class WebGPUTensors implements Tensors {
    _instance?: WebGPUInstance;

    commands: Array<any>;

    constructor() {
        this.commands = [];
    }

    static create() {
        return new WebGPUTensors() as Tensors;
    }

    async init(): Promise<WebGPUInstance> {
        if (!("gpu" in navigator)) {
            throw new Error("WebGPU not supported on this browser.");
        }

        if (!this._instance) {
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) {
                throw new Error("No appropriate GPUAdapter found.");
            }

            const device = await adapter.requestDevice();

            this._instance = { device };
        }
        return this._instance;
    }

    get instance() {
        return (async () => {
            return this.init();
        })();
    }

    async empty(shape: Shape, options: TensorOptions | undefined) {
        const { usage = GPUBufferUsage.STORAGE, mappedAtCreation = false } = options || {};
        const { device } = await this.instance || {};
        const [cols, rows] = shape;
        let size = cols * rows * F32SIZE;
        return new GPUTensor(device.createBuffer({
            mappedAtCreation,
            size,
            usage
        }), shape);
    }

    async tensor(array: Array<Array<number>>, options?: TensorOptions | undefined): Promise<Tensor> {
        let tensor = await this.empty([array[0].length, array.length], { ...options, mappedAtCreation: true });
        return tensor.set(array);
    }

    async ones(shape: Shape, options?: TensorOptions | undefined): Promise<Tensor> {
        let tensor = await this.empty(shape, { ...options, mappedAtCreation: true });
        // TODO build array with 1 values
        return tensor.set([]);
    }

    async rand(shape: Shape, options?: TensorOptions | undefined): Promise<Tensor> {
        let tensor = await this.empty(shape, { ...options, mappedAtCreation: true });
        // TODO build array with random values
        return tensor.set([]);
    }

    async zeros(shape: Shape, options?: TensorOptions | undefined): Promise<Tensor> {
        let tensor = await this.empty(shape, { ...options, mappedAtCreation: true });
        // TODO build array with zeros values
        return tensor.set([]);
    }

    async compute() {
        const instance = await this.instance;
        instance?.device.queue.submit(this.commands);
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
}

class GPUTensor implements Tensor {
    buffer: GPUBuffer;
    shape: Shape;
    dtype: DType;

    constructor(buffer: GPUBuffer, shape: Shape, dtype = DType.Float16) {
        this.buffer = buffer;
        this.shape = shape;
        this.dtype = dtype;
    }

    get device() {
        return Device.GPU;
    }

    async set(array: Array<Array<number>>) {
        if (array) {
            const arrayBufferTensor = this.buffer.getMappedRange();
            new Float32Array(arrayBufferTensor).set(array[0]);
            this.buffer.unmap();
        }
        return this;
    }

    async read({ mode = GPUMapMode.READ } = {}) {
        await this.buffer.mapAsync(mode);
        return this.buffer.getMappedRange();
    }

    async readF16(options?: { mode: GPUFlagsConstant }) {
        const array = await this.read(options);
        return (new Float32Array(array));
    }
}

export default WebGPUTensors;
