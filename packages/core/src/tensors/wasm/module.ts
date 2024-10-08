import { Tensor, TensorsOperations } from "../../types";

export type WASMTensorsImpl = TensorsOperations & {
    init(): void;
    item(tensor: Tensor): number;
    read_array<T>(tensor: Tensor): Promise<T[]>;
    read_float32<T>(tensor: Tensor): Promise<T[]>;
    mul_scalar(tensorA: Tensor, tensorB: Tensor | number): Tensor;
};

export default async function importWASMModule() {
    const module = await import('../../../../wasm/webgpu-tensors');
    return module;
}