import { expect, test, describe, beforeEach } from 'vitest';
import t, { Tensors, DType } from '../src/webgpu-tensors';

describe('WebGPUTensors', () => {
  let tensors: Tensors;

  beforeEach(async () => {
    tensors = t;
    await t.init();
  });

  test('Create empty tensor', () => {
    const shape = [2, 3];
    const tensor = tensors.empty(shape);
    expect(tensor).toBeInstanceOf(Object);
    expect(tensor.shape.data).toEqual(shape);
    expect(tensor.dtype).toBe(DType.Float32);
  });

  test('Create ones tensor', async () => {
    const shape = [2, 2];
    const tensor = tensors.ones(shape);
    expect(tensor).toBeInstanceOf(Object);
    expect(tensor.shape.data).toEqual(shape);
    const data = await t.clone(tensor).readArray<number>();
    expect(data).toEqual([[1, 1], [1, 1]]);
  });

  test('Matrix multiplication', async () => {
    const a = tensors.tensor([[1, 2], [3, 4]]);
    const b = tensors.tensor([[5, 6], [7, 8]]);
    const result = tensors.matmul(a, b);
    const data = await t.clone(result).readArray<number>();
    expect(data).toEqual([[19, 22], [43, 50]]);
  });

  test('ReLU activation', async () => {
    const input = tensors.tensor([-2, -1, 0, 1, 2]);
    const result = tensors.relu(input);
    const data = await t.clone(result).readFloat32();
    expect(Array.from(data as Float32Array)).toEqual([0, 0, 0, 1, 2]);
  });
});
