import { expect, test, describe, beforeEach } from 'vitest';
import { Tensors, DType } from '../src/webgpu-tensors';
import createJSTensors from '../src/js-tensors';

describe('JSTensors', () => {
  let tensors: Tensors;

  beforeEach(async () => {
    tensors = createJSTensors();
    await tensors.init();
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

  test('Create zeros tensor', async () => {
    const shape = [2, 3];
    const tensor = tensors.zeros(shape);
    expect(tensor).toBeInstanceOf(Object);
    expect(tensor.shape.data).toEqual(shape);
    const data = await t.clone(tensor).readArray<number>();
    expect(data).toEqual([[0, 0, 0], [0, 0, 0]]);
  });

  test('Create rand tensor', async () => {
    const shape = [2, 3];
    const tensor = tensors.rand(shape);
    expect(tensor).toBeInstanceOf(Object);
    expect(tensor.shape.data).toEqual(shape);
    const data = await t.clone(tensor).readArray<number>();
    data.flat().forEach(value => {
      expect(value).toBeGreaterThanOrEqual(0);
      expect(value).toBeLessThan(1);
    });
  });

  test('Create randn tensor', async () => {
    const shape = [1000];
    const tensor = tensors.randn(shape);
    expect(tensor).toBeInstanceOf(Object);
    expect(tensor.shape.data).toEqual(shape);
    const data = await t.clone(tensor).readArray<number>();
    const mean = data.reduce((sum, val) => sum + val, 0) / data.length;
    const variance = data.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / data.length;
    expect(mean).toBeCloseTo(0, 1);
    expect(Math.sqrt(variance)).toBeCloseTo(1, 1);
  });

  test('Matrix multiplication', async () => {
    const a = tensors.tensor([[1, 2], [3, 4]]);
    const b = tensors.tensor([[5, 6], [7, 8]]);
    const result = tensors.matmul(a, b);
    const data = await t.clone(result).readArray<number>();
    expect(data).toEqual([[19, 22], [43, 50]]);
  });

  test('Subtraction', async () => {
    const a = tensors.tensor([1, 2, 3]);
    const b = tensors.tensor([4, 5, 6]);
    const result = tensors.sub(a, b);
    const data = await t.clone(result).readArray<number>();
    expect(data).toEqual([-3, -3, -3]);
  });

  test('Power', async () => {
    const a = tensors.tensor([1, 2, 3]);
    const result = tensors.pow(a, 2);
    const data = await t.clone(result).readArray<number>();
    expect(data).toEqual([1, 4, 9]);
  });

  test('Multiplication', async () => {
    const a = tensors.tensor([1, 2, 3]);
    const b = tensors.tensor([4, 5, 6]);
    const result = tensors.mul(a, b);
    const data = await t.clone(result).readArray<number>();
    expect(data).toEqual([4, 10, 18]);
  });

  test('Greater than', async () => {
    const a = tensors.tensor([1, 2, 3]);
    const result = tensors.gt(a, 2);
    const data = await t.clone(result).readArray<number>();
    expect(data).toEqual([0, 0, 1]);
  });

  test('Transpose', async () => {
    const a = tensors.tensor([[1, 2, 3], [4, 5, 6]]);
    const result = tensors.transpose(a);
    const data = await t.clone(result).readArray<number>();
    expect(data).toEqual([[1, 4], [2, 5], [3, 6]]);
  });

  test('Maximum', async () => {
    const a = tensors.tensor([-1, 0, 1]);
    const result = tensors.maximum(a, 0);
    const data = await t.clone(result).readArray<number>();
    expect(data).toEqual([0, 0, 1]);
  });

  test('ReLU activation', async () => {
    const input = tensors.tensor([-2, -1, 0, 1, 2]);
    const result = tensors.relu(input);
    const data = await t.clone(result).readFloat32();
    expect(Array.from(data as Float32Array)).toEqual([0, 0, 0, 1, 2]);
  });

  test('Mean', async () => {
    const a = tensors.tensor([1, 2, 3, 4, 5]);
    const result = tensors.mean(a);
    const data = await t.clone(result).readArray<number>();
    expect(data[0]).toBeCloseTo(3);
  });

  test('Softmax', async () => {
    const a = tensors.tensor([1, 2, 3]);
    const result = tensors.softmax(a);
    const data = await t.clone(result).readArray<number>();
    const expectedSoftmax = [0.0900, 0.2447, 0.6652];
    data.forEach((value, index) => {
      expect(value).toBeCloseTo(expectedSoftmax[index], 4);
    });
  });

  test('Sigmoid', async () => {
    const a = tensors.tensor([-2, -1, 0, 1, 2]);
    const result = tensors.sigmoid(a);
    const data = await t.clone(result).readArray<number>();
    const expectedSigmoid = [0.1192, 0.2689, 0.5000, 0.7311, 0.8808];
    data.forEach((value, index) => {
      expect(value).toBeCloseTo(expectedSigmoid[index], 4);
    });
  });

  test('Max', async () => {
    const a = tensors.tensor([1, 5, 3, 2, 4]);
    const result = tensors.max(a);
    const data = await t.clone(result).readArray<number>();
    expect(data[0]).toBe(5);
  });

  test('Sum', async () => {
    const a = tensors.tensor([1, 2, 3, 4, 5]);
    const result = tensors.sum(a);
    const data = await t.clone(result).readArray<number>();
    expect(data[0]).toBe(15);
  });
});
