import { getShapeSize } from './size';
import { Float32NestedArray, NestedArray } from './types';

export function stridedToNestedArray<T>(
  buffer: T[],
  dim: number[],
  offset = 0,
  depth = 0,
): NestedArray<T> {
  if (dim.length === 1) {
    const length = getShapeSize(dim);
    return buffer.slice(offset, offset + length) as NestedArray<T>;
  }
  const array: NestedArray<T> = [];
  for (let n = 0; n < dim[0]; n++) {
    const nestedShape = [...dim];
    nestedShape.shift();
    const length = getShapeSize(nestedShape);
    const i = length * n;
    const nestedArray = stridedToNestedArray(buffer, nestedShape, offset + i, depth + 1);
    array.push(Array.from(nestedArray) as NestedArray<T>);
  }
  return array;
}

export function stridedToNestedFloat32Array<T>(
  buffer: Array<T>,
  dim: number[],
  offset = 0,
  depth = 0,
): Float32NestedArray {
  if (dim.length === 1) {
    const length = getShapeSize(dim);
    return new Float32Array(buffer.slice(offset, offset + length) as number[]);
  }
  const array: Float32NestedArray = [];
  for (let n = 0; n < dim[0]; n++) {
    const nestedShape = [...dim];
    nestedShape.shift();
    const length = getShapeSize(nestedShape);
    const i = length * n;
    const nestedArray = stridedToNestedFloat32Array(buffer, nestedShape, offset + i, depth + 1);
    array.push(nestedArray);
  }
  return array;
}
