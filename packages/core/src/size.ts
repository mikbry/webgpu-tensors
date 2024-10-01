import { NestedArray, Size } from './types';

export function getShapeSize(dim: number[]): number {
  return dim.reduce((acc, s) => s * acc, 1);
}

export default class ShapeSize implements Size {
  data: number[];
  constructor(data: number[]) {
    this.data = data;
  }

  static createFromNestedArray(array: NestedArray<number>): ShapeSize {
    let a: number | NestedArray<number> = array;
    const shape = new ShapeSize([]);
    while (Array.isArray(a)) {
      shape.data = [...shape.data, a.length];
      a = a[0];
    }
    return shape;
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
