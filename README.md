# webgpu-tensors

Experimental WebGPU/Js Tensors library. Inspired by Pytorch.

## Run examples

- [Example 0](./examples/ex0.ts) : just a tensors copy ([Rust native implementation](./examples/ex0.rs)).
- [Example 1](./examples/ex1.ts) : several tests.
- [Example 2](./examples/ex2.ts) : Create a simple 2-layer neural network (Typescript and [Pytorch implementation](./examples/ex2.py) to compare results)
- Example 3 : Import a csv dataset (TODO)
- Example x : Transformer (TODO)

### Using Deno

```bash
deno run ./examples/ex1.ts
```

### Using Vite

```bash
npm run dev
```
> It will run on your browser using WebGPU if present, otherwise CPU.

### Using Rust native

```bash
cargo run --example ex0
```

### Using Python

> Pytorch reference

```bash
python examples/ex2.py
```

## Run tests

### Using Deno

```bash
deno task test
```

> Deno coverage is not working: [Issue](https://github.com/denoland/deno/issues/25004)

### Using Vite

```bash
npm run test
```

> No WebGPU on Node.js

### Using Python

> Pytorch reference

```bash
python -m unittest tests/tensors_test.py
```
