# webgpu-tensors

Experimental WebGPU/Js Tensors library. Inspired by Pytorch.

## Run examples

### Using Deno

```bash
deno run ./examples/ex1.ts
```

### Using Vite

```bash
npm run dev
```
> It will run on your browser using WebGPU if present, otherwise CPU.

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
