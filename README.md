# webgpu-tensors

Experimental WebGPU Tensors

## Run examples using Deno

```bash
deno run --allow-read=. --allow-write=. --unstable-webgpu --unstable-sloppy-imports ./examples/ex1.ts
```

## Run Deno tests
```bash
deno task test
```
Deno coverage is not working: [Issue](https://github.com/denoland/deno/issues/25004)

## Run Python tests
```bash
python -m unittest tests/tensors_test.py
```