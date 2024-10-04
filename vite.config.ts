import { defineConfig } from 'vite';
import path from 'path';
import { fileURLToPath } from 'url';
import wasm from 'vite-plugin-wasm';

const filename = fileURLToPath(import.meta.url);
const dirname = path.dirname(filename);

export default defineConfig({
  resolve: {
    alias: {
      '@/webgpu-tensors': path.resolve(dirname, './packages/core/src'),
      '@/webgpu-tensors-wasm': path.resolve(dirname, './crates/wasm'),
    },
  },
  plugins: [wasm()],
});
