import { defineConfig } from 'vite';
import path from 'path';
import { fileURLToPath } from 'url';

const filename = fileURLToPath(import.meta.url);
const dirname = path.dirname(filename);

export default defineConfig({
  resolve: {
    alias: {
      '@/webgpu-tensors': path.resolve(dirname, './packages/core/src'),
    },
  },
  plugins: [],
});
