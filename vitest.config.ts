import { configDefaults, defineConfig } from 'vitest/config';
import config from './vite.config';

export default defineConfig({
  ...config,
  test: {
    exclude: [...configDefaults.exclude, './examples/*.ts'],
  },
});
