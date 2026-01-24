import { defineConfig } from 'vitest/config';
import path from 'path';

export default defineConfig({
    test: {
        environment: 'jsdom',
        globals: true,
        setupFiles: ['./tests/setup.ts'],
        include: ['**/*.test.ts'],
        exclude: ['**/node_modules/**', '**/e2e/**', '**/dist/**'],
        alias: {
            '@engine': path.resolve(__dirname, './src'),
        },
    },
});
