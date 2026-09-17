import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'
import { existsSync } from 'fs'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const srcDir = path.resolve(__dirname, 'src')

// Rollup skips extension resolution for absolute IDs returned by alias.
// This plugin adds .ts/.tsx so @/lib/api resolves to @/lib/api.ts.
const resolveAbsoluteExtensions = {
  name: 'resolve-absolute-extensions',
  resolveId(id) {
    if (id.startsWith(srcDir) && !path.extname(id) && !existsSync(id)) {
      for (const ext of ['.ts', '.tsx']) {
        if (existsSync(id + ext)) return id + ext
      }
    }
  },
}

export default defineConfig({
  plugins: [react(), resolveAbsoluteExtensions],
  resolve: {
    alias: { '@': srcDir },
  },
  server: {
    port: 5173,
    proxy: {
      '/auth':           { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/resumes':        { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/jobs':           { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/analyze':        { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/results':        { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/ats':            { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/career':         { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/benchmark':      { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/health':         { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/ollama-models':  { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/pdf':            { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/list_results':   { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/skill-match':    { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/interview_prep': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/record_outcome': { target: 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
  build: { outDir: 'dist' },
})
