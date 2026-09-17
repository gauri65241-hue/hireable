import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { existsSync } from 'fs'
import path from 'path'

// process.cwd() = /frontend when running `npm run build` — no import.meta.url needed
const srcDir = path.resolve(process.cwd(), 'src')

// enforce:'pre' intercepts @/ imports before vite:resolve applies alias,
// so we own the full resolution including extension trial.
const resolveAtAlias = {
  name: 'resolve-at-alias',
  enforce: 'pre',
  resolveId(id) {
    if (!id.startsWith('@/')) return
    const subpath = id.slice(2)
    for (const ext of ['.ts', '.tsx', '.js', '.jsx']) {
      const full = path.join(srcDir, subpath + ext)
      if (existsSync(full)) return full
    }
    for (const ext of ['.ts', '.tsx', '.js', '.jsx']) {
      const full = path.join(srcDir, subpath, 'index' + ext)
      if (existsSync(full)) return full
    }
  },
}

export default defineConfig({
  plugins: [resolveAtAlias, react()],
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
