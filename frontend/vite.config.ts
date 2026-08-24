import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: { alias: { '@': path.resolve(__dirname, './src') } },
  server: {
    port: 5173,
    proxy: {
      '/auth':          { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/resumes':       { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/jobs':          { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/analyze':       { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/results':       { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/ats':           { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/career':        { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/benchmark':     { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/health':        { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/ollama-models': { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/pdf':           { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/list_results':  { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/skill-match':   { target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/interview_prep':{ target: 'http://127.0.0.1:8000', changeOrigin: true },
      '/record_outcome':{ target: 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
  build: { outDir: '../frontend_dist' },
})
