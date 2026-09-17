import { useCallback, useRef } from 'react'

export type SSEEvent = Record<string, unknown>

export function useSSE() {
  const esRef = useRef<EventSource | null>(null)

  // NOTE: this hook is not on the critical path — Analyzer uses fetch streaming instead.
  // token is passed as a query param (safe for SSE since EventSource only supports GET).
  const connect = useCallback((
    url: string,
    onEvent: (e: SSEEvent) => void,
    onDone?: () => void,
    token?: string,
  ) => {
    esRef.current?.close()
    const urlWithAuth = token ? `${url}${url.includes('?') ? '&' : '?'}token=${encodeURIComponent(token)}` : url
    const es = new EventSource(urlWithAuth)
    esRef.current = es
    es.onmessage = (e) => {
      try { onEvent(JSON.parse(e.data as string) as SSEEvent) } catch { /* skip malformed */ }
    }
    es.addEventListener('close', () => { es.close(); onDone?.() })
    es.onerror = () => { es.close(); onDone?.() }
  }, [])

  const close = useCallback(() => { esRef.current?.close() }, [])

  return { connect, close }
}
