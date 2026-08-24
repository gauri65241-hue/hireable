import { useCallback, useRef } from 'react'

export type SSEEvent = Record<string, unknown>

export function useSSE() {
  const esRef = useRef<EventSource | null>(null)

  const connect = useCallback((
    url: string,
    onEvent: (e: SSEEvent) => void,
    onDone?: () => void,
  ) => {
    esRef.current?.close()
    const es = new EventSource(url)
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
