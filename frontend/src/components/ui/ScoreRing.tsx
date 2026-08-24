import { useEffect, useState } from 'react'

interface ScoreRingProps {
  score: number
  size?: number
  strokeWidth?: number
  animate?: boolean
  label?: string
}

export function ScoreRing({ score, size = 120, strokeWidth = 8, animate = true, label }: ScoreRingProps) {
  const [displayed, setDisplayed] = useState(animate ? 0 : score)
  const r = (size - strokeWidth) / 2
  const circ = 2 * Math.PI * r
  const offset = circ * (1 - displayed / 100)

  const color = score >= 70 ? 'var(--green)' : score >= 40 ? 'var(--amber)' : 'var(--red)'
  const glow  = score >= 70 ? 'rgba(34,211,160,.3)' : score >= 40 ? 'rgba(245,166,35,.3)' : 'rgba(240,97,79,.3)'

  useEffect(() => {
    if (!animate) { setDisplayed(score); return }
    let start: number | null = null
    const duration = 900
    const step = (ts: number) => {
      if (!start) start = ts
      const p = Math.min((ts - start) / duration, 1)
      const ease = 1 - Math.pow(1 - p, 3)
      setDisplayed(Math.round(score * ease))
      if (p < 1) requestAnimationFrame(step)
    }
    requestAnimationFrame(step)
  }, [score, animate])

  return (
    <div className="relative inline-flex items-center justify-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} style={{ transform: 'rotate(-90deg)' }}>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="var(--surface3)" strokeWidth={strokeWidth} />
        <circle
          cx={size / 2} cy={size / 2} r={r} fill="none"
          stroke={color} strokeWidth={strokeWidth}
          strokeDasharray={circ} strokeDashoffset={offset}
          strokeLinecap="round"
          style={{ transition: 'stroke-dashoffset 0.04s ease', filter: `drop-shadow(0 0 8px ${glow})` }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
        <span className="font-display font-black text-[var(--text)] leading-none" style={{ fontSize: size * 0.22 }}>{displayed}</span>
        {label && <span className="text-[var(--text3)] font-semibold mt-0.5" style={{ fontSize: size * 0.085 }}>{label}</span>}
      </div>
    </div>
  )
}
