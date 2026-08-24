import { cn } from '@/lib/utils'

interface ScoreBadgeProps { score: number; className?: string }

export function ScoreBadge({ score, className }: ScoreBadgeProps) {
  const cls = score >= 70
    ? 'bg-[var(--green-bg)] text-[var(--green)] border-[var(--green-border)]'
    : score >= 40
    ? 'bg-[var(--amber-bg)] text-[var(--amber)] border-[var(--amber-border)]'
    : 'bg-[var(--red-bg)] text-[var(--red)] border-[var(--red-border)]'
  return (
    <span className={cn('inline-flex items-center justify-center px-2 py-0.5 rounded border text-[11.5px] font-bold font-mono', cls, className)}>
      {score}
    </span>
  )
}
