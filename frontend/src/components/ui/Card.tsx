import { cn } from '@/lib/utils'
import { type HTMLAttributes } from 'react'

export function Card({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        'relative bg-[var(--surface)] border border-[var(--border)] rounded-[var(--radius)] p-[18px] transition-colors duration-150 overflow-hidden hover:border-[var(--border2)]',
        'before:absolute before:inset-x-0 before:top-0 before:h-px before:bg-gradient-to-r before:from-transparent before:via-[rgba(99,102,241,.18)] before:to-transparent before:pointer-events-none',
        className,
      )}
      {...props}
    />
  )
}

export function CardHeader({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return <div className={cn('flex items-center justify-between mb-3.5', className)} {...props} />
}

export function CardTitle({ className, ...props }: HTMLAttributes<HTMLHeadingElement>) {
  return <h3 className={cn('text-[12.5px] font-bold tracking-[-0.1px] text-[var(--text)]', className)} {...props} />
}
