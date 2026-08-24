import { cn } from '@/lib/utils'
import { type ButtonHTMLAttributes, forwardRef } from 'react'

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost' | 'danger'
  size?: 'sm' | 'md'
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'primary', size = 'md', ...props }, ref) => (
    <button
      ref={ref}
      className={cn(
        'inline-flex items-center gap-1.5 font-semibold rounded-[var(--radius-sm)] transition-all duration-100 cursor-pointer border-0 font-body disabled:opacity-40 disabled:cursor-not-allowed active:scale-[.98] whitespace-nowrap',
        size === 'sm' ? 'px-2.5 py-1 text-[11.5px]' : 'px-4 py-2 text-[13px]',
        variant === 'primary'   && 'bg-[var(--accent)] text-white shadow-[0_1px_2px_rgba(0,0,0,.4),inset_0_1px_0_rgba(255,255,255,.08)] hover:bg-[var(--accent-light)] hover:shadow-[0_0_0_3px_var(--accent-glow)]',
        variant === 'secondary' && 'bg-[var(--surface2)] text-[var(--text)] border border-[var(--border2)] shadow-[var(--shadow)] hover:bg-[var(--surface3)] hover:border-[var(--border3)]',
        variant === 'ghost'     && 'bg-transparent text-[var(--text2)] border border-transparent hover:bg-[var(--surface2)] hover:text-[var(--text)]',
        variant === 'danger'    && 'bg-[var(--red)] text-white hover:opacity-90 hover:shadow-[0_0_0_3px_var(--red-bg)]',
        className,
      )}
      {...props}
    />
  ),
)
Button.displayName = 'Button'
